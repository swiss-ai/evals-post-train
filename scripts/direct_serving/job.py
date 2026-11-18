#!/usr/bin/env python3
"""One Slurm job that serves a model and runs evals against it, on the same nodes.

The model is served by model-launch's own job script (one `srun` step per node, Ray
across the nodes of a replica, so a model on several nodes works), with OpenTela, metrics
and telemetry off: nothing registers on the serving gateway. Only the end of that script
is ours. Where model-launch waits for the server to exit, this job waits for the server
to answer on the head node's localhost, runs each eval command in order as a step beside
the server, and ends:

    sbatch
    |- srun   model server(s)         model-launch's script
    '- srun   eval command 1, 2, ..   in the eval image, against http://127.0.0.1:8080/v1

The evals call the server directly, so the gateway's 503s, rate limits and header timeout
never reach them, and the model is loaded once for all of them. Every eval command sees
the server's URL as $EVAL_API_BASE_URL. The job ends with status 1 if the server died or
any command failed, and the other commands still run.

The commands are the launchers of any framework here (lm-eval's launch_evaluations.sh,
run_inspect_eval.sh, run_harbor_eval.sh, run_evalscope_eval.sh). Each runs with `bash -c`
from this checkout, with `sbatch` and `srun` that run in place (inline_slurm/), inside the prebuilt eval image (docs/lm-eval-harness.md): the image of the last
commit that changed what it is built from, which CI published (.github/workflows/
eval-image.yml). The job writes the EDF that starts it; the checkout and the usual scratch
directories are mounted (more with `--mount`).

Print the job script (to submit it another way):

    python -m scripts.direct_serving.job --model <checkpoint dir> --nodes 2 \
        --environment containers/env_vllm_apertus2.toml --framework-args "..." \
        --eval 'scripts/inspect_ai/run_inspect_eval.sh --task gsm8k --model m --api-base-url "$EVAL_API_BASE_URL"'

`--submit` runs sbatch on it (on a cluster login node).
"""

import argparse
import getpass
import grp
import json
import os
import re
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from swiss_ai_model_launch.launchers import framework as ml_framework
from swiss_ai_model_launch.launchers.launch_args import FRAMEWORK_PORT, LaunchArgs
from swiss_ai_model_launch.launchers.topology import Topology

HEALTH_URL = f"http://127.0.0.1:{FRAMEWORK_PORT}/health"
API_BASE_URL = f"http://127.0.0.1:{FRAMEWORK_PORT}/v1"
HEALTH_POLL_SECONDS = 10

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_REPOSITORY = "swiss-ai/evals-post-train-eval"
# What the image is built from: the paths of .github/workflows/eval-image.yml. A commit
# that changes none of them runs in the image of the last one that did.
IMAGE_INPUTS = ("containers/Dockerfile.eval", "requirements", ".github/workflows/eval-image.yml")
# sbatch and srun that run their script or command right here: a command that submits
# Slurm jobs of its own (lm-eval's launcher) runs inline, in the job's allocation, where
# 127.0.0.1 is the server.
INLINE_SLURM_DIR = REPO_ROOT / "scripts" / "direct_serving" / "inline_slurm"
# What the eval image sees: this checkout, and the places runs read and write (as
# containers/env.toml mounts them).
DEFAULT_MOUNTS = (
    str(REPO_ROOT),
    "/users/${USER}",
    "/capstor/scratch/cscs/${USER}",
    "/iopsstor/scratch/cscs/${USER}",
)


def build_launch_args(
    *,
    model: str,
    served_model_name: str,
    nodes: int,
    environment: str,
    framework: str,
    framework_args: str,
    account: str,
    partition: str,
    time: str,
    job_name: str,
) -> LaunchArgs:
    """model-launch's LaunchArgs for a server that nothing outside the job reaches."""
    return LaunchArgs(
        job_name=job_name,
        served_model_name=served_model_name,
        account=account,
        partition=partition,
        topology=Topology(replicas=1, nodes_per_replica=nodes),
        time=time,
        environment=str(Path(environment).resolve()),
        framework=framework,
        # What model-launch's own launch puts in front of the caller's arguments.
        framework_args=(
            f"--model {shlex.quote(model)} "
            f"--served-model-name {shlex.quote(served_model_name)} "
            f"--host 0.0.0.0 {framework_args}"
        ).strip(),
        disable_opentela=True,
        disable_metrics=True,
        disable_dcgm_exporter=True,
        telemetry_endpoint=None,
    )


def eval_image_commit(repo_root: Path = REPO_ROOT) -> str:
    """The commit whose image these scripts run in: the last one that changed the image's inputs."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "log", "-1", "--format=%H", "--", *IMAGE_INPUTS],
        capture_output=True,
        text=True,
        check=False,
    )
    commit = result.stdout.strip()
    if result.returncode != 0 or not commit:
        raise RuntimeError(f"cannot tell which eval image {repo_root} runs in: {result.stderr.strip()}")
    return commit


def image_published(commit: str) -> bool | None:
    """Whether ghcr has the eval image of `commit` (an anonymous manifest request), or None
    when ghcr could not be asked."""
    try:
        with urllib.request.urlopen(
            f"https://ghcr.io/token?service=ghcr.io&scope=repository:{IMAGE_REPOSITORY}:pull", timeout=15
        ) as response:
            token = json.load(response)["token"]
        request = urllib.request.Request(
            f"https://ghcr.io/v2/{IMAGE_REPOSITORY}/manifests/{commit}",
            method="HEAD",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.oci.image.index.v1+json, "
                "application/vnd.docker.distribution.manifest.list.v2+json",
            },
        )
        with urllib.request.urlopen(request, timeout=15):
            return True
    except urllib.error.HTTPError as error:
        return False if error.code == 404 else None
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return None


def eval_edf(image_commit: str, mounts: list[str]) -> str:
    """The EDF that starts the eval image of `image_commit`."""
    listed = ",\n".join(f'    "{mount}"' for mount in dict.fromkeys(mounts))
    return (
        f'image = "ghcr.io#{IMAGE_REPOSITORY}:{image_commit}"\n\n'
        f"mounts = [\n{listed}\n]\n\n"
        "writable = true\n\n"
        '[env]\nPMIX_MCA_psec = "native"\n\n'
        '[annotations]\ncom.hooks.cxi.enabled = "false"\n'
    )


def serving_half(launch_args: LaunchArgs) -> str:
    """model-launch's job script without its last section, which waits for the server
    to exit (ours waits for the evals)."""
    master = ml_framework.render_master(launch_args)
    their_footer = ml_framework._render_footer()
    if not master.endswith(f"{their_footer}\n"):
        # A model-launch whose footer changed: stop here rather than submit a job that
        # never runs the evals.
        raise RuntimeError(
            "the installed model-launch renders a job script whose last section is not the "
            "one this launcher replaces"
        )
    return master.removesuffix(f"{their_footer}\n")


def _fill(template: str, **values: str) -> str:
    """`template` with each @@NAME@@ replaced, in one pass (a value is never re-expanded)."""
    return re.sub(r"@@(\w+)@@", lambda match: values[match.group(1)], template)


# Raw text with @@NAME@@ marks, so it reads as bash. `nodes`, `critical_pids` and
# `health_checker_pid` come from model-launch's script above it.
_FOOTER = r"""
cleanup() {
    if [[ -n "$health_checker_pid" ]]; then
        kill "$health_checker_pid" 2>/dev/null || true
    fi
    if (( ${#critical_pids[@]} > 0 )); then
        kill "${critical_pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# The eval image's EDF, next to the job's logs (the job runs from its working directory).
eval_edf="$PWD/eval_image_${SLURM_JOB_ID}.toml"
cat > "$eval_edf" <<'EVAL_EDF_EOF'
@@EDF@@EVAL_EDF_EOF

server_alive() {
    local pid
    for pid in "${critical_pids[@]}"; do
        kill -0 "$pid" 2>/dev/null || return 1
    done
}

server_log_tail() {
    tail -n 60 "logs/${SLURM_JOB_ID}/replica_0.err" "logs/${SLURM_JOB_ID}/replica_0.out" >&2 || true
}

ready() {
    # Never through a proxy, and never blocking the loop that watches the server.
    curl -sf --noproxy '*' --max-time 5 @@HEALTH_URL@@ >/dev/null
}

echo "Waiting for the model server on ${nodes[0]} (@@HEALTH_URL@@)"
waited=0
until ready; do
    if ! server_alive; then
        echo "The model server exited before it was ready:" >&2
        server_log_tail
        exit 1
    fi
    if (( waited >= @@READY_TIMEOUT@@ )); then
        echo "The model server was not ready after ${waited}s" >&2
        exit 1
    fi
    sleep @@POLL@@
    waited=$((waited + @@POLL@@))
done
echo "The model server is ready after ${waited}s"

export EVAL_API_BASE_URL=@@API_BASE_URL@@
# The key the runners send to --api-base-url (TARGET_API_KEY): this server has none, but
# they want one. Judge keys (CSCS_SERVING_API) are left alone.
export TARGET_API_KEY=local
failed=0

# run_eval <number> <command>: the command in the eval image on the head node, beside
# the server. Stopped if the server dies, which it would otherwise call until the time limit.
run_eval() {
    echo "== eval $1: $2"
    srun --overlap --nodes=1 --ntasks=1 --nodelist="${nodes[0]}" \
        --environment="$eval_edf" bash -c "$2" &
    local eval_pid=$! rc=0
    while kill -0 "$eval_pid" 2>/dev/null; do
        if ! server_alive; then
            echo "The model server exited during eval $1; stopping it" >&2
            server_log_tail
            kill "$eval_pid" 2>/dev/null || true
            rc=1
            break
        fi
        sleep 5
    done
    wait "$eval_pid" || { (( rc )) || rc=$?; }
    echo "== eval $1 finished with code $rc at $(date)"
    return "$rc"
}

@@EVALS@@
exit "$failed"
"""


def footer(
    edf: str, evals: list[str], ready_timeout: int, repo_root: Path = REPO_ROOT, inline_slurm_dir: Path = INLINE_SLURM_DIR
) -> str:
    prefix = f"cd {shlex.quote(str(repo_root))} && export PATH={shlex.quote(str(inline_slurm_dir))}:$PATH && "
    calls = "\n".join(
        f"run_eval {number} {shlex.quote(prefix + command)} || failed=1"
        for number, command in enumerate(evals, start=1)
    )
    return _fill(
        _FOOTER,
        HEALTH_URL=HEALTH_URL,
        API_BASE_URL=shlex.quote(API_BASE_URL),
        READY_TIMEOUT=str(ready_timeout),
        POLL=str(HEALTH_POLL_SECONDS),
        EDF=edf,
        EVALS=calls,
    )


def render(
    launch_args: LaunchArgs,
    *,
    image_commit: str,
    mounts: list[str],
    evals: list[str],
    ready_timeout: int,
    reservation: str | None = None,
    qos: str | None = None,
) -> str:
    """The whole job script, `#SBATCH` header included."""
    options = launch_args.to_sbatch_args(reservation=reservation)
    if qos:
        options.append(f"--qos={qos}")
    header = [f"#SBATCH {arg}" for arg in options]
    return (
        "\n".join(["#!/bin/bash", *header])
        + "\n"
        + serving_half(launch_args)
        + footer(eval_edf(image_commit, mounts), evals, ready_timeout)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve a model and run evals against it in one Slurm job.",
        epilog="Run with `python -m scripts.direct_serving.job`.",
    )
    parser.add_argument("--model", required=True, help="Checkpoint directory or model path the server loads.")
    parser.add_argument("--served-model-name", help="Name the server answers to (default: the model's basename).")
    parser.add_argument("--nodes", type=int, default=1, help="Nodes for the one replica (default 1).")
    parser.add_argument("--environment", required=True, help="EDF of the serving image.")
    parser.add_argument("--framework", default="vllm", help="Serving framework (default vllm).")
    parser.add_argument("--framework-args", default="", help="Extra arguments for the serving command.")
    parser.add_argument(
        "--eval-image-commit",
        help="Commit of the eval image the commands run in (default: the last commit here that "
        "changed what the image is built from).",
    )
    parser.add_argument(
        "--mount",
        action="append",
        default=[],
        help="A directory to mount in the eval image besides the defaults, repeatable.",
    )
    parser.add_argument(
        "--eval",
        dest="evals",
        action="append",
        required=True,
        metavar="COMMAND",
        help="A command to run against the server, repeatable (run in order). "
        "The server's URL is $EVAL_API_BASE_URL.",
    )
    parser.add_argument("--account", help="Slurm account (default: your primary group).")
    parser.add_argument("--partition", default="normal")
    parser.add_argument("--reservation")
    parser.add_argument("--qos", help="Slurm QOS, e.g. highprio (with --partition highprio).")
    parser.add_argument("--time", default="04:00:00", help="Time limit: the load plus every eval (HH:MM:SS).")
    parser.add_argument("--job-name", help="Slurm job name (default: serve-eval-<user>).")
    parser.add_argument(
        "--ready-timeout",
        type=int,
        default=3600,
        help="Seconds to wait for the server before giving up (default 3600).",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="sbatch the script from --workdir instead of printing it.",
    )
    parser.add_argument(
        "--workdir",
        default=str(Path.home() / ".sml"),
        help="Where --submit writes the script and the job's logs/ go (default ~/.sml).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    user = getpass.getuser()
    launch_args = build_launch_args(
        model=args.model,
        served_model_name=args.served_model_name or os.path.basename(args.model.rstrip("/")),
        nodes=args.nodes,
        environment=args.environment,
        framework=args.framework,
        framework_args=args.framework_args,
        account=args.account or grp.getgrgid(os.getgid()).gr_name,
        partition=args.partition,
        time=args.time,
        job_name=args.job_name or f"serve-eval-{user}",
    )
    image_commit = args.eval_image_commit or eval_image_commit()
    published = image_published(image_commit)
    if published is False:
        sys.exit(
            f"ghcr.io/{IMAGE_REPOSITORY}:{image_commit} does not exist: push the commit that "
            "changed the image's inputs (CI builds it), or name another with --eval-image-commit."
        )
    if published is None:
        print(f"warning: could not check that the eval image {image_commit} exists", file=sys.stderr)
    script = render(
        launch_args,
        image_commit=image_commit,
        mounts=[*DEFAULT_MOUNTS, *args.mount],
        evals=args.evals,
        ready_timeout=args.ready_timeout,
        reservation=args.reservation,
        qos=args.qos,
    )
    if not args.submit:
        sys.stdout.write(script)
        return 0
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / f"job_{launch_args.job_name}.sh"
    path.write_text(script)
    path.chmod(0o755)
    return subprocess.run(["sbatch", "--chdir", str(workdir), str(path)], check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
