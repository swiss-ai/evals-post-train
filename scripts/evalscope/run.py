"""Run SWE-bench (Verified) and/or SWE-bench Pro through EvalScope (see run_evalscope_eval.sh).

Builds an EvalScope TaskConfig from the flags, routes SWE-bench's scoring through
the sandbox service (swe_bench_scoring.install()) and, with --sandbox k8s, every
sandbox to a Kubernetes pod (scripts/common/k8s_sandbox.py: install()), or with --sandbox
opensandbox to a Kata microVM (scripts/common/opensandbox_sandbox.py: install()), runs it, and writes
<work-dir>/summary.json: the scores plus everything needed to reproduce the run
(evalscope version, dataset revisions, SWE-bench_Pro-os commit, sandbox engine).

--gold skips the agent and the model: each sample's reference patch is the
prediction (install_gold()), so a run checks the scoring pipeline itself --
every sample should come out resolved.

A sample whose evaluation itself failed (sandbox, cluster or network trouble) is
scored unresolved by EvalScope, like a wrong patch. Those are counted in
summary.json's evaluation_errors, and the run exits non-zero if there are any:
its score is not a model score.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import sys
import uuid
from typing import Any, Dict, List, Optional

# Our benchmark names -> the EvalScope benchmark, its pinned Hugging Face dataset revision
# (--dataset-revision overrides) and the extra_params we always set.
BENCHMARKS: Dict[str, Dict[str, Any]] = {
    # SWE-bench Verified (500 tasks) as a bash agent loop, the same protocol as swe_bench_pro.
    # Images are the prebuilt swebench/sweb.eval.x86_64.* ones from DockerHub: EvalScope's
    # default builds them locally, which needs a Docker daemon (and arm64 builds differ).
    "swe_bench": {
        "evalscope": "swe_bench_verified_agentic",
        "revision": "c104f840cc67f8b6eec6f759ebc8b2693d585d4a",  # princeton-nlp/SWE-bench_Verified, 2025-02-18
        "extra_params": {"build_docker_images": False, "force_arch": "x86_64"},
    },
    # EvalScope implements SWE-bench Pro *v1* (731 tasks, jefzda/sweap-images, the
    # SWE-bench_Pro-os v1 run scripts it pins at ca10a60). Since 2026-09-22 the dataset's main
    # branch is V2 (642 Harbor tasks, other images and test setup): v1 scoring against it fails
    # even gold patches (its before_repo_set_cmd applies a /tests/test_patch.patch v1 images do
    # not have). 7ab5114 is the last pre-V2 commit (2026-02-23).
    "swe_bench_pro": {
        "evalscope": "swe_bench_pro",
        "revision": "7ab5114912baf22bb098818e604c02fe7ad2c11f",
        "extra_params": {},
    },
}

# EvalScope applies its own default for API models (DEFAULT_TEXT_GEN_SERVICE_CONFIG, greedy)
# only when generation_config is empty. The model's sampling settings are therefore always set:
# the defaults of configs/sampling_defaults.json, then --generation-config over them, then the
# sampling flags (see generation_config()).

# --sandbox k8s defaults: ms-enclave's 4g OOM-kills SWE-bench Pro's element-web and teleport
# tests, while an 8 GiB request often does not schedule on Breithorn's amd64 nodes. So the
# limit is 8g and the request (SANDBOX_K8S_MEMORY_REQUEST) 4g, unless memory_limit is given.
K8S_SANDBOX_DEFAULTS: Dict[str, Any] = {"memory_limit": "8g"}
K8S_MEMORY_REQUEST = "4g"
# --sandbox opensandbox: the same 8g, as the microVM's memory (there is no separate request).
OPENSANDBOX_SANDBOX_DEFAULTS: Dict[str, Any] = {"memory_limit": "8g"}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", required=True, help=f"Comma-separated, from: {', '.join(BENCHMARKS)}.")
    p.add_argument("--model", default=None, help="Served model name (not with --gold).")
    p.add_argument("--api-base-url", default=None, help="OpenAI-compatible /v1 root (not with --gold).")
    p.add_argument("--gold", action="store_true", help="Score the reference patches instead of running a model.")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--sandbox", choices=("docker", "k8s", "opensandbox"), default="docker")
    p.add_argument("--sandbox-config", default="{}", help="JSON merged into every sandbox's config.")
    p.add_argument(
        "--agent-network",
        choices=("allowlist", "public"),
        default="allowlist",
        help="--sandbox opensandbox: the agent loop's sandboxes reach only package indexes "
        "(allowlist, the default; scripts/common/swe_bench_egress.py) or the whole internet "
        "(public). Scoring keeps its network. Other sandboxes are always public.",
    )
    p.add_argument("--limit", type=float, default=None)
    p.add_argument("--shuffle", action="store_true", help="Shuffle (seed 42) before --limit.")
    p.add_argument("--max-concurrency", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--strategy", default=None)
    p.add_argument("--command-timeout", type=float, default=None)
    p.add_argument("--generation-config", default="{}", help="JSON merged over the sampling defaults.")
    for flag in ("temperature", "top-p", "top-k", "min-p", "repetition-penalty"):
        p.add_argument("--" + flag, default=None, help="Sampling setting, or 'none' to leave it out.")
    p.add_argument("--sampling-profile", default=None, help="Profile of the sampling config (e.g. greedy).")
    p.add_argument("--sampling-config", default=None, help="Sampling config (default: configs/sampling_defaults.json).")
    p.add_argument("--dataset-revision", action="append", default=[], help="<benchmark>=<revision>.")
    p.add_argument("--extra-param", action="append", default=[], help="key=value (JSON value or string).")
    p.add_argument("--api-timeout", type=float, default=None)
    args = p.parse_args(argv)
    if not args.gold and not (args.model and args.api_base_url):
        p.error("--model and --api-base-url are required (unless --gold)")
    if args.sandbox == "opensandbox" and not os.environ.get("OPENSANDBOX_API_KEY"):
        p.error("--sandbox opensandbox needs OPENSANDBOX_API_KEY")
    return args


def agent_allowed_hosts(args: argparse.Namespace) -> Optional[tuple]:
    """What the agent loop's sandboxes may reach with --sandbox opensandbox (None: everything):
    package indexes (scripts/common/swe_bench_egress.py) and, as Harbor adds it, the model
    endpoint's host, for an agent that runs inside the sandbox and calls the model from there
    (EvalScope's external runners; they also install themselves from other hosts, so they need
    --agent-network public). The default strategies call the model from this process."""
    if args.agent_network != "allowlist":
        return None
    from urllib.parse import urlsplit

    from scripts.common.swe_bench_egress import AGENT_ALLOWED_HOSTS

    host = urlsplit(args.api_base_url).hostname if args.api_base_url else None
    return AGENT_ALLOWED_HOSTS + ((host,) if host and host not in AGENT_ALLOWED_HOSTS else ())


def parse_pairs(pairs: List[str], flag: str, as_json: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"{flag} needs key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        try:
            out[key] = json.loads(value) if as_json else value
        except json.JSONDecodeError:
            out[key] = value
    return out


def selected_benchmarks(value: str) -> List[str]:
    names = [b.strip() for b in value.split(",") if b.strip()]
    unknown = [n for n in names if n not in BENCHMARKS]
    if unknown or not names:
        raise SystemExit(f"--benchmark must be one or more of {', '.join(BENCHMARKS)}; got {value!r}")
    return names


def resolved_revisions(names: List[str], overrides: List[str]) -> Dict[str, str]:
    given = parse_pairs(overrides, "--dataset-revision", as_json=False)
    stray = sorted(set(given) - set(names))
    if stray:
        raise SystemExit(f"--dataset-revision for a benchmark that is not selected: {', '.join(stray)}")
    return {n: given.get(n, BENCHMARKS[n]["revision"]) for n in names}


def sandbox_config(args: argparse.Namespace) -> Dict[str, Any]:
    """--sandbox-config, over the --sandbox k8s or opensandbox defaults."""
    given = json.loads(args.sandbox_config)
    defaults = {"k8s": K8S_SANDBOX_DEFAULTS, "opensandbox": OPENSANDBOX_SANDBOX_DEFAULTS}.get(args.sandbox, {})
    return {**defaults, **given}


def environment_extra_params(name: str) -> Dict[str, Any]:
    """extra_params the environment supplies: the eval image checks SWE-bench_Pro-os out at
    EvalScope's pinned commit (EVALSCOPE_SWE_BENCH_PRO_REPO), so nothing is cloned at run time."""
    repo = os.environ.get("EVALSCOPE_SWE_BENCH_PRO_REPO")
    if name == "swe_bench_pro" and repo:
        return {"swe_bench_pro_repo_path": repo}
    return {}


def generation_config(args: argparse.Namespace) -> Dict[str, Any]:
    """The model's generation_config: sampling defaults, --generation-config over them, flags over both.

    top_k, top_p, temperature and repetition_penalty are EvalScope fields; min_p goes in extra_body,
    which the OpenAI-compatible server reads. 'none' on a flag removes that setting.
    """
    from scripts.common import sampling

    def to_config(values: Dict[str, Any]) -> Dict[str, Any]:
        cfg = {key: value for key, value in values.items() if key != "min_p"}
        if "min_p" in values:
            cfg["extra_body"] = {"min_p": values["min_p"]}
        return cfg

    try:
        config = sampling.read_config(args.sampling_config)
        explicit = {
            name: sampling.parse_value(name, getattr(args, name))
            for name in sampling.SAMPLERS
            if getattr(args, name) is not None
        }
        cfg = to_config(sampling.resolve("evalscope", config=config, profile=args.sampling_profile))
        user = json.loads(args.generation_config)
        cfg.update({key: value for key, value in user.items() if key != "extra_body"})
        if "extra_body" in user:
            cfg["extra_body"] = {**cfg.get("extra_body", {}), **user["extra_body"]}
        for name, value in explicit.items():
            if name == "min_p":
                body = {key: val for key, val in cfg.get("extra_body", {}).items() if key != "min_p"}
                cfg["extra_body"] = {**body, "min_p": value} if value is not None else body
                if not cfg["extra_body"]:
                    del cfg["extra_body"]
            elif value is None:
                cfg.pop(name, None)
            else:
                cfg[name] = value
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    chosen = bool(explicit or args.sampling_profile or "temperature" in json.loads(args.generation_config))
    if chosen and (warning := sampling.greedy_warning(cfg)):
        print(warning, file=sys.stderr)
    return cfg


def build_task_config(args: argparse.Namespace, names: List[str], revisions: Dict[str, str]) -> Dict[str, Any]:
    extra_params = parse_pairs(args.extra_param, "--extra-param", as_json=True)
    dataset_args = {
        BENCHMARKS[n]["evalscope"]: {
            "hf_dataset_revision": revisions[n],
            "extra_params": {**BENCHMARKS[n]["extra_params"], **environment_extra_params(n), **extra_params},
            **({"shuffle": True} if args.shuffle else {}),
        }
        for n in names
    }

    agent: Dict[str, Any] = {}
    if args.strategy:
        agent["strategy"] = args.strategy
    if args.max_steps:
        agent["max_steps"] = args.max_steps
    if args.command_timeout:
        agent["command_timeout"] = args.command_timeout

    if args.gold:
        # install_gold() never calls the model; mock_llm keeps EvalScope from needing an endpoint.
        model_cfg: Dict[str, Any] = {"model": "gold", "eval_type": "mock_llm"}
    else:
        model_cfg = {
            "model": args.model,
            "api_url": args.api_base_url,
            "api_key": os.environ.get("EVALSCOPE_API_KEY") or "EMPTY",
            "eval_type": "openai_api",
        }
    cfg: Dict[str, Any] = {
        **model_cfg,
        "datasets": [BENCHMARKS[n]["evalscope"] for n in names],
        "dataset_args": dataset_args,
        "dataset_hub": "huggingface",
        "eval_batch_size": args.max_concurrency,
        "generation_config": generation_config(args),
        "work_dir": args.work_dir,
        "no_timestamp": True,
        # Every sandbox (agent loop and scorer) reads this; the engine stays "docker"
        # either way -- with --sandbox k8s or opensandbox, the Docker manager itself is swapped out.
        "sandbox": {"enabled": True, "engine": "docker", "default_config": sandbox_config(args)},
    }
    if agent:
        # Only explicitly given fields: EvalScope keeps the benchmark's own
        # strategy/max_steps for anything left unset.
        cfg["agent_config"] = {"mode": "native", **agent}
    if args.limit is not None:
        cfg["limit"] = int(args.limit) if args.limit >= 1 else args.limit
    if args.api_timeout:
        cfg["timeout"] = args.api_timeout
    return cfg


def collect_reports(work_dir: str) -> List[Dict[str, Any]]:
    reports = []
    for path in sorted(glob.glob(os.path.join(work_dir, "reports", "**", "*.json"), recursive=True)):
        try:
            with open(path) as f:
                report = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        reports.append(
            {
                "path": os.path.relpath(path, work_dir),
                "dataset_name": report.get("dataset_name"),
                "model_name": report.get("model_name"),
                "score": report.get("score"),
                "metrics": report.get("metrics"),
            }
        )
    return reports


# Reasons SWE-bench's eval_instance gives up that are the model's doing, not the
# infrastructure's: an unresolved sample, not an evaluation error.
_MODEL_FAILURES = ("Patch Apply Failed", "Test timed out")


def evaluation_errors(work_dir: str) -> Dict[str, Dict[str, Any]]:
    """Per review file, the samples whose evaluation did not complete for reasons other
    than the model's patch: {"<model>/<dataset>_<subset>": {"count", "instances"}}."""
    errors: Dict[str, Dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(work_dir, "reviews", "*", "*.jsonl"))):
        instances = []
        with open(path) as f:
            for line in f:
                sample = json.loads(line)["sample_score"]
                result = sample["score"].get("metadata") or {}
                if result.get("completed", True):
                    continue
                error = str((result.get("report") or {}).get("error", ""))
                if any(marker in error for marker in _MODEL_FAILURES):
                    continue
                instances.append((sample.get("sample_metadata") or {}).get("instance_id") or sample.get("sample_id"))
        if instances:
            key = os.path.splitext(os.path.relpath(path, os.path.join(work_dir, "reviews")))[0]
            errors[key] = {"count": len(instances), "instances": instances}
    return errors


def swe_bench_pro_os_commit() -> Optional[str]:
    try:
        from evalscope.benchmarks.swe_bench_pro.utils import PINNED_COMMIT
    except Exception:
        return None
    return PINNED_COMMIT


def opensandbox_endpoint() -> str:
    from scripts.common import opensandbox_sandbox

    return opensandbox_sandbox.domain()


def delete_leftover_sandboxes() -> None:
    """Kill the sandboxes the run left (a sample whose cleanup failed, a stopped run)."""
    import asyncio

    from scripts.common import opensandbox_sandbox

    label = os.environ[opensandbox_sandbox.RUN_LABEL_ENV]
    try:
        n = asyncio.run(asyncio.wait_for(opensandbox_sandbox.delete_run_sandboxes(label), 120))
    except Exception as exc:  # noqa: BLE001 -- best effort: they time out on their own
        print(f"Could not delete leftover sandboxes (sandbox-run={label}): {exc}", file=sys.stderr)
        return
    if n:
        print(f"Deleted {n} leftover sandbox(es) of sandbox-run={label}", file=sys.stderr)


def exit_on_sigterm(signum: int, frame: Any) -> None:
    """SIGTERM (a cancelled or timed-out job, the sbatch's time limit) as a normal exit, so the
    finally around run_task still deletes the run's OpenSandbox sandboxes; Python's default
    would end the process on the spot and leave them until their timeout."""
    raise SystemExit(128 + signum)


def install_gold() -> None:
    """Make the SWE-bench agentic adapters predict each sample's reference patch, with no agent loop."""
    from evalscope.api.model import ModelOutput
    from evalscope.benchmarks.swe_bench.swe_bench_agentic_adapter import _SWEBenchAgenticAdapterBase
    from evalscope.benchmarks.swe_bench_pro.swe_bench_pro_agentic_adapter import SWEBenchProAgenticAdapter

    def _on_inference(self, model: Any, sample: Any) -> Any:
        return ModelOutput.from_content(model="gold", content=sample.metadata["patch"])

    for adapter in (_SWEBenchAgenticAdapterBase, SWEBenchProAgenticAdapter):
        adapter._on_inference = _on_inference


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    names = selected_benchmarks(args.benchmark)
    revisions = resolved_revisions(names, args.dataset_revision)
    os.makedirs(args.work_dir, exist_ok=True)

    from scripts.evalscope import swe_bench_scoring

    swe_bench_scoring.install(sandbox_config(args))
    if args.sandbox == "k8s":
        from scripts.common import k8s_sandbox

        if "memory_limit" not in json.loads(args.sandbox_config) and not k8s_sandbox.setting("MEMORY_REQUEST"):
            os.environ["SANDBOX_K8S_MEMORY_REQUEST"] = K8S_MEMORY_REQUEST
        k8s_sandbox.install()
    elif args.sandbox == "opensandbox":
        from scripts.common import opensandbox_sandbox

        # One label for the whole run, so its leftovers can be deleted at the end.
        os.environ.setdefault(opensandbox_sandbox.RUN_LABEL_ENV, f"evalscope-{uuid.uuid4().hex[:12]}")
        opensandbox_sandbox.install(agent_allowed_hosts(args))
        signal.signal(signal.SIGTERM, exit_on_sigterm)
    if args.gold:
        install_gold()

    import evalscope
    from evalscope.run import run_task

    cfg = build_task_config(args, names, revisions)
    printable = {k: v for k, v in cfg.items() if k != "api_key"}
    print("EvalScope task config:", json.dumps(printable, indent=2), flush=True)

    try:
        run_task(cfg)
    finally:
        if args.sandbox == "opensandbox":
            delete_leftover_sandboxes()

    summary = {
        "evalscope_version": evalscope.__version__,
        "benchmarks": {n: BENCHMARKS[n]["evalscope"] for n in names},
        "model": "gold (reference patches)" if args.gold else args.model,
        "api_base_url": args.api_base_url,
        "sandbox": args.sandbox,
        **({"opensandbox_endpoint": opensandbox_endpoint()} if args.sandbox == "opensandbox" else {}),
        # Only enforced on OpenSandbox; docker and k8s sandboxes have open internet.
        "agent_network": args.agent_network if args.sandbox == "opensandbox" else "public",
        "dataset_revisions": revisions,
        "limit": cfg.get("limit"),
        "reports": collect_reports(args.work_dir),
        "evaluation_errors": evaluation_errors(args.work_dir),
    }
    if "swe_bench_pro" in names:
        summary["swe_bench_pro_os_commit"] = swe_bench_pro_os_commit()
    with open(os.path.join(args.work_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if not summary["reports"]:
        print(f"No EvalScope reports found under {args.work_dir}/reports", file=sys.stderr)
        return 1
    if summary["evaluation_errors"]:
        total = sum(e["count"] for e in summary["evaluation_errors"].values())
        print(
            f"{total} sample(s) could not be evaluated (see evaluation_errors in {args.work_dir}/summary.json "
            "and their logs): the scores count them as unresolved.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
