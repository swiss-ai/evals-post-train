#!/bin/bash
#
# run_terminal_bench.sh - Run Terminal-Bench 2.x on Harbor (https://github.com/laude-institute/
# terminal-bench) against a model, standalone -- the same real protocol evals-svc's own
# terminal_bench.py suite runs, extracted here so it's runnable directly from evals-post-train
# without going through evals-svc's API. There is no inspect_evals equivalent for this one
# (unlike tau-bench/gdpval, which have narrower parallel inspect_evals ports documented
# elsewhere in this README) -- Harbor is the only path.
#
# The default is Artificial Analysis's protocol: dataset terminal-bench/terminal-bench-2-1 (89
# tasks), the Terminus 2 agent, 3 trials per task, pass@1 (reported here as Harbor's own
# pass@k). Every default can be overridden; a run that deviates is just a different, non-AA-
# protocol run -- this script does not track/refuse that the way evals-svc's own aa_protocol
# flag does.
#
# A trial is a task container the agent works in: Harbor needs a container runtime wherever
# this runs. This script uses a Docker daemon when it finds one, else rootless podman through
# its docker shim (same fallback evals-svc's own runner uses, e.g. on Clariden where `docker`
# is podman's shim with no compose provider that satisfies Harbor's own compose usage). Task
# images are linux/amd64 -- this will not work on an aarch64-only node.
#
# The agent reaches the model under test through litellm's openai provider: --api-base-url
# wraps --model as an OpenAI-compatible endpoint (OPENAI_BASE_URL/OPENAI_API_KEY); without it,
# --model is passed straight through (needs its own OPENAI_API_KEY).
#
# Usage:
#   aaii/run_terminal_bench.sh --model <model> [--api-base-url <url>] [options]
#
# Required:
#   --model <model>            Model under test. With --api-base-url, the served model name;
#                               without it, an OpenAI model string litellm understands directly.
#
# Options:
#   --api-base-url <url>       OpenAI-compatible endpoint serving --model. Reads the key from
#                               TARGET_API_KEY (default: scripts/cscs_serving_api_key.txt, same
#                               fallback run_inspect_eval.sh uses).
#   --agent <name>              terminus-2 (default, AA's protocol), claude-code, codex, oracle.
#   --dataset <name>            Registry dataset (default: terminal-bench/terminal-bench-2-1,
#                               AA's protocol).
#   --num-trials <n>            Trials per task (default: 3, AA's protocol).
#   --num-tasks <n>              Restrict to the first n tasks (smoke-testing).
#   --task-names <n1,n2,...>    Exactly these registry tasks, instead of the dataset.
#   --timeout-multiplier <f>    Multiplies each task's own timeout (default: 1.0).
#   --max-concurrency <n>       Parallel trials (default: 4).
#   --harbor-version <v>        Harbor version to install if not already provisioned (default:
#                               the version this repo pins, $TB_HARBOR_VERSION_DEFAULT below).
#   --workdir <path>            Scratch dir for the Harbor job output (default: a temp dir under
#                               /tmp).
#   -- <extra args>              Forwarded verbatim to `harbor run`.
#
# Results: Harbor writes <workdir>/jobs/evalspt/result.json (job stats) and one result.json per
# trial under the same dir -- this script prints an accuracy/pass@k summary computed from them,
# same metrics evals-svc's own results-parsing computes.
#
# Example (smoke test against a CSCS-served model):
#   aaii/run_terminal_bench.sh --model CSCS-Inference/swiss-ai/Apertus-v1.5-8B \
#     --api-base-url https://api.swissai.svc.cscs.ch/v1 --num-tasks 1 --num-trials 1

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

die() { echo "$*" >&2; exit 1; }

usage() {
    sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}

TB_HARBOR_VERSION_DEFAULT="0.22.0"

MODEL=""
API_BASE_URL=${API_BASE_URL:-""}
AGENT="terminus-2"
DATASET="terminal-bench/terminal-bench-2-1"
NUM_TRIALS=3
NUM_TASKS=""
TASK_NAMES=""
TIMEOUT_MULTIPLIER="1.0"
MAX_CONCURRENCY=4
HARBOR_VERSION="$TB_HARBOR_VERSION_DEFAULT"
WORKDIR=""
EXTRA_ARGS=()

while (( $# > 0 )); do
    case "$1" in
        --model) MODEL=$2; shift 2 ;;
        --api-base-url) API_BASE_URL=$2; shift 2 ;;
        --agent) AGENT=$2; shift 2 ;;
        --dataset) DATASET=$2; shift 2 ;;
        --num-trials) NUM_TRIALS=$2; shift 2 ;;
        --num-tasks) NUM_TASKS=$2; shift 2 ;;
        --task-names) TASK_NAMES=$2; shift 2 ;;
        --timeout-multiplier) TIMEOUT_MULTIPLIER=$2; shift 2 ;;
        --max-concurrency) MAX_CONCURRENCY=$2; shift 2 ;;
        --harbor-version) HARBOR_VERSION=$2; shift 2 ;;
        --workdir) WORKDIR=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *) die "Unknown argument: $1 (see --help)" ;;
    esac
done

[[ -n "$MODEL" ]] || die "Missing --model. See --help."
WORKDIR=${WORKDIR:-"$(mktemp -d /tmp/terminal-bench.XXXXXX)"}
mkdir -p "$WORKDIR"

if [[ -n "$API_BASE_URL" ]]; then
    [[ "$API_BASE_URL" == */v1 || "$API_BASE_URL" == */v1/ ]] || API_BASE_URL="${API_BASE_URL%/}/v1"
    TARGET_API_KEY="${TARGET_API_KEY:-}"
    if [[ -z "$TARGET_API_KEY" && -f ./scripts/cscs_serving_api_key.txt ]]; then
        TARGET_API_KEY="$(tr -d '\r\n' < ./scripts/cscs_serving_api_key.txt)"
    fi
    export OPENAI_API_KEY="$TARGET_API_KEY"
    export OPENAI_BASE_URL="$API_BASE_URL"
    export OPENAI_API_BASE="$API_BASE_URL"
    AGENT_MODEL="openai/$MODEL"
else
    AGENT_MODEL="$MODEL"
fi

if python3 -c "import harbor" >/dev/null 2>&1; then
    echo "provisioned harbor $(python3 -c 'import importlib.metadata as m; print(m.version("harbor"))')"
else
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh 1>&2
    uv venv --python 3.12 -q "$WORKDIR/venv" 1>&2
    uv pip install -q --python "$WORKDIR/venv/bin/python" "harbor==$HARBOR_VERSION" 1>&2
    export PATH="$WORKDIR/venv/bin:$PATH"
fi

# Container runtime for the task environments: a real Docker daemon if there is one, else
# rootless podman through a docker-compatible shim (same fallback evals-svc's own runner uses;
# see the module docstring for why this is needed on Clariden).
STATUS_OK=1
DOCKER_IS_PODMAN=0
if command -v podman >/dev/null 2>&1; then
    if ! command -v docker >/dev/null 2>&1; then
        DOCKER_IS_PODMAN=1
    elif [[ "$(docker --version 2>&1 || true)" == *odman* ]]; then
        DOCKER_IS_PODMAN=1
    fi
fi
if [[ "$DOCKER_IS_PODMAN" = 1 ]]; then
    STORAGE_ROOT="${TMPDIR:-/tmp}/terminal-bench-$(id -u)/podman"
    mkdir -p "$STORAGE_ROOT/run" "$WORKDIR/bin"
    cat > "$WORKDIR/storage.conf" <<CONF
[storage]
driver = "overlay"
graphroot = "$STORAGE_ROOT/storage"
runroot = "$STORAGE_ROOT/run"
CONF
    export CONTAINERS_STORAGE_CONF="$WORKDIR/storage.conf"
    podman system service --time=0 "unix://$STORAGE_ROOT/run/podman.sock" \
        > "$WORKDIR/podman-service.log" 2>&1 &
    PODMAN_SERVICE_PID=$!
    export DOCKER_HOST="unix://$STORAGE_ROOT/run/podman.sock"
    COMPOSE_ARCH=$(uname -m | sed 's/arm64/aarch64/')
    curl -fsSL -o "$WORKDIR/bin/docker-compose" \
        "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$COMPOSE_ARCH"
    chmod +x "$WORKDIR/bin/docker-compose"
    cat > "$WORKDIR/bin/docker" <<'WRAP'
#!/bin/sh
if [ "$1" = "compose" ]; then
    shift
    exec "$(dirname "$0")/docker-compose" "$@"
fi
exec podman "$@"
WRAP
    chmod +x "$WORKDIR/bin/docker"
    export PATH="$WORKDIR/bin:$PATH"
    sleep 2
    echo "container runtime: podman $(podman --version 2>/dev/null | awk '{print $NF}')" \
        "+ compose $("$WORKDIR/bin/docker-compose" version --short 2>/dev/null)"
elif docker info >/dev/null 2>&1; then
    echo "container runtime: docker ($(docker version --format '{{.Server.Version}}' 2>/dev/null))"
else
    echo "no container runtime: neither a Docker daemon nor podman is available here" 1>&2
    STATUS_OK=0
fi

ARGS=(run -a "$AGENT" -m "$AGENT_MODEL"
      -k "$NUM_TRIALS" -n "$MAX_CONCURRENCY" --timeout-multiplier "$TIMEOUT_MULTIPLIER"
      -o "$WORKDIR/jobs" --job-name evalspt -y -q)
if [[ -n "$TASK_NAMES" ]]; then
    IFS=',' read -ra NAMES <<< "$TASK_NAMES"
    for t in "${NAMES[@]}"; do ARGS+=(-t "$t"); done
else
    ARGS+=(-d "$DATASET")
    [[ -n "$NUM_TASKS" ]] && ARGS+=(-l "$NUM_TASKS")
fi
(( ${#EXTRA_ARGS[@]} > 0 )) && ARGS+=("${EXTRA_ARGS[@]}")

echo "+ harbor run -a $AGENT -m $AGENT_MODEL -k $NUM_TRIALS -n $MAX_CONCURRENCY" \
     "--timeout-multiplier $TIMEOUT_MULTIPLIER ${DATASET:+-d $DATASET} ${NUM_TASKS:+-l $NUM_TASKS}"
if [[ "$STATUS_OK" = 1 ]]; then
    harbor "${ARGS[@]}"
fi
[[ -n "${PODMAN_SERVICE_PID:-}" ]] && kill "$PODMAN_SERVICE_PID" 2>/dev/null

RESULTS_FILE="$WORKDIR/jobs/evalspt/result.json"
TRIALS_DIR="$WORKDIR/jobs/evalspt"
if [[ ! -f "$RESULTS_FILE" ]]; then
    RESULTS_FILE=$(find "$WORKDIR/jobs" -maxdepth 2 -name result.json 2>/dev/null | head -1)
    [[ -n "$RESULTS_FILE" ]] && TRIALS_DIR=$(dirname "$RESULTS_FILE")
fi
[[ -n "$RESULTS_FILE" && -f "$RESULTS_FILE" ]] || die "harbor produced no result.json"
echo "results: $RESULTS_FILE"

python3 - "$RESULTS_FILE" "$TRIALS_DIR" "$DATASET" "$NUM_TRIALS" <<'PY'
import glob
import json
import os
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
trials_dir, dataset, num_trials = sys.argv[2], sys.argv[3], int(sys.argv[4])

trials = []
for tp in sorted(glob.glob(os.path.join(trials_dir, "*", "result.json"))):
    try:
        with open(tp) as f:
            trials.append(json.load(f))
    except (OSError, ValueError):
        pass
if not trials:
    trials = data.get("trial_results") or []

stats = data.get("stats") or {}
scored = [t for t in trials if (t.get("verifier_result") or {}).get("rewards")]
if not scored:
    print("no scored trials")
    raise SystemExit(1)

rewards = [float((t["verifier_result"]["rewards"]).get("reward", 0.0)) for t in scored]
metrics = {"accuracy": sum(rewards) / len(rewards)}
for ev in (stats.get("evals") or {}).values():
    for k, v in (ev.get("pass_at_k") or {}).items():
        metrics[f"pass@{k}"] = v
metrics["n_tasks"] = len({t.get("task_name") for t in scored})
metrics["n_trials"] = num_trials
metrics["n_scored"] = len(scored)
metrics["n_errors"] = int(stats.get("n_errored_trials") or 0)
print(json.dumps(metrics, indent=2))
PY
