#!/bin/bash
#
# run_tau2_bench.sh - Run tau2-bench (https://github.com/sierra-research/tau2-bench) against a
# model with Artificial Analysis's protocol (AA_PROTOCOL below). NOT the same benchmark as this
# repo's inspect_evals/tau2 (docs/inspect-ai.md): that's a narrower, independent inspect_evals
# port; this script drives the real tau2-bench package.
#
# The default is Artificial Analysis's tau^3-Banking protocol, so scores line up with the AA
# Intelligence Index: tau2-bench v1.0.1, domain banking_knowledge, retrieval bm25_grep, 5 trials
# per task, 200 max steps, user simulator + NL-assertion judge GPT-5.4 mini (medium reasoning),
# pass@1 averaged over trials (reported here as pass^k for k=1..num_trials, tau2-bench's own
# metric). Every default can be overridden; a run that deviates is just a different, non-AA-
# protocol run.
#
# The agent (the model under test) is reached the same way run_inspect_eval.sh reaches its
# model: --api-base-url wraps --model as an OpenAI-compatible endpoint via litellm; without
# --api-base-url, --model is passed to tau2 as-is (e.g. an OpenAI model string). The user
# simulator is a SEPARATE model (--user-llm), and unlike the agent, tau2/litellm's own model
# naming decides how it's reached, not --api-base-url:
#   - A BARE name (default: gpt-5.4-mini, AA's protocol judge, reasoning_effort=medium) is a
#     real OpenAI model -- litellm resolves it via the standard OPENAI_API_KEY env var, which
#     this script does NOT set for you; export it yourself before running (or pass --user-llm
#     with a gateway model instead, see below).
#   - A GATEWAY name ("openai/<id>", any id --api-base-url actually serves) is routed through
#     the same endpoint/key as the agent, same convention as run_inspect_eval.sh's tau2/
#     AA-Omniscience "user"/"grader" roles -- no separate credential needed. Use this for a
#     smoke test with no OpenAI key at hand, e.g. --user-llm openai/<the same --model value>.
#
# Usage:
#   scripts/aaii/run_tau2_bench.sh --model <model> [--api-base-url <url>] [options]
#
# Required:
#   --model <model>            Model under test. With --api-base-url, this is the served model
#                               name; without it, an OpenAI model string tau2 understands
#                               directly (needs OPENAI_API_KEY of its own).
#
# Options:
#   --api-base-url <url>       OpenAI-compatible endpoint serving --model (bare host or /v1
#                               root). Wraps --model through litellm's openai/ provider against
#                               this endpoint, reading the key from TARGET_API_KEY (default:
#                               CSCS_SERVING_API; both also from scripts/secrets.env,
#                               as in run_inspect_eval.sh).
#   --agent-llm-args <json>    Override the agent's own litellm kwargs entirely (default: the
#                               api_base/api_key pair built from --api-base-url/TARGET_API_KEY).
#   --domain <name>            tau2-bench domain: airline, banking_knowledge (default, AA's
#                               protocol), retail, telecom.
#   --retrieval-config <name>  Only meaningful for banking_knowledge. Default: bm25_grep (AA's
#                               protocol). Other values: qwen_embeddings_reranker,
#                               qwen_embeddings_reranker_grep, full_kb, no_knowledge,
#                               golden_retrieval.
#   --user-llm <model>         User-simulator/judge model (default: gpt-5.4-mini, a real OpenAI
#                               model -- needs OPENAI_API_KEY of your own). A gateway model
#                               ("openai/<id>", same --api-base-url as --model) needs no
#                               separate credential -- see above.
#   --user-llm-args <json>     Extra litellm kwargs for the user-llm, merged over the default
#                               key by key ({"reasoning_effort": "medium"} for a bare model, or
#                               the gateway api_base/api_key for a gateway one).
#   --num-trials <n>           Trials per task (default: 5, AA's protocol).
#   --max-steps <n>            Max steps per trial (default: 200, AA's protocol).
#   --num-tasks <n>            Restrict to the first n tasks (useful for smoke-testing).
#   --seed <n>                 Random seed (default: 300).
#   --max-concurrency <n>      Parallel trials (default: 4).
#   --git-ref <ref>            tau2-bench ref (default: v1.0.1, AA's protocol). It must name the
#                               commit requirements/tau2-runtime.lock pins, which is what runs
#                               (the prebuilt image's install, or the lock's cached venv
#                               elsewhere); anything else is refused. Another version: bump
#                               requirements/tau2-runtime.txt and relock.
#   --repo <url>                tau2-bench repo --git-ref is looked up in (default: upstream).
#   --save-to <name>           tau2's own run-name, used as the results subdirectory (default:
#                               evalspt).
#   --workdir <path>           Scratch dir for tau2-bench's data dir (checked out at the pinned commit)
#                               (default: a temp dir under /tmp).
#   -- <extra args>             Forwarded verbatim to `tau2 run`.
#
# Results: tau2 writes <data dir>/simulations/<--save-to>/results.json (every simulation, its
# reward and messages) -- this script also prints a pass^k summary computed from it.
#
# Example (banking_knowledge smoke test against a CSCS-served model):
#   scripts/aaii/run_tau2_bench.sh --model CSCS-Inference/swiss-ai/Apertus-v1.5-8B \
#     --api-base-url https://api.swissai.svc.cscs.ch/v1 --num-tasks 2 --num-trials 1

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

die() { echo "$*" >&2; exit 1; }

usage() {
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}

MODEL=""
API_BASE_URL=${API_BASE_URL:-""}
DOMAIN="banking_knowledge"
RETRIEVAL_CONFIG="bm25_grep"
USER_LLM="gpt-5.4-mini"
USER_LLM_ARGS_OVERRIDE=""
AGENT_LLM_ARGS_OVERRIDE=""
NUM_TRIALS=5
MAX_STEPS=200
NUM_TASKS=""
SEED=300
MAX_CONCURRENCY=4
TAU_GIT_REF="v1.0.1"
TAU_REPO="https://github.com/sierra-research/tau2-bench.git"
SAVE_TO="evalspt"
WORKDIR=""
EXTRA_ARGS=()

while (( $# > 0 )); do
    case "$1" in
        --model) MODEL=$2; shift 2 ;;
        --api-base-url) API_BASE_URL=$2; shift 2 ;;
        --domain) DOMAIN=$2; shift 2 ;;
        --retrieval-config) RETRIEVAL_CONFIG=$2; shift 2 ;;
        --user-llm) USER_LLM=$2; shift 2 ;;
        --user-llm-args) USER_LLM_ARGS_OVERRIDE=$2; shift 2 ;;
        --agent-llm-args) AGENT_LLM_ARGS_OVERRIDE=$2; shift 2 ;;
        --num-trials) NUM_TRIALS=$2; shift 2 ;;
        --max-steps) MAX_STEPS=$2; shift 2 ;;
        --num-tasks) NUM_TASKS=$2; shift 2 ;;
        --seed) SEED=$2; shift 2 ;;
        --max-concurrency) MAX_CONCURRENCY=$2; shift 2 ;;
        --git-ref) TAU_GIT_REF=$2; shift 2 ;;
        --repo) TAU_REPO=$2; shift 2 ;;
        --save-to) SAVE_TO=$2; shift 2 ;;
        --workdir) WORKDIR=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *) die "Unknown argument: $1 (see --help)" ;;
    esac
done

[[ -n "$MODEL" ]] || die "Missing --model. See --help."
WORKDIR=${WORKDIR:-"$(mktemp -d /tmp/tau2-bench.XXXXXX)"}
mkdir -p "$WORKDIR"

AGENT_LLM_ARGS='{}'
if [[ -n "$API_BASE_URL" ]]; then
    [[ "$API_BASE_URL" == */v1 || "$API_BASE_URL" == */v1/ ]] || API_BASE_URL="${API_BASE_URL%/}/v1"
    # shellcheck source=scripts/common/secrets.sh
    source scripts/common/secrets.sh
    TARGET_API_KEY="${TARGET_API_KEY:-${CSCS_SERVING_API:-}}"
    AGENT_LLM_ARGS=$(python3 -c "import json,sys; print(json.dumps({'api_base': sys.argv[1], 'api_key': sys.argv[2]}))" \
        "$API_BASE_URL" "$TARGET_API_KEY")
    AGENT_LLM="openai/$MODEL"
else
    AGENT_LLM="$MODEL"
fi
if [[ -n "$AGENT_LLM_ARGS_OVERRIDE" ]]; then
    AGENT_LLM_ARGS="$AGENT_LLM_ARGS_OVERRIDE"
fi

if [[ "$USER_LLM" == openai/* && -n "$API_BASE_URL" ]]; then
    # A gateway user-llm (same convention as run_inspect_eval.sh's tau2 "user" role): route it
    # through the same endpoint/key as the agent, not a real OpenAI credential.
    USER_LLM_ARGS="$AGENT_LLM_ARGS"
else
    # A bare model name (default: gpt-5.4-mini) is a real OpenAI model -- litellm resolves it
    # via OPENAI_API_KEY directly, no api_base/api_key override here. AA's own protocol default
    # is reasoning_effort=medium (gpt-5.4-mini's API default is none).
    USER_LLM_ARGS='{"reasoning_effort": "medium"}'
fi
if [[ -n "$USER_LLM_ARGS_OVERRIDE" ]]; then
    # Merged over the default, key by key: a caller that only passes a credential
    # ({"api_key": ...}) keeps reasoning_effort=medium; an explicit key still wins.
    USER_LLM_ARGS=$(python3 -c '
import json, sys
merged = json.loads(sys.argv[1])
merged.update(json.loads(sys.argv[2]))
print(json.dumps(merged))' "$USER_LLM_ARGS" "$USER_LLM_ARGS_OVERRIDE")
fi

# The commit a tau2-bench ref names in $TAU_REPO (a full commit is taken as is).
resolve_tau_ref() {
    local ref=$1 out sha
    if [[ "$ref" =~ ^[0-9a-f]{40}$ ]]; then echo "$ref"; return; fi
    out=$(git ls-remote "$TAU_REPO" "refs/tags/$ref^{}" "refs/tags/$ref" "refs/heads/$ref") || return 1
    sha=$(grep -m1 '\^{}$' <<<"$out" | cut -f1 || true)
    [[ -n "$sha" ]] || sha=$(head -1 <<<"$out" | cut -f1)
    [[ -n "$sha" ]] && echo "$sha"
}

# --git-ref must name the commit that runs: the lock's pin (requirements/tau2-runtime.txt).
check_tau_ref() {
    local pinned=$1 where=$2 wanted
    wanted=$(resolve_tau_ref "$TAU_GIT_REF") || die "--git-ref $TAU_GIT_REF: not found in $TAU_REPO"
    [[ "$wanted" == "$pinned" ]] ||
        die "--git-ref $TAU_GIT_REF is tau2-bench@$wanted, but $where has tau2-bench@$pinned." \
            "Bump requirements/tau2-runtime.txt and relock (scripts/common/lock_requirements.sh)."
}

if [[ "${EVAL_PREBUILT_ENV:-0}" == "1" ]]; then
    # The prebuilt image (containers/Dockerfile.eval): tau2-bench in its own venv, its data at
    # the commit requirements/tau2-runtime.txt pins. Nothing is installed.
    [[ -x "${TAU2_VENV:-}/bin/tau2" && -d "${TAU2_DATA_DIR:-}" && -f "${TAU2_COMMIT_FILE:-}" ]] ||
        die "EVAL_PREBUILT_ENV=1 but this image has no tau2-bench (TAU2_VENV=${TAU2_VENV:-unset}," \
            "TAU2_DATA_DIR=${TAU2_DATA_DIR:-unset}, TAU2_COMMIT_FILE=${TAU2_COMMIT_FILE:-unset})." \
            "Use the image built from this commit (containers/Dockerfile.eval)."
    INSTALLED_TAU=$(tr -d '[:space:]' < "$TAU2_COMMIT_FILE")
    check_tau_ref "$INSTALLED_TAU" "this image"
    export PATH="$TAU2_VENV/bin:$PATH"
    echo "EVAL_PREBUILT_ENV=1: tau2-bench@$INSTALLED_TAU ($TAU_GIT_REF) from $TAU2_VENV, no install" \
         "(data: $TAU2_DATA_DIR)"
    if [[ ! -w "$TAU2_DATA_DIR" ]]; then
        # A read-only image (Clariden's Container Engine mounts it as squashfs), but tau2 writes
        # its results to <data dir>/simulations: a writable view of the data in the workdir.
        VIEW="$WORKDIR/tau2-data"
        mkdir -p "$VIEW/simulations"
        for entry in "$TAU2_DATA_DIR"/* "$TAU2_DATA_DIR"/.[!.]*; do
            [[ -e "$entry" && "$(basename "$entry")" != simulations ]] && ln -sfn "$entry" "$VIEW/"
        done
        export TAU2_DATA_DIR="$VIEW"
        echo "read-only data dir: running from a writable view, $VIEW"
    fi
elif python3 -c "import tau2" >/dev/null 2>&1 && [[ -n "${TAU2_DATA_DIR:-}" && -d "$TAU2_DATA_DIR" ]]; then
    # An environment someone provisioned by hand: its version is not checked against --git-ref.
    echo "provisioned tau2-bench $(python3 -c 'import importlib.metadata as m; print(m.version("tau2"))')" \
         "(data: $TAU2_DATA_DIR; --git-ref $TAU_GIT_REF is not checked against it)"
else
    # As the image does it: the lock's cached venv (scripts/common/locked_venv.sh) and the data
    # directory checked out at the commit the lock pins.
    PINNED_TAU=$(grep -oE 'tau2-bench@[0-9a-f]{40}' requirements/tau2-runtime.lock | head -1 | cut -d@ -f2)
    [[ -n "$PINNED_TAU" ]] || die "requirements/tau2-runtime.lock pins no tau2-bench commit"
    check_tau_ref "$PINNED_TAU" "requirements/tau2-runtime.lock"
    # shellcheck source=scripts/common/locked_venv.sh
    source scripts/common/locked_venv.sh
    locked_venv requirements/tau2-runtime.lock 3.12 || die "Installing requirements/tau2-runtime.lock failed."
    export PATH="$LOCKED_VENV/bin:$PATH"
    git init --quiet "$WORKDIR/tau2-bench"
    git -C "$WORKDIR/tau2-bench" fetch --quiet --depth 1 "$TAU_REPO" "$PINNED_TAU"
    git -C "$WORKDIR/tau2-bench" checkout --quiet FETCH_HEAD
    export TAU2_DATA_DIR="$WORKDIR/tau2-bench/data"
    echo "tau2-bench@$PINNED_TAU ($TAU_GIT_REF) from $LOCKED_VENV (data: $TAU2_DATA_DIR)"
fi

ARGS=(run --domain "$DOMAIN"
      --agent-llm "$AGENT_LLM" --agent-llm-args "$AGENT_LLM_ARGS"
      --user-llm "$USER_LLM" --user-llm-args "$USER_LLM_ARGS"
      --num-trials "$NUM_TRIALS" --max-steps "$MAX_STEPS" --seed "$SEED"
      --max-concurrency "$MAX_CONCURRENCY"
      --save-to "$SAVE_TO" --auto-resume --log-level INFO)
[[ "$DOMAIN" == "banking_knowledge" ]] && ARGS+=(--retrieval-config "$RETRIEVAL_CONFIG")
[[ -n "$NUM_TASKS" ]] && ARGS+=(--num-tasks "$NUM_TASKS")
(( ${#EXTRA_ARGS[@]} > 0 )) && ARGS+=("${EXTRA_ARGS[@]}")

echo "+ tau2 run --domain $DOMAIN --agent-llm $AGENT_LLM --user-llm $USER_LLM" \
     "--num-trials $NUM_TRIALS --max-steps $MAX_STEPS --seed $SEED" \
     "--max-concurrency $MAX_CONCURRENCY (llm args carry the keys; not shown)"
tau2 "${ARGS[@]}"

RESULTS_FILE="$TAU2_DATA_DIR/simulations/$SAVE_TO/results.json"
[[ -f "$RESULTS_FILE" ]] || RESULTS_FILE=$(find "$TAU2_DATA_DIR" "$WORKDIR" -path "*/$SAVE_TO/results.json" 2>/dev/null | head -1)
[[ -n "$RESULTS_FILE" && -f "$RESULTS_FILE" ]] || die "tau2 produced no results.json"
echo "results: $RESULTS_FILE"

python3 - "$RESULTS_FILE" "$NUM_TRIALS" <<'PY'
import json
import math
import sys
from collections import defaultdict

with open(sys.argv[1]) as f:
    data = json.load(f)
num_trials = int(sys.argv[2])

sims = [s for s in data.get("simulations") or [] if (s.get("reward_info") or {}).get("reward") is not None]
per_task = defaultdict(list)
for s in sims:
    per_task[s["task_id"]].append(s)

if not per_task:
    print("no scored simulations")
    raise SystemExit(1)

rewards = [s["reward_info"]["reward"] for s in sims]
metrics = {"avg_reward": sum(rewards) / len(rewards)}


def is_success(reward):
    return (1 - 1e-6) <= reward <= (1 + 1e-6)


for k in range(1, num_trials + 1):
    vals = []
    for runs in per_task.values():
        n = len(runs)
        c = sum(is_success(s["reward_info"]["reward"]) for s in runs)
        if n >= k:
            vals.append(math.comb(c, k) / math.comb(n, k))
    if vals:
        metrics[f"pass^{k}"] = sum(vals) / len(vals)
metrics["n_tasks"] = len(per_task)
metrics["n_simulations"] = len(sims)
print(json.dumps(metrics, indent=2))
PY
