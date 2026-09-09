#!/bin/bash
#
# run_tau_bench.sh - Run tau2-bench (https://github.com/sierra-research/tau2-bench) against a
# model, standalone -- the same real protocol evals-svc's own tau_bench.py suite runs, extracted
# here so it's runnable directly from evals-post-train without going through evals-svc's API.
# NOT the same benchmark as this repo's inspect_evals/tau2 (documented separately above): that's
# a narrower, independent inspect_evals port; this script drives the real tau2-bench package,
# matching evals-svc's own protocol exactly (AA_PROTOCOL below).
#
# The default is Artificial Analysis's tau^3-Banking protocol, so scores line up with the AA
# Intelligence Index: tau2-bench v1.0.1, domain banking_knowledge, retrieval bm25_grep, 5 trials
# per task, 200 max steps, user simulator + NL-assertion judge GPT-5.4 mini (medium reasoning),
# pass@1 averaged over trials (reported here as pass^k for k=1..num_trials, tau2-bench's own
# metric). Every default can be overridden; a run that deviates is just a different, non-AA-
# protocol run -- this script does not track/refuse that the way evals-svc's own aa_protocol
# flag does.
#
# The agent (the model under test) is reached the same way run_inspect_eval.sh reaches its
# model: --api-base-url wraps --model as an OpenAI-compatible endpoint via litellm; without
# --api-base-url, --model is passed to tau2 as-is (e.g. an OpenAI model string). The user
# simulator is a SEPARATE model (--user-llm, default openai/gpt-5.4-mini, needing
# OPENAI_API_KEY) -- it does not need to be a real OpenAI model; any served model works too
# (--user-llm openai/<id> against the same --api-base-url, same convention as
# run_inspect_eval.sh's tau2/AA-Omniscience "user"/"grader" roles).
#
# Usage:
#   scripts/run_tau_bench.sh --model <model> [--api-base-url <url>] [options]
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
#                               scripts/cscs_serving_api_key.txt, same fallback
#                               run_inspect_eval.sh uses).
#   --domain <name>            tau2-bench domain: airline, banking_knowledge (default, AA's
#                               protocol), retail, telecom.
#   --retrieval-config <name>  Only meaningful for banking_knowledge. Default: bm25_grep (AA's
#                               protocol). Other values: qwen_embeddings_reranker,
#                               qwen_embeddings_reranker_grep, full_kb, no_knowledge,
#                               golden_retrieval.
#   --user-llm <model>         User-simulator/judge model (default: openai/gpt-5.4-mini). A
#                               gateway model (same --api-base-url as --model) works too.
#   --num-trials <n>           Trials per task (default: 5, AA's protocol).
#   --max-steps <n>            Max steps per trial (default: 200, AA's protocol).
#   --num-tasks <n>            Restrict to the first n tasks (useful for smoke-testing).
#   --seed <n>                 Random seed (default: 300).
#   --max-concurrency <n>      Parallel trials (default: 4).
#   --git-ref <ref>            tau2-bench ref to clone (default: v1.0.1, AA's protocol).
#   --repo <url>                tau2-bench repo to clone (default: the upstream GitHub repo).
#   --workdir <path>           Scratch dir for the tau2-bench checkout + its own data dir
#                               (default: a temp dir under /tmp).
#   -- <extra args>             Forwarded verbatim to `tau2 run`.
#
# Results: tau2 writes <data dir>/simulations/evalspt/results.json (every simulation, its
# reward and messages) -- this script also prints a pass^k summary computed from it, same
# metric evals-svc's own results-parsing computes.
#
# Example (banking_knowledge smoke test against a CSCS-served model):
#   scripts/run_tau_bench.sh --model CSCS-Inference/swiss-ai/Apertus-v1.5-8B \
#     --api-base-url https://api.swissai.svc.cscs.ch/v1 --num-tasks 2 --num-trials 1

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

die() { echo "$*" >&2; exit 1; }

usage() {
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}

MODEL=""
API_BASE_URL=${API_BASE_URL:-""}
DOMAIN="banking_knowledge"
RETRIEVAL_CONFIG="bm25_grep"
USER_LLM="openai/gpt-5.4-mini"
NUM_TRIALS=5
MAX_STEPS=200
NUM_TASKS=""
SEED=300
MAX_CONCURRENCY=4
TAU_GIT_REF="v1.0.1"
TAU_REPO="https://github.com/sierra-research/tau2-bench.git"
WORKDIR=""
EXTRA_ARGS=()

while (( $# > 0 )); do
    case "$1" in
        --model) MODEL=$2; shift 2 ;;
        --api-base-url) API_BASE_URL=$2; shift 2 ;;
        --domain) DOMAIN=$2; shift 2 ;;
        --retrieval-config) RETRIEVAL_CONFIG=$2; shift 2 ;;
        --user-llm) USER_LLM=$2; shift 2 ;;
        --num-trials) NUM_TRIALS=$2; shift 2 ;;
        --max-steps) MAX_STEPS=$2; shift 2 ;;
        --num-tasks) NUM_TASKS=$2; shift 2 ;;
        --seed) SEED=$2; shift 2 ;;
        --max-concurrency) MAX_CONCURRENCY=$2; shift 2 ;;
        --git-ref) TAU_GIT_REF=$2; shift 2 ;;
        --repo) TAU_REPO=$2; shift 2 ;;
        --workdir) WORKDIR=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *) die "Unknown argument: $1 (see --help)" ;;
    esac
done

[[ -n "$MODEL" ]] || die "Missing --model. See --help."
WORKDIR=${WORKDIR:-"$(mktemp -d /tmp/tau-bench.XXXXXX)"}
mkdir -p "$WORKDIR"

AGENT_LLM_ARGS='{}'
if [[ -n "$API_BASE_URL" ]]; then
    [[ "$API_BASE_URL" == */v1 || "$API_BASE_URL" == */v1/ ]] || API_BASE_URL="${API_BASE_URL%/}/v1"
    TARGET_API_KEY="${TARGET_API_KEY:-}"
    if [[ -z "$TARGET_API_KEY" && -f ./scripts/cscs_serving_api_key.txt ]]; then
        TARGET_API_KEY="$(tr -d '\r\n' < ./scripts/cscs_serving_api_key.txt)"
    fi
    AGENT_LLM_ARGS=$(python3 -c "import json,sys; print(json.dumps({'api_base': sys.argv[1], 'api_key': sys.argv[2]}))" \
        "$API_BASE_URL" "$TARGET_API_KEY")
    AGENT_LLM="openai/$MODEL"
else
    AGENT_LLM="$MODEL"
fi

USER_LLM_ARGS='{}'
if [[ "$USER_LLM" == openai/* && -n "$API_BASE_URL" ]]; then
    # A gateway user-llm (same convention as run_inspect_eval.sh's tau2 "user" role): route it
    # through the same endpoint/key as the agent, not a real OpenAI credential.
    USER_LLM_ARGS="$AGENT_LLM_ARGS"
fi

if python3 -c "import tau2" >/dev/null 2>&1 && [[ -n "${TAU2_DATA_DIR:-}" && -d "$TAU2_DATA_DIR" ]]; then
    echo "provisioned tau2-bench $(python3 -c 'import importlib.metadata as m; print(m.version("tau2"))') (data: $TAU2_DATA_DIR)"
else
    git clone --depth 1 --branch "$TAU_GIT_REF" "$TAU_REPO" "$WORKDIR/tau2-bench"
    pip install --no-cache-dir -q "$WORKDIR/tau2-bench[knowledge]" 1>&2
    export TAU2_DATA_DIR="$WORKDIR/tau2-bench/data"
fi

SAVE_TO="evalspt"
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
