#!/bin/bash
#
# run_tau2_bench.sh - Run tau2-bench (https://github.com/sierra-research/tau2-bench) straight from
# upstream: any ref of the repository, with the dependencies upstream's own uv.lock pins, and any
# domain that ref has (mock, airline, retail, telecom, telecom-workflow, banking_knowledge, intake
# at v2.0.0). Self-contained: no framework of this repo in between, and no lock of ours -- the ref
# is the version. The model under test is reached the way run_inspect_eval.sh reaches its model;
# the defaults are plain tau2-bench's (nothing of any leaderboard's protocol).
#
# The ref is checked out into a cache keyed by its commit, and `uv sync --frozen --extra knowledge`
# installs upstream's locked dependencies into the checkout's own .venv (Python 3.12). tau2 reads
# its data from that checkout and writes its results to <checkout>/data/simulations/<--save-to>.
# Needs git, uv and network access to the repository the first time a commit is used.
#
# Usage:
#   scripts/custom/run_tau2_bench.sh --domain <domain> --model <model> [--api-base-url <url>] [options]
#
# Required:
#   --domain <name>            A domain of the ref (see above; `tau2 --help` lists the CLI's).
#   --model <model>            Model under test. With --api-base-url, the served model name (wrapped
#                               through litellm's openai/ provider against that endpoint); without
#                               it, a model string litellm understands (with its own credentials).
#
# Options:
#   --api-base-url <url>       OpenAI-compatible endpoint serving --model (bare host or /v1 root).
#                               Its key: TARGET_API_KEY (default: CSCS_SERVING_API; both also from
#                               scripts/secrets.env, as in run_inspect_eval.sh).
#   --agent-llm-args <json>    Replace the agent's litellm kwargs (default: api_base/api_key built
#                               from --api-base-url/TARGET_API_KEY).
#   --user-llm <model>         The simulated user (and NL-assertion judge) (default: --model on the
#                               same endpoint, so no second credential). A bare name is reached by
#                               litellm with its own credentials (e.g. OPENAI_API_KEY).
#   --user-llm-args <json>     Extra litellm kwargs for the user, merged over its default key by key.
#   --retrieval-config <name>  banking_knowledge: the retrieval pipeline (tau2's default if unset).
#   --num-trials <n>           Trials per task (default: 1). The summary prints pass^k for k = 1..n.
#   --max-steps <n>            Max steps per trial (default: 100, tau2's).
#   --num-tasks <n>            Only the first n tasks (a smoke test).
#   --seed <n>                 Random seed (default: 300, tau2's).
#   --max-concurrency <n>      Parallel trials (default: 4).
#   --temperature V, --top-p V, --top-k V, --min-p V, --repetition-penalty V
#                              Sampling of the agent model, over its litellm args. tau2 has no default
#                              here: it sends no sampling setting unless you pass one (docs/sampling.md).
#                              temperature and top-p are litellm kwargs; the rest go in extra_body.
#   --sampling-profile <name>  A profile of the sampling config (e.g. thinking, greedy);
#   --sampling-config <file>   another file instead of configs/sampling_defaults.json.
#   --enable-thinking, --no-enable-thinking, --reasoning-effort <level>
#                              Thinking and reasoning effort of the agent model (scripts/common/thinking.py).
#                              The user simulator never gets them (--user-llm-args does).
#   --git-ref <ref>            tau2-bench tag, branch or full commit (default: v2.0.0).
#   --repo <url>               Repository the ref is looked up in (default: upstream).
#   --save-to <name>           tau2's run name, the results subdirectory (default: <domain>-<date and time>, a new
#                               run each time). Giving the name of an earlier run resumes it (--auto-resume):
#                               only with the same domain, model and options, else tau2 refuses.
#   --cache-dir <path>         Where checkouts live (default: ${EVAL_ENV_CACHE_ROOT:-$SCRATCH/eval-envs}/tau2-upstream,
#                               ~/.cache/evals-post-train/eval-envs/tau2-upstream without $SCRATCH).
#   -- <extra args>             Forwarded verbatim to `tau2 run` (e.g. a domain's own options).
#
# Results: <checkout>/data/simulations/<--save-to>/results.json (every simulation, reward and
# messages); this script prints avg_reward and pass^k computed from it.
#
# Example (retail smoke test against a CSCS-served model):
#   scripts/custom/run_tau2_bench.sh --domain retail --model CSCS-Inference/swiss-ai/Apertus-v1.5-8B \
#     --api-base-url https://api.swissai.svc.cscs.ch/v1 --num-tasks 2

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

die() { echo "$*" >&2; exit 1; }

usage() {
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}

DOMAIN=""
MODEL=""
API_BASE_URL=${API_BASE_URL:-""}
AGENT_LLM_ARGS_OVERRIDE=""
USER_LLM=""
USER_LLM_ARGS_OVERRIDE=""
RETRIEVAL_CONFIG=""
NUM_TRIALS=1
MAX_STEPS=100
NUM_TASKS=""
SEED=300
MAX_CONCURRENCY=4
TAU_GIT_REF="v2.0.0"
TAU_REPO="https://github.com/sierra-research/tau2-bench.git"
SAVE_TO=""
CACHE_DIR=""
SAMPLING_ARGS=(--framework tau2 --format litellm --warn-greedy)
THINKING_ARGS=(--format json)
EXTRA_ARGS=()

while (( $# > 0 )); do
    case "$1" in
        --domain) DOMAIN=$2; shift 2 ;;
        --model) MODEL=$2; shift 2 ;;
        --api-base-url) API_BASE_URL=$2; shift 2 ;;
        --agent-llm-args) AGENT_LLM_ARGS_OVERRIDE=$2; shift 2 ;;
        --user-llm) USER_LLM=$2; shift 2 ;;
        --user-llm-args) USER_LLM_ARGS_OVERRIDE=$2; shift 2 ;;
        --retrieval-config) RETRIEVAL_CONFIG=$2; shift 2 ;;
        --num-trials) NUM_TRIALS=$2; shift 2 ;;
        --max-steps) MAX_STEPS=$2; shift 2 ;;
        --num-tasks) NUM_TASKS=$2; shift 2 ;;
        --seed) SEED=$2; shift 2 ;;
        --max-concurrency) MAX_CONCURRENCY=$2; shift 2 ;;
        --temperature|--top-p|--top-k|--min-p|--repetition-penalty|--sampling-profile)
            SAMPLING_ARGS+=("$1" "$2"); shift 2 ;;
        --sampling-config) SAMPLING_ARGS+=(--config "$2"); shift 2 ;;
        --enable-thinking|--no-enable-thinking)
            SAMPLING_ARGS+=("$1"); THINKING_ARGS+=("$1"); shift ;;
        --reasoning-effort)
            SAMPLING_ARGS+=("$1" "$2"); THINKING_ARGS+=("$1" "$2"); shift 2 ;;
        --git-ref) TAU_GIT_REF=$2; shift 2 ;;
        --repo) TAU_REPO=$2; shift 2 ;;
        --save-to) SAVE_TO=$2; shift 2 ;;
        --cache-dir) CACHE_DIR=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *) die "Unknown argument: $1 (see --help)" ;;
    esac
done

[[ -n "$DOMAIN" ]] || die "Missing --domain. See --help."
[[ -n "$MODEL" ]] || die "Missing --model. See --help."
# The domain goes into a path and a command line: a name, nothing else.
[[ "$DOMAIN" =~ ^[A-Za-z0-9_-]+$ ]] || die "--domain must be a domain name, got: $DOMAIN"
SAVE_TO=${SAVE_TO:-"$DOMAIN-$(date +%Y%m%d-%H%M%S)"}
[[ "$SAVE_TO" =~ ^[A-Za-z0-9_.-]+$ ]] || die "--save-to must be a plain name, got: $SAVE_TO"
command -v git >/dev/null 2>&1 || die "git is required."
command -v uv >/dev/null 2>&1 || die "uv is required (https://docs.astral.sh/uv/): it installs upstream's uv.lock."

# The agent: through the endpoint like run_inspect_eval.sh does, or a bare litellm model string.
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
[[ -z "$AGENT_LLM_ARGS_OVERRIDE" ]] || AGENT_LLM_ARGS=$AGENT_LLM_ARGS_OVERRIDE

# Sampling of the agent (scripts/common/sampling.py), over its litellm args key by key (extra_body too).
SAMPLING_LITELLM=$(python3 -m scripts.common.sampling "${SAMPLING_ARGS[@]}") || exit 1
AGENT_LLM_ARGS=$(python3 -c '
import json, sys
args, sampling = json.loads(sys.argv[1]), json.loads(sys.argv[2])
body = {**args.get("extra_body", {}), **sampling.pop("extra_body", {})}
args.update(sampling)
if body:
    args["extra_body"] = body
print(json.dumps(args))' "$AGENT_LLM_ARGS" "$SAMPLING_LITELLM")

# The user simulator: by default the model under test on the same endpoint, with the agent's litellm
# args as they are now (sampling included, but not the thinking flags, added below). A bare name is
# litellm's own business (its credentials, no api_base).
if [[ -z "$USER_LLM" ]]; then
    [[ -n "$API_BASE_URL" ]] || die "Without --api-base-url the user simulator needs its own --user-llm."
    USER_LLM=$AGENT_LLM
fi
if [[ "$USER_LLM" == openai/* && -n "$API_BASE_URL" ]]; then
    USER_LLM_ARGS=$AGENT_LLM_ARGS
else
    USER_LLM_ARGS='{}'
fi
if [[ -n "$USER_LLM_ARGS_OVERRIDE" ]]; then
    USER_LLM_ARGS=$(python3 -c '
import json, sys
merged = json.loads(sys.argv[1])
merged.update(json.loads(sys.argv[2]))
print(json.dumps(merged))' "$USER_LLM_ARGS" "$USER_LLM_ARGS_OVERRIDE")
fi

# Thinking and reasoning effort of the agent (scripts/common/thinking.py), after the user copied its args.
THINKING_VALUES=$(python3 -m scripts.common.thinking "${THINKING_ARGS[@]}") || exit 1
AGENT_LLM_ARGS=$(python3 -c '
import json, sys
from scripts.common import thinking
args, values = json.loads(sys.argv[1]), json.loads(sys.argv[2])
args.update({k: v for k, v in thinking.litellm_kwargs(values).items() if k != "extra_body"})
body = thinking.merge_extra_body(args.get("extra_body"), values)
if body:
    args["extra_body"] = body
print(json.dumps(args))' "$AGENT_LLM_ARGS" "$THINKING_VALUES")

# The commit a ref names in $TAU_REPO (a full commit is taken as is).
resolve_ref() {
    local ref=$1 out sha
    if [[ "$ref" =~ ^[0-9a-f]{40}$ ]]; then echo "$ref"; return; fi
    out=$(git ls-remote "$TAU_REPO" "refs/tags/$ref^{}" "refs/tags/$ref" "refs/heads/$ref") || return 1
    sha=$(grep -m1 '\^{}$' <<<"$out" | cut -f1 || true)
    [[ -n "$sha" ]] || sha=$(head -1 <<<"$out" | cut -f1)
    [[ -n "$sha" ]] && echo "$sha"
}

COMMIT=$(resolve_ref "$TAU_GIT_REF") || die "--git-ref $TAU_GIT_REF: not found in $TAU_REPO"
[[ -n "$COMMIT" ]] || die "--git-ref $TAU_GIT_REF: not found in $TAU_REPO"
ROOT=${CACHE_DIR:-${EVAL_ENV_CACHE_ROOT:-${SCRATCH:-${XDG_CACHE_HOME:-$HOME/.cache}/evals-post-train}/eval-envs}/tau2-upstream}
# A venv only works with the machine and Python that built it, so the key has the architecture too.
CHECKOUT="$ROOT/$COMMIT-$(uname -m)"
mkdir -p "$ROOT"

# Simultaneous runs wait for the first builder (flock is missing on macOS, where runs rarely race).
exec 9>"$ROOT/$COMMIT.lock"
if command -v flock >/dev/null 2>&1; then flock 9; fi
if [[ ! -f "$CHECKOUT/.complete" ]]; then
    echo "Installing tau2-bench@$COMMIT ($TAU_GIT_REF) into $CHECKOUT"
    # Built in place (a venv embeds its path); an interrupted build is kept for diagnosis.
    if [[ -e "$CHECKOUT" ]]; then mv "$CHECKOUT" "$CHECKOUT.incomplete.$(date +%s).$$"; fi
    git init --quiet "$CHECKOUT"
    git -C "$CHECKOUT" fetch --quiet --depth 1 "$TAU_REPO" "$COMMIT"
    git -C "$CHECKOUT" checkout --quiet FETCH_HEAD
    # Upstream's own pins (uv.lock); the knowledge extra is banking_knowledge's retrieval.
    (cd "$CHECKOUT" && uv sync --frozen --quiet --extra knowledge --python 3.12) ||
        die "uv sync --frozen of tau2-bench@$COMMIT failed (does that ref have a uv.lock and a knowledge extra?)"
    [[ -x "$CHECKOUT/.venv/bin/tau2" ]] || die "tau2-bench@$COMMIT installed no tau2 command."
    touch "$CHECKOUT/.complete"
fi
exec 9>&-
touch "$CHECKOUT/.complete"  # also refreshes it for age-based scratch retention
echo "tau2-bench@$COMMIT ($TAU_GIT_REF) from $CHECKOUT"

ARGS=(run --domain "$DOMAIN"
      --agent-llm "$AGENT_LLM" --agent-llm-args "$AGENT_LLM_ARGS"
      --user-llm "$USER_LLM" --user-llm-args "$USER_LLM_ARGS"
      --num-trials "$NUM_TRIALS" --max-steps "$MAX_STEPS" --seed "$SEED"
      --max-concurrency "$MAX_CONCURRENCY"
      --save-to "$SAVE_TO" --auto-resume --log-level INFO)
[[ -z "$RETRIEVAL_CONFIG" ]] || ARGS+=(--retrieval-config "$RETRIEVAL_CONFIG")
[[ -z "$NUM_TASKS" ]] || ARGS+=(--num-tasks "$NUM_TASKS")
(( ${#EXTRA_ARGS[@]} == 0 )) || ARGS+=("${EXTRA_ARGS[@]}")

echo "+ tau2 run --domain $DOMAIN --agent-llm $AGENT_LLM --user-llm $USER_LLM" \
     "--num-trials $NUM_TRIALS --max-steps $MAX_STEPS --seed $SEED" \
     "--max-concurrency $MAX_CONCURRENCY (llm args carry the keys; not shown)"
# Data and results are the checkout's: tau2 resolves them relative to its own source tree.
(cd "$CHECKOUT" && .venv/bin/tau2 "${ARGS[@]}")

RESULTS_FILE="$CHECKOUT/data/simulations/$SAVE_TO/results.json"
[[ -f "$RESULTS_FILE" ]] || RESULTS_FILE=$(find "$CHECKOUT/data" -path "*/$SAVE_TO/results.json" 2>/dev/null | head -1)
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
