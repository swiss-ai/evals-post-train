#!/bin/bash
#
# run_bfcl.sh - Run BFCL (the Berkeley Function Call Leaderboard) with its official package, bfcl-eval:
# its generation, its checkers and its score aggregation. Self-contained: no framework of this repo in
# between (scripts/custom/bfcl.py is the whole driver; docs/custom-bfcl.md). The package and its dependencies
# come from requirements/bfcl-runtime.lock and requirements/bfcl-eval.txt: baked into the prebuilt image (containers/Dockerfile.eval,
# EVAL_PREBUILT_ENV=1, nothing installed), elsewhere installed into a cached venv (scripts/common/locked_venv.sh).
# The model under test is any OpenAI-compatible endpoint, reached as run_tau2_bench.sh reaches its model.
#
# Usage:
#   scripts/custom/run_bfcl.sh --version <v3|v4> --model <name> --api-base-url <url> [options]
#
# Required:
#   --version <v3|v4>          v3: the non-live, live and multi-turn categories (4441 entries);
#                               v4: those and the memory categories (4906). The web search categories need a
#                               SerpAPI key and are not run: add them with --test-category.
#   --model <name>             The served model name (function-calling mode).
#   --api-base-url <url>       OpenAI-compatible endpoint serving --model (bare host or /v1 root).
#                               Its key: TARGET_API_KEY (default: CSCS_SERVING_API; both also from
#                               scripts/secrets.env, as in run_inspect_eval.sh).
#
# Options:
#   --workdir <path>           Results (<workdir>/result), scores (<workdir>/score) and summary.json
#                               (default: logs/bfcl/<model>-<version>). Rerunning with the same workdir resumes:
#                               bfcl-eval skips the entries it already has.
#   --num-threads <n>          Parallel requests (default: 32).
#   --request-timeout <s>      Seconds per request (default: 3600; the OpenAI SDK's own 600 cuts off a few long
#                               thinking answers per run).
#   --num-tasks <n>            Only the first n entries of each category (a smoke test); scores then cover
#                               that subset and are not the leaderboard's.
#   --test-category <name>     Replace the version's categories (repeatable; bfcl-eval's names or groups).
#   --temperature V, --top-p V, --top-k V, --min-p V, --repetition-penalty V
#                              Sampling. bfcl-eval's own default is a temperature of 0.001 and no other setting;
#                              nothing else is sent unless you pass it (docs/sampling.md).
#   --sampling-profile <name>  A profile of the sampling config (e.g. thinking, greedy);
#   --sampling-config <file>   another file instead of configs/sampling_defaults.json.
#   --enable-thinking, --no-enable-thinking, --reasoning-effort <level>
#                              Thinking and reasoning effort (scripts/common/thinking.py).
#   --name <name>              The run's name in W&B (default: the model name, "/" and ":" as "-").
#   --wandb-entity <entity>    W&B entity and project to upload summary.json to (also WANDB_ENTITY, WANDB_PROJECT;
#   --wandb-project <project>   default apertus/e-p-t-pool), when WANDB_API_KEY is set (scripts/common/wandb.sh).
#   --no-wandb                 Don't upload (also WANDB_MODE=disabled).
#   --cache-dir <path>         Outside the image: where the venv is cached (default:
#                               ${EVAL_ENV_CACHE_ROOT:-$SCRATCH/eval-envs}).
#
# Results: <workdir>/summary.json holds the score of every category, the official groups (non-live, live,
# multi-turn, agentic) and, for v3, the overall (the mean of its three groups, as bfcl-eval 3.0.0 computed it);
# bfcl-eval's own tables are <workdir>/score/data_*.csv. Without all four groups there is no official v4 overall.
#
# Example (smoke test against a CSCS-served model):
#   scripts/custom/run_bfcl.sh --version v3 --model CSCS-Inference/swiss-ai/Apertus-v1.5-8B \
#     --api-base-url https://api.swissai.svc.cscs.ch/v1 --num-tasks 2

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

die() { echo "$*" >&2; exit 1; }

usage() {
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}

VERSION=""
MODEL=""
API_BASE_URL=${API_BASE_URL:-""}
WORKDIR=""
NUM_THREADS=32
REQUEST_TIMEOUT=3600
NUM_TASKS=""
CATEGORY_ARGS=()
SAMPLING_ARGS=(--framework bfcl --format litellm)
THINKING_ARGS=(--format json)
NAME=""
NO_WANDB=0

while (( $# > 0 )); do
    case "$1" in
        --version) VERSION=$2; shift 2 ;;
        --model) MODEL=$2; shift 2 ;;
        --api-base-url) API_BASE_URL=$2; shift 2 ;;
        --workdir) WORKDIR=$2; shift 2 ;;
        --num-threads) NUM_THREADS=$2; shift 2 ;;
        --request-timeout) REQUEST_TIMEOUT=$2; shift 2 ;;
        --num-tasks) NUM_TASKS=$2; shift 2 ;;
        --test-category) CATEGORY_ARGS+=(--test-category "$2"); shift 2 ;;
        --temperature|--top-p|--top-k|--min-p|--repetition-penalty)
            SAMPLING_ARGS+=("$1" "$2"); shift 2 ;;
        --sampling-profile) SAMPLING_ARGS+=(--profile "$2"); shift 2 ;;
        --sampling-config) SAMPLING_ARGS+=(--config "$2"); shift 2 ;;
        --enable-thinking|--no-enable-thinking)
            SAMPLING_ARGS+=("$1"); THINKING_ARGS+=("$1"); shift ;;
        --reasoning-effort)
            SAMPLING_ARGS+=("$1" "$2"); THINKING_ARGS+=("$1" "$2"); shift 2 ;;
        --name) NAME=$2; shift 2 ;;
        --wandb-entity) WANDB_ENTITY=$2; shift 2 ;;
        --wandb-project) WANDB_PROJECT=$2; shift 2 ;;
        --no-wandb) NO_WANDB=1; shift ;;
        --cache-dir) export EVAL_ENV_CACHE_ROOT=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown argument: $1 (see --help)" ;;
    esac
done

[[ "$VERSION" == v3 || "$VERSION" == v4 ]] || die "--version must be v3 or v4. See --help."
[[ -n "$MODEL" ]] || die "Missing --model. See --help."
[[ -n "$API_BASE_URL" ]] || die "Missing --api-base-url. See --help."
[[ "$API_BASE_URL" == */v1 || "$API_BASE_URL" == */v1/ ]] || API_BASE_URL="${API_BASE_URL%/}/v1"
WORKDIR=${WORKDIR:-"logs/bfcl/$(tr '/' '_' <<<"$MODEL")-$VERSION"}

# shellcheck source=scripts/common/secrets.sh
source scripts/common/secrets.sh
# shellcheck source=scripts/common/wandb.sh
source scripts/common/wandb.sh
if (( NO_WANDB )); then wandb_setup "--no-wandb"; else wandb_setup || exit 1; fi
NAME=${NAME:-$(echo "$MODEL" | tr '/: ' '---')}
export TARGET_API_KEY="${TARGET_API_KEY:-${CSCS_SERVING_API:-}}"

# Sampling (scripts/common/sampling.py), then thinking and reasoning effort (scripts/common/thinking.py), as the
# keyword arguments of every request: temperature goes to the handler, the rest to chat.completions.create.
SAMPLING=$(python3 -m scripts.common.sampling "${SAMPLING_ARGS[@]}") || exit 1
THINKING=$(python3 -m scripts.common.thinking "${THINKING_ARGS[@]}") || exit 1
LLM_ARGS=$(python3 -c '
import json, sys
from scripts.common import thinking
args, values = json.loads(sys.argv[1]), json.loads(sys.argv[2])
args.update({k: v for k, v in thinking.litellm_kwargs(values).items() if k != "extra_body"})
body = thinking.merge_extra_body(args.get("extra_body"), values)
if body:
    args["extra_body"] = body
print(json.dumps(args))' "$SAMPLING" "$THINKING")

if [[ "${EVAL_PREBUILT_ENV:-0}" == "1" ]]; then
    # The prebuilt image (containers/Dockerfile.eval) has the venv: nothing is installed at run time.
    [[ -n "${BFCL_VENV:-}" && -x "$BFCL_VENV/bin/python" ]] ||
        die "EVAL_PREBUILT_ENV=1 but BFCL_VENV (${BFCL_VENV:-unset}) has no bin/python:" \
            "this image has no BFCL environment, and a prebuilt image installs nothing at run time."
    LOCKED_VENV=$BFCL_VENV
    echo "EVAL_PREBUILT_ENV=1: using $BFCL_VENV (no install)"
else
    # shellcheck source=scripts/common/locked_venv.sh
    source scripts/common/locked_venv.sh
    locked_venv requirements/bfcl-runtime.lock 3.11 requirements/bfcl-eval.txt || die "Installing requirements/bfcl-runtime.lock failed."
fi

ARGS=(--version "$VERSION" --model "$MODEL" --api-base-url "$API_BASE_URL" --workdir "$WORKDIR"
      --llm-args "$LLM_ARGS" --num-threads "$NUM_THREADS" --request-timeout "$REQUEST_TIMEOUT")
[[ -z "$NUM_TASKS" ]] || ARGS+=(--num-tasks "$NUM_TASKS")
(( ${#CATEGORY_ARGS[@]} == 0 )) || ARGS+=("${CATEGORY_ARGS[@]}")

echo "+ python -m scripts.custom.bfcl --version $VERSION --model $MODEL --api-base-url $API_BASE_URL" \
     "--workdir $WORKDIR --num-threads $NUM_THREADS ${NUM_TASKS:+--num-tasks $NUM_TASKS} (llm args not shown)"
# Written at the end of the run: one left by an earlier run here must not be taken (and uploaded) for this one's.
rm -f "$WORKDIR/summary.json"
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" "$LOCKED_VENV/bin/python" -m scripts.custom.bfcl "${ARGS[@]}"

if (( WANDB_UPLOAD )); then
    echo ""
    echo "Uploading results to wandb"
    PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" "$LOCKED_VENV/bin/python" -m scripts.results.update_wandb_bfcl \
        --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" --name "$NAME" --work-dir "$WORKDIR"
fi
