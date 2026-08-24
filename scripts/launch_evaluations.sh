#!/bin/bash

# launch_evaluations.sh - Launch Apertus benchmark suite evaluations
#
# Usage:
#   bash launch_evaluations.sh <mode> [options]
#
# Modes:

# Apertus:
#   default          - Apertus multilingual suite
#   multi-lingual    - Multi-lingual suite (taken from 1.0)
#   apertus-previous - Apertus previous benchmark suite (from 1.0)
#   best-of-k        - Multi-repeat/self-consistency suite
#   gpt              - Experimental OpenAI GPT-judge chat suite (future harness support)
#   eval-debug       - Small set of loglikelihood and generative benchmarks to test eval script
#   single           - Run a single task (requires --task <task_name>)

#
# Olmo3:
#   olmo-easy        - Base Easy Suite (minerva_math, mmlu, hellaswag, ...)
#   olmo-main        - Base Main Suite (gsm8k_cot, humaneval, arc, ...)
#   olmo-heldout     - Held-out Suite (mmlu_pro, bbh)
#   olmo-safety      - Safety (harmbench, toxigen, wmdp, bbq)
#   olmo-longcontext - Long-Context (RULER)
#   olmo-complete    - All suites combined (default, excludes long-context)

#
# Model selection (pick one):
#   --model <path>            - Single HF model or local checkpoint path. Required unless
#                               --script is given, or --backend openai is used with
#                               --api-model-name (which then also fills --model's slot).
#   --script <path>           - Run a model-list script (e.g. hf_eval_multiple_other_models.sh)
#   --megatron-iter <iter>    - For Megatron models, specify the iteration number to evaluate
#                               (e.g. 8926), defaults to "latest"
#
# Options:
#   --name <name>        - Override the eval run name (default: auto-derived from model path)
#   --task <task>         - Task name for 'single' mode (e.g. hellaswag, gsm8k_cot)
#   --chat-template      - Apply chat template (auto-detected for Instruct/Chat/SFT/DPO models)
#   --no-chat-template   - Force disable chat template
#   --tokenizer <tok>    - Custom tokenizer (default: same as model)
#   --bos                - Prepend BOS token
#   --num-fewshot N      - Override num_fewshot for all tasks (default: use task YAML defaults)
#                          Note: tasks with num_fewshot=0 in YAML are never overridden.
#                          OLMo3 uses 5-shot for most MC tasks; pass --num-fewshot 5 to match.
#   --backend <backend>  - lm-eval backend: hf, vllm, sglang, megatron_lm, openai (default: from sbatch script)
#   --api-base-url <url> - OpenAI-compatible endpoint for the 'openai' backend (required with it,
#                          unless API_BASE_URL is exported). Bare host, /v1 root, or full endpoint URL.
#   --api-model-name <n> - 'model' field sent in API requests. Required with the 'openai' backend
#                          (unless API_MODEL_NAME is exported) -- --model's value may be
#                          catalog-prefixed and not what the gateway expects. If --model is
#                          omitted, it defaults to this value so single-model dispatch still runs.
#   --splits K           - Split tasks across K parallel nodes per model
#   --limit N            - Optional argument to pass as --limit to the lm-evaluation-harness, to limit the number of samples per task (default: no limit).
#   --harness-branch B   - Install lm-evaluation-harness from branch/ref B (default: repo default branch)
#   --reservation <name> - Submit jobs under a SLURM reservation, including an auto-launched judge
#                          (exported as SBATCH_RESERVATION; ambient SBATCH_RESERVATION is respected
#                          for evaluation jobs when the flag is absent)
#   --judge <none|auto|preset> - Judge model control:
#                          none (default): disable judge auto-launch
#                          auto: detect judge-dependent tasks and launch needed judges
#                          <preset>: launch a specific preset (qwen3.5-27b, llama-3.3-70b)
#   --judge-args <str>   - Extra arguments forwarded to scripts/launch_judge.py
#   --keep-judge         - Do not auto-cancel judge model after evaluation finishes
#
# Thinking / reasoning metrics (hf and vllm backends only):
#   --thinking           - Umbrella flag: make the model reason AND record the thinking metrics.
#                          Implies --enable-thinking, --autodetect-think-tokens (unless
#                          --think-end-token is given), --track-thinking-metrics true,
#                          --log-length-metrics, and forces the chat template on.
#   --enable-thinking    - Chat-template argument: let the model reason. On its own this records
#   --no-enable-thinking   NOTHING - a reasoning close token must also be known.
#   --think-end-token <s>  - Force the reasoning close token, e.g. '</think>'. Arms the trace
#                            strip and the thinking metrics.
#   --think-start-token <s> - Force the reasoning open token, e.g. '<think>'. Needed for
#                             thinking_format_has_open; without it thinking_format_correct
#                             degrades to == thinking_format_has_close.
#   --autodetect-think-tokens - Read the open/close tokens from the model's chat template.
#   --track-thinking-metrics <true|false>  - Force the thinking metrics on/off.
#   --no-track-thinking-metrics              Default: on iff a close token is known.
#   --log-length-metrics - Aggregate response_length_* / thinking_length_* into results and W&B.
#                          thinking_format_* is aggregated regardless.
#
# Examples:
#   # Single HF model, auto-detect everything
#   bash launch_evaluations.sh complete --model meta-llama/Llama-3.1-8B-Instruct
#
#   # Single model with splits
#   bash launch_evaluations.sh main --model allenai/OLMo-2-1124-7B --splits 4
#
#   # Base model, explicit no chat template
#   bash launch_evaluations.sh easy --model Qwen/Qwen2.5-7B --no-chat-template
#
#   # Run a multi-model script
#   bash launch_evaluations.sh complete --script runners/hf_eval_multiple_other_models.sh
#
#   # Run a single task
#   bash launch_evaluations.sh single --task hellaswag --model meta-llama/Llama-3.1-8B-Instruct

set -euo pipefail

# --- Argument parsing ---
EVAL_MODE=${1:-complete}
shift || true

NUM_SPLITS=1
MODEL_PATH=""
MODEL_NAME=""
SCRIPT_PATH=""
CHAT_TEMPLATE_OVERRIDE=""  # "", "true", "false"
CUSTOM_TOKENIZER=""
BOS_FLAG=""
BACKEND_FLAG=""
API_BASE_URL_FLAG=""
API_MODEL_NAME_FLAG=""
FEWSHOT_FLAG=""
# Keep an ambient HARNESS_LIMIT (the graceful launcher has no --limit flag, so callers export
# it); blanking it here would ship the cleared value into the job. --limit still overrides.
HARNESS_LIMIT="${HARNESS_LIMIT:-}"
MEGATRON_ITER=""
SINGLE_TASK=""
HARNESS_BRANCH=""
RESERVATION_FLAG=""
JUDGE_MODE="none"       # auto, none, or a preset name
JUDGE_EXTRA_ARGS=""
KEEP_JUDGE="false"
THINKING_UMBRELLA=""
ENABLE_THINKING_OVERRIDE=""   # "", "true", "false"
THINK_END_TOKEN=""
THINK_START_TOKEN=""
AUTODETECT_THINK_TOKENS=""
TRACK_THINKING_METRICS=""     # "", "true", "false"
LOG_LENGTH_METRICS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL_PATH="$2";              shift 2 ;;
        --name)         MODEL_NAME="$2";              shift 2 ;;
        --script)       SCRIPT_PATH="$2";             shift 2 ;;
        --splits)       NUM_SPLITS="$2";              shift 2 ;;
        --num-fewshot)  FEWSHOT_FLAG="$2";            shift 2 ;;
        --task)         SINGLE_TASK="$2";             shift 2 ;;
        --chat-template)    CHAT_TEMPLATE_OVERRIDE="true";  shift ;;
        --no-chat-template) CHAT_TEMPLATE_OVERRIDE="false"; shift ;;
        --tokenizer)    CUSTOM_TOKENIZER="$2";        shift 2 ;;
        --bos)          BOS_FLAG="true";              shift ;;
        --backend)      BACKEND_FLAG="$2";            shift 2 ;;
        --api-base-url) API_BASE_URL_FLAG="$2";       shift 2 ;;
        --api-model-name) API_MODEL_NAME_FLAG="$2";   shift 2 ;;
        --megatron-iter) MEGATRON_ITER="$2";            shift 2 ;;
        --limit) HARNESS_LIMIT="$2";            shift 2 ;;
        --harness-branch) HARNESS_BRANCH="$2";        shift 2 ;;
        --reservation)   RESERVATION_FLAG="$2";        shift 2 ;;
        --judge)         JUDGE_MODE="$2";              shift 2 ;;
        --judge-args)    JUDGE_EXTRA_ARGS="$2";        shift 2 ;;
        --keep-judge)    KEEP_JUDGE="true";            shift ;;
        --thinking)                  THINKING_UMBRELLA="true";          shift ;;
        --enable-thinking)           ENABLE_THINKING_OVERRIDE="true";   shift ;;
        --no-enable-thinking)        ENABLE_THINKING_OVERRIDE="false";  shift ;;
        --think-end-token)           THINK_END_TOKEN="$2";              shift 2 ;;
        --think-start-token)         THINK_START_TOKEN="$2";            shift 2 ;;
        --autodetect-think-tokens)   AUTODETECT_THINK_TOKENS="true";    shift ;;
        --track-thinking-metrics)    TRACK_THINKING_METRICS="$2";       shift 2 ;;
        --no-track-thinking-metrics) TRACK_THINKING_METRICS="false";    shift ;;
        --log-length-metrics)        LOG_LENGTH_METRICS="true";         shift ;;
        *)
            echo "Error: Unknown option '$1'"
            echo "Run with no arguments for usage."
            exit 1
            ;;
    esac
done

# --- Validate mode ---
VALID_MODES=("default" "multi-lingual" "apertus-previous" "pretrain" "posttrain" "best-of-k" "gpt" "olmo-easy" "olmo-main" "olmo-heldout" "olmo-safety" "olmo-longcontext" "olmo-complete" "eval-debug" "single" "custom")
if [[ ! " ${VALID_MODES[*]} " =~ " ${EVAL_MODE} " ]]; then
    echo "Error: Invalid mode '$EVAL_MODE'"
    echo "Valid modes: ${VALID_MODES[*]}"
    exit 1
fi

# --- Validate single mode ---
if [[ "$EVAL_MODE" == "single" ]]; then
    if [[ -z "$SINGLE_TASK" ]]; then
        echo "Error: 'single' mode requires --task <task_name>"
        echo "Example: bash launch_evaluations.sh single --task hellaswag --model meta-llama/Llama-3.1-8B"
        exit 1
    fi
elif [[ -n "$SINGLE_TASK" ]]; then
    echo "Error: --task can only be used with 'single' mode"
    exit 1
fi

if (( NUM_SPLITS < 1 )); then
    echo "Error: --splits must be >= 1"
    exit 1
fi

if [[ -n "$MEGATRON_ITER" ]] && [[ "$MEGATRON_ITER" != "latest" ]] && [[ ! "$MEGATRON_ITER" =~ ^[0-9]+$ ]]; then
    echo "Error: --megatron-iter must be an integer or 'latest' (got '$MEGATRON_ITER')"
    exit 1
fi

# Can't specify both --model and --script
if [[ -n "$MODEL_PATH" && -n "$SCRIPT_PATH" ]]; then
    echo "Error: --model and --script are mutually exclusive"
    exit 1
fi

# --- Resolve thinking / reasoning configuration ---
# Must run before dispatch so the forced chat template reaches all model-selection modes.
if [[ -n "$TRACK_THINKING_METRICS" && "$TRACK_THINKING_METRICS" != "true" && "$TRACK_THINKING_METRICS" != "false" ]]; then
    echo "Error: --track-thinking-metrics expects 'true' or 'false' (got '$TRACK_THINKING_METRICS')"
    exit 1
fi
# lm_eval splits --model_args on commas, so a comma in a token would inject an extra key.
for _tok_var in THINK_END_TOKEN THINK_START_TOKEN; do
    if [[ "${!_tok_var}" == *,* ]]; then
        echo "Error: ${_tok_var} must not contain a comma (got '${!_tok_var}')"
        exit 1
    fi
done

if [[ "$THINKING_UMBRELLA" == "true" ]]; then
    [[ -z "$ENABLE_THINKING_OVERRIDE" ]] && ENABLE_THINKING_OVERRIDE="true"
    [[ -z "$TRACK_THINKING_METRICS"   ]] && TRACK_THINKING_METRICS="true"
    [[ -z "$LOG_LENGTH_METRICS"       ]] && LOG_LENGTH_METRICS="true"
    # A close token must come from somewhere; prefer the user's if they named one.
    [[ -z "$THINK_END_TOKEN" && -z "$AUTODETECT_THINK_TOKENS" ]] && AUTODETECT_THINK_TOKENS="true"
fi

# Two distinct questions (conflating them rejected every "off" switch):
#   THINKING_TOUCHED       - any thinking/length flag was passed; drives the megatron guard.
#   THINKING_METRICS_ASKED - the user asked to RECORD metrics; needs a close token + chat template.
# --log-length-metrics alone is neither: response_length_* is recorded for any generative response.
THINKING_TOUCHED="false"
if [[ "$THINKING_UMBRELLA" == "true" || -n "$ENABLE_THINKING_OVERRIDE" \
      || -n "$THINK_END_TOKEN" || -n "$THINK_START_TOKEN" \
      || "$AUTODETECT_THINK_TOKENS" == "true" \
      || -n "$TRACK_THINKING_METRICS" || "$LOG_LENGTH_METRICS" == "true" ]]; then
    THINKING_TOUCHED="true"
fi

THINKING_METRICS_ASKED="false"
if [[ "$THINKING_UMBRELLA" == "true" || "$ENABLE_THINKING_OVERRIDE" == "true" \
      || "$TRACK_THINKING_METRICS" == "true" \
      || -n "$THINK_END_TOKEN" || -n "$THINK_START_TOKEN" \
      || "$AUTODETECT_THINK_TOKENS" == "true" ]]; then
    THINKING_METRICS_ASKED="true"
fi

# Length/reasoning producers exist only for hf/vllm/sglang. Resolve the backend as
# evaluate.sbatch will, so an ambient LM_EVAL_BACKEND fails here, not after scheduling.
EFFECTIVE_BACKEND="${BACKEND_FLAG:-${LM_EVAL_BACKEND:-}}"
if [[ "$THINKING_TOUCHED" == "true" && ( "$EFFECTIVE_BACKEND" == "megatron_lm" || "$EFFECTIVE_BACKEND" == "openai" ) ]]; then
    echo "Error: thinking and length metrics are not supported with the $EFFECTIVE_BACKEND backend"
    exit 1
fi

# The openai backend needs an endpoint; fail here, not after scheduling.
if [[ "$EFFECTIVE_BACKEND" == "openai" && -z "${API_BASE_URL_FLAG:-${API_BASE_URL:-}}" ]]; then
    echo "Error: --backend openai requires --api-base-url <url> (or an exported API_BASE_URL)"
    exit 1
fi
# --model defaults API_MODEL_NAME (evaluate.sbatch) to its own, possibly catalog-prefixed
# value, which the gateway may not recognize -- require the caller to say explicitly what
# the server should see rather than relying on that default.
if [[ "$EFFECTIVE_BACKEND" == "openai" && -z "${API_MODEL_NAME_FLAG:-${API_MODEL_NAME:-}}" ]]; then
    echo "Error: --backend openai requires --api-model-name <name> (or an exported API_MODEL_NAME)"
    exit 1
fi
if [[ -n "$API_BASE_URL_FLAG" || -n "$API_MODEL_NAME_FLAG" ]] && [[ "$EFFECTIVE_BACKEND" != "openai" ]]; then
    echo "Error: --api-base-url/--api-model-name only apply with --backend openai"
    exit 1
fi
# Non-openai backends load a checkpoint in-job, so a model has to be named one way or
# another: --model for a single checkpoint, --script for a model-list. Without either, this
# used to fall through silently to the hardcoded default EVALUATION_SCRIPTS array below --
# exactly the trap that bit the missing-'--model' openai case. Require it explicitly instead.
if [[ "$EFFECTIVE_BACKEND" != "openai" && -z "$MODEL_PATH" && -z "$SCRIPT_PATH" ]]; then
    echo "Error: --model or --script is required (unless --backend openai is used with --api-model-name)"
    exit 1
fi
[[ -n "$API_BASE_URL_FLAG"   ]] && export API_BASE_URL="$API_BASE_URL_FLAG"
[[ -n "$API_MODEL_NAME_FLAG" ]] && export API_MODEL_NAME="$API_MODEL_NAME_FLAG"

# The openai backend loads no local checkpoint, so --api-model-name (flag or ambient
# API_MODEL_NAME, same fallback the requiredness check above used) alone identifies the
# model. Default --model from it so single-model dispatch (MODE 1 below) still triggers
# when the caller only passed --api-model-name.
if [[ -z "$MODEL_PATH" && -z "$SCRIPT_PATH" && "$EFFECTIVE_BACKEND" == "openai" ]]; then
    MODEL_PATH="${API_MODEL_NAME_FLAG:-$API_MODEL_NAME}"
fi

if [[ "$THINKING_METRICS_ASKED" == "true" ]]; then
    # The reasoning tokens live in the chat template, so it has to be rendered.
    if [[ "$CHAT_TEMPLATE_OVERRIDE" == "false" ]]; then
        echo "Error: thinking metrics require the chat template; drop --no-chat-template"
        exit 1
    fi
    CHAT_TEMPLATE_OVERRIDE="true"

    # Without a close token nothing is stripped or recorded -- the run silently produces nothing.
    if [[ -z "$THINK_END_TOKEN" && "$AUTODETECT_THINK_TOKENS" != "true" ]]; then
        echo "Error: thinking metrics requested but no reasoning close token is known."
        echo "       Pass --think-end-token '</think>', --autodetect-think-tokens, or use --thinking."
        exit 1
    fi

    if [[ -z "$THINK_START_TOKEN" && "$AUTODETECT_THINK_TOKENS" != "true" ]]; then
        echo "WARNING: no reasoning open token (--think-start-token). thinking_format_has_open"
        echo "         will not be recorded and thinking_format_correct degrades to == has_close."
    fi
fi

if [[ "$TRACK_THINKING_METRICS" == "false" && "$LOG_LENGTH_METRICS" == "true" ]]; then
    echo "WARNING: --track-thinking-metrics false drops thinking_length_*;"
    echo "         --log-length-metrics will only aggregate response_length_*."
fi

# Export only what was set: unset ENABLE_THINKING keeps the hf template default; unset
# TRACK_THINKING_METRICS lets the harness derive it.
[[ -n "$ENABLE_THINKING_OVERRIDE"       ]] && export ENABLE_THINKING="$ENABLE_THINKING_OVERRIDE"
[[ -n "$THINK_END_TOKEN"                ]] && export THINK_END_TOKEN
[[ -n "$THINK_START_TOKEN"              ]] && export THINK_START_TOKEN
[[ "$AUTODETECT_THINK_TOKENS" == "true" ]] && export AUTODETECT_THINK_TOKENS="true"
[[ -n "$TRACK_THINKING_METRICS"         ]] && export TRACK_THINKING_METRICS
[[ "$LOG_LENGTH_METRICS" == "true"      ]] && export LOG_LENGTH_METRICS="true"

# --- Environment defaults ---
# sbatch reads SBATCH_RESERVATION natively (CLI > env > script directives).
[[ -n "$RESERVATION_FLAG" ]] && export SBATCH_RESERVATION="$RESERVATION_FLAG"
export WANDB_ENTITY=${WANDB_ENTITY:-apertus}
export WANDB_PROJECT=${WANDB_PROJECT:-test-new-evals-pipeline}
export NUM_SPLITS
export SBATCH_SCRIPT=${SBATCH_SCRIPT:-scripts/evaluate.sbatch}
# Global checkpoint iteration override for Megatron checkpoints.
# Consumed by the runner and forwarded to evaluate.sbatch as CKPT_ITER.
[[ -n "$MEGATRON_ITER" ]] && export CKPT_ITERATION="$MEGATRON_ITER"

# --- Configure task suite ---
case "$EVAL_MODE" in
    "default")
        export TASKS=./configs/apertus/tasks_default.txt
        export TABLE_METRICS=./configs/apertus/tasks_default_main_table.txt
        ;;
    "multi-lingual")
        export TASKS=./configs/apertus/tasks_multilingual.txt
        export TABLE_METRICS=./configs/apertus/tasks_multilingual_main_table.txt
        ;;
    "apertus-previous")
        export TASKS=./configs/apertus/tasks_english.txt
        export TABLE_METRICS=./configs/apertus/tasks_english_main_table.txt
        ;;
    "pretrain")
        export TASKS=./configs/apertus/tasks_pretrain.txt
        export TABLE_METRICS=./configs/apertus/tasks_pretrain_main_table.txt
        ;;
    "posttrain")
        export TASKS=./configs/apertus/tasks_posttrain_final.txt
        export TABLE_METRICS=./configs/apertus/tasks_posttrain_final_main_table.txt
        ;;
    "best-of-k")
        export TASKS=./configs/apertus/tasks_best_of_k.txt
        export TABLE_METRICS=./configs/apertus/tasks_best_of_k_main_table.txt
        ;;
    "gpt")
        export TASKS=./configs/apertus/tasks_gpt.txt
        export TABLE_METRICS=./configs/apertus/tasks_gpt_main_table.txt
        [[ -z "$CHAT_TEMPLATE_OVERRIDE" ]] && CHAT_TEMPLATE_OVERRIDE="true"
        ;;
    "olmo-easy")
        export TASKS=./configs/olmo/olmo3_easy.txt
        export TABLE_METRICS=./configs/olmo/olmo3_easy_main_table.txt
        ;;
    "olmo-main")
        export TASKS=./configs/olmo/olmo3_main.txt
        export TABLE_METRICS=./configs/olmo/olmo3_main_main_table.txt
        ;;
    "olmo-heldout")
        export TASKS=./configs/olmo/olmo3_heldout.txt
        export TABLE_METRICS=./configs/olmo/olmo3_heldout_main_table.txt
        ;;
    "olmo-safety")
        export TASKS=./configs/olmo/olmo3_safety.txt
        export TABLE_METRICS=./configs/olmo/olmo3_safety_main_table.txt
        ;;
    "olmo-longcontext")
        export TASKS=./configs/olmo/olmo3_longcontext.txt
        export TABLE_METRICS=./configs/olmo/olmo3_longcontext_main_table.txt
        ;;
    "olmo-complete")
        export TASKS=./configs/olmo/olmo3_complete.txt
        export TABLE_METRICS=./configs/olmo/olmo3_complete_main_table.txt
        ;;
    "eval-debug")
        export TASKS=./configs/apertus/eval_debug.txt
        export TABLE_METRICS=./configs/olmo/eval_debug_main_table.txt
        ;;
    "single")
        export TASKS="$SINGLE_TASK"
        export TABLE_METRICS="$SINGLE_TASK"
        ;;
    "custom")
        ;;
esac

# The GPT path is deliberately only a suite/config placeholder for now. It uses
# the normal Swiss-AI harness and will gain an explicit judge selector once that
# support lands upstream; do not pass a speculative flag today.
if [[ "$EVAL_MODE" == "gpt" ]]; then
    echo "WARNING: gpt mode is experimental; no judge-type flag is passed to lm-eval yet." >&2
    echo "         It requires the corresponding GPT-judge support to exist in the Swiss-AI harness." >&2
    if [[ -z "${OPENAI_API_KEY:-}" && ! -f ./scripts/openai_api_key.txt ]]; then
        echo "WARNING: neither OPENAI_API_KEY nor scripts/openai_api_key.txt is available." >&2
    fi
fi

# --- Validate split count vs task count ---
if (( NUM_SPLITS > 1 )); then
    TASK_COUNT=$(grep -v '^\s*#' "$TASKS" | grep -v '^\s*$' | wc -l | tr -d ' ')
    if (( TASK_COUNT < NUM_SPLITS )); then
        echo "WARNING: Only $TASK_COUNT tasks but $NUM_SPLITS splits requested. Reducing."
        NUM_SPLITS=$TASK_COUNT
        export NUM_SPLITS
    fi
fi

# --- Auto-derive name and chat template for --model mode ---
auto_detect_chat_template() {
    local model="$1"
    # Check for common instruct/chat model name patterns
    if [[ "$model" =~ -[Ii]nstruct ]] || \
       [[ "$model" =~ -[Cc]hat ]] || \
       [[ "$model" =~ -[Ss][Ff][Tt] ]] || \
       [[ "$model" =~ -[Dd][Pp][Oo] ]] || \
       [[ "$model" =~ -[Ii]t$ ]] || \
       [[ "$model" =~ -aligned ]]; then
        echo "true"
    else
        echo "false"
    fi
}

auto_derive_name() {
    local model="$1"
    # For HF paths like "meta-llama/Llama-3.1-8B-Instruct" -> "Llama-3.1-8B-Instruct"
    # For local paths like "/capstor/.../checkpoint-8926" -> last meaningful dir component
    if [[ "$model" == */* && "$model" != /* ]]; then
        # HF-style org/model path
        echo "${model##*/}"
    elif [[ "$model" == /* ]]; then
        # Local path - use the last directory component that isn't "checkpoint-*"
        local basename
        basename=$(basename "$model")
        if [[ "$basename" =~ ^checkpoint- ]]; then
            basename=$(basename "$(dirname "$model")")
        fi
        echo "$basename"
    else
        echo "$model"
    fi
}

# --- Print configuration ---
echo "======================================"
echo "Apertus Evaluation Launcher"
echo "  Mode:   $EVAL_MODE"
[[ "$EVAL_MODE" == "single" ]] && echo "  Task:   $SINGLE_TASK"
echo "  Splits: $NUM_SPLITS"
echo "  Harness: auto (Swiss-AI; ymetz only for BFCL/Charter)${HARNESS_BRANCH:+@$HARNESS_BRANCH}"
if [[ "$THINKING_TOUCHED" == "true" ]]; then
    echo "  Thinking: enable=${ENABLE_THINKING_OVERRIDE:-<unset>} autodetect=${AUTODETECT_THINK_TOKENS:-false} track=${TRACK_THINKING_METRICS:-<derive>} lengths=${LOG_LENGTH_METRICS:-false}"
    [[ -n "$THINK_START_TOKEN" || -n "$THINK_END_TOKEN" ]] && echo "  Think tokens: start='${THINK_START_TOKEN:-<none>}' end='${THINK_END_TOKEN:-<none>}'"
fi
if [[ "$EFFECTIVE_BACKEND" == "openai" ]]; then
    echo "  API:    ${API_BASE_URL} (model=${API_MODEL_NAME:-<from --model>})"
fi

# --- Few-shot override ---
[[ -n "$FEWSHOT_FLAG" ]] && export NUM_FEWSHOT="$FEWSHOT_FLAG"

# --- Harness limit override ---
[[ -n "$HARNESS_LIMIT" ]] && export HARNESS_LIMIT="$HARNESS_LIMIT"
[[ -n "$HARNESS_BRANCH" ]] && export LM_EVAL_HARNESS_BRANCH="$HARNESS_BRANCH"

# --- Judge model launch - if none is set, rely on already hosted judge or manual launch ---
JUDGE_JOB_IDS=""
JUDGE_TASKS_PATTERN="alpaca_eval|multijail|aya_redteaming|arena_hard_v01|arena_hard_v2|harmbench|hallulens|realtoxicitypromptsllama"

if [[ "$JUDGE_MODE" != "none" ]]; then

    NEEDS_JUDGE=false
    JUDGE_LAUNCH_ARGS=""

    if [[ "$JUDGE_MODE" == "auto" ]]; then
        # Delegate detection to launch_judge.py so TASK_TO_JUDGE remains the
        # single source of truth for automatic judge selection.
        NEEDS_JUDGE=true
        JUDGE_LAUNCH_ARGS="--detect-from-tasks $TASKS"
    else
        # Explicit preset
        NEEDS_JUDGE=true
        JUDGE_LAUNCH_ARGS="--preset $JUDGE_MODE"
    fi

    if [[ "$NEEDS_JUDGE" == "true" ]]; then
        echo ""
        echo "--- Judge Model Launch ---"
        if [[ -n "$RESERVATION_FLAG" ]]; then
            JUDGE_LAUNCH_ARGS="$JUDGE_LAUNCH_ARGS --reservation $RESERVATION_FLAG"
        fi
        # Capture machine-readable output (JUDGE_JOB_ID=...) from stdout,
        # while letting human-readable logs flow to stderr (visible to user).
        JUDGE_STDOUT=$(python3 scripts/launch_judge.py $JUDGE_LAUNCH_ARGS $JUDGE_EXTRA_ARGS)
        JUDGE_EXIT=$?

        if [[ $JUDGE_EXIT -ne 0 ]]; then
            echo "ERROR: Judge model launch failed (exit code $JUDGE_EXIT)"
            exit 1
        fi

        JUDGE_JOB_IDS=$(echo "$JUDGE_STDOUT" | grep "^JUDGE_JOB_ID=" | cut -d= -f2 | tr '\n' ' ')
        JUDGE_MODELS=$(echo "$JUDGE_STDOUT" | grep "^JUDGE_MODEL_NAME=" | cut -d= -f2 | tr '\n' ', ')

        if [[ -n "$JUDGE_JOB_IDS" ]]; then
            echo "  Judge jobs: $JUDGE_JOB_IDS"
            echo "  Judge models: $JUDGE_MODELS"
            export JUDGE_JOB_IDS
        fi
        echo "--------------------------"
        echo ""
    fi
fi

# warn if tasks are detected but judge is explicitly disabled (mode=none)
if [[ "$JUDGE_MODE" == "none" && "$EVAL_MODE" != "gpt" ]]; then
    if [[ -f "$TASKS" ]]; then
        if grep -qE "$JUDGE_TASKS_PATTERN" "$TASKS"; then
            echo "WARNING: Detected judge-dependent tasks but judge model launching is disabled (--judge none)"
            echo "Make sure a judge model is available via the CSCS serving platform or launch it manually"
        fi
    fi
fi

# --- Dispatch based on model selection mode ---

if [[ -n "$MODEL_PATH" ]]; then
    # ===== MODE 1: Single model =====
    if [[ -z "$MODEL_NAME" ]]; then
        MODEL_NAME=$(auto_derive_name "$MODEL_PATH")
    fi

    if [[ -z "$CHAT_TEMPLATE_OVERRIDE" ]]; then
        export APPLY_CHAT_TEMPLATE=$(auto_detect_chat_template "$MODEL_PATH")
    else
        export APPLY_CHAT_TEMPLATE="$CHAT_TEMPLATE_OVERRIDE"
    fi

    [[ -n "$CUSTOM_TOKENIZER" ]] && export TOKENIZER="$CUSTOM_TOKENIZER"
    [[ -n "$BOS_FLAG" ]] && export BOS="$BOS_FLAG"
    [[ -n "$BACKEND_FLAG" ]] && export LM_EVAL_BACKEND="$BACKEND_FLAG"

    echo "  Model:  $MODEL_PATH"
    echo "  Name:   $MODEL_NAME"
    echo "  Checkpoint Iter: ${MEGATRON_ITER:-N/A} (only applies to local Megatron checkpoints)"
    echo "  Chat:   $APPLY_CHAT_TEMPLATE"
    [[ -n "$CUSTOM_TOKENIZER" ]] && echo "  Tok:    $CUSTOM_TOKENIZER"
    [[ -n "$BOS_FLAG" ]] && echo "  BOS:    $BOS_FLAG"
    [[ -n "$FEWSHOT_FLAG" ]] && echo "  Fewshot: $FEWSHOT_FLAG"
    [[ -n "$HARNESS_BRANCH" ]] && echo "  Harness branch: $HARNESS_BRANCH"
    echo "  W&B:    $WANDB_ENTITY/$WANDB_PROJECT"
    echo "======================================"

    # Build a single-model checkpoint array and source the runner
    declare -A MODEL_CHECKPOINTS=(
        ["$MODEL_NAME"]="$MODEL_PATH"
    )
    source runners/hf_base_runner.sh "model"

else
    # ===== MODE 2: Run a model-list script =====
    # SCRIPT_PATH is guaranteed non-empty here: the requiredness check above ensures
    # --model or --script is always set (or --model is defaulted from --api-model-name),
    # and --model/--script are mutually exclusive.
    if [[ ! -f "$SCRIPT_PATH" ]]; then
        echo "Error: Script not found: $SCRIPT_PATH"
        exit 1
    fi

    [[ -n "$CHAT_TEMPLATE_OVERRIDE" ]] && export APPLY_CHAT_TEMPLATE="$CHAT_TEMPLATE_OVERRIDE"
    [[ -n "$CUSTOM_TOKENIZER" ]] && export TOKENIZER="$CUSTOM_TOKENIZER"
    [[ -n "$BOS_FLAG" ]] && export BOS="$BOS_FLAG"
    [[ -n "$BACKEND_FLAG" ]] && export LM_EVAL_BACKEND="$BACKEND_FLAG"

    echo "  Script: $SCRIPT_PATH"
    [[ -n "$HARNESS_BRANCH" ]] && echo "  Harness branch: $HARNESS_BRANCH"
    echo "  W&B:    $WANDB_ENTITY/$WANDB_PROJECT"
    echo "======================================"

    bash "$SCRIPT_PATH"
fi

# --- Judge cleanup job ---
# After all eval jobs are submitted, schedule a cleanup job that cancels judge
# SLURM jobs once all evaluations finish.
if [[ -n "$JUDGE_JOB_IDS" && "$KEEP_JUDGE" != "true" && ${#EVAL_JOB_IDS[@]} -gt 0 ]]; then
    DEP_STRING=$(IFS=':'; echo "${EVAL_JOB_IDS[*]}")
    SCANCEL_CMD="scancel $JUDGE_JOB_IDS"
    CLEANUP_JOB=$(sbatch --parsable \
        --account=infra01 \
        --partition=normal \
        --job-name "judge-cleanup" \
        --dependency="afterany:${DEP_STRING}" \
        --time=00:05:00 \
        --wrap="$SCANCEL_CMD")
    echo "Judge cleanup job $CLEANUP_JOB will cancel judge(s) [$JUDGE_JOB_IDS] after evals finish"
fi
