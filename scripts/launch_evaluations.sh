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
#   eval-debug       - Small set of loglikelihood and generative benchmarks to test eval script
#   single           - Run a single task (requires --task <task_name>)
#   non-gated        - Subset of default with non swiss-ai gated datasets (todo: full access)

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
#   --model <path>            - Single HF model or local checkpoint path
#   --script <path>           - Run a model-list script (e.g. hf_eval_multiple_other_models.sh)
#   (neither)                 - Uses the EVALUATION_SCRIPTS array defined below
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
#   --backend <backend>  - lm-eval backend: hf, vllm, megatron_lm (default: from sbatch script)
#   --splits K           - Split tasks across K parallel nodes per model
#   --limit N            - Optional argument to pass as --limit to the lm-evaluation-harness, to limit the number of samples per task (default: no limit).
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
#   # Use default EVALUATION_SCRIPTS (edit the array below)
#   bash launch_evaluations.sh complete --splits 4
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
FEWSHOT_FLAG=""
HARNESS_LIMIT=""
MEGATRON_ITER=""
SINGLE_TASK=""

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
        --megatron-iter) MEGATRON_ITER="$2";            shift 2 ;;
        --limit) HARNESS_LIMIT="$2";            shift 2 ;;
        *)
            echo "Error: Unknown option '$1'"
            echo "Run with no arguments for usage."
            exit 1
            ;;
    esac
done

# --- Validate mode ---
VALID_MODES=("default" "multi-lingual" "apertus-previous" "pretrain" "olmo-easy" "olmo-main" "olmo-heldout" "olmo-safety" "olmo-longcontext" "olmo-complete" "eval-debug" non-gated "single" "custom")
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

# --- Environment defaults ---
export WANDB_ENTITY=${WANDB_ENTITY:-apertus}
export WANDB_PROJECT=${WANDB_PROJECT:-apertus-1.5-post-training-v0.0}
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
        export WANDB_PROJECT="apertus-1.5-pre-training-v0.0"
        ;;
    "olmo-easy")
        export TASKS=./configs/olmo/olmo3_easy.txt
        export TABLE_METRICS=./configs/olmo/olmo3_easy_main_table.txt
        export WANDB_PROJECT="${WANDB_PROJECT}-olmo-easy"
        ;;
    "olmo-main")
        export TASKS=./configs/olmo/olmo3_main.txt
        export TABLE_METRICS=./configs/olmo/olmo3_main_main_table.txt
        export WANDB_PROJECT="${WANDB_PROJECT}-olmo-main"
        ;;
    "olmo-heldout")
        export TASKS=./configs/olmo/olmo3_heldout.txt
        export TABLE_METRICS=./configs/olmo/olmo3_heldout_main_table.txt
        export WANDB_PROJECT="${WANDB_PROJECT}-olmo-heldout"
        ;;
    "olmo-safety")
        export TASKS=./configs/olmo/olmo3_safety.txt
        export TABLE_METRICS=./configs/olmo/olmo3_safety_main_table.txt
        export WANDB_PROJECT="${WANDB_PROJECT}-olmo-safety"
        ;;
    "olmo-longcontext")
        export TASKS=./configs/olmo/olmo3_longcontext.txt
        export TABLE_METRICS=./configs/olmo/olmo3_longcontext_main_table.txt
        export WANDB_PROJECT="${WANDB_PROJECT}-olmo-longcontext"
        ;;
    "olmo-complete")
        export TASKS=./configs/olmo/olmo3_complete.txt
        export TABLE_METRICS=./configs/olmo/olmo3_complete_main_table.txt
        export WANDB_PROJECT="${WANDB_PROJECT}-olmo-complete"
        ;;
    "eval-debug")
        export TASKS=./configs/apertus/eval_debug.txt
        export TABLE_METRICS=./configs/olmo/eval_debug_main_table.txt
        ;;
    "non-gated")
        export TASKS=./configs/apertus/tasks_non_gated.txt
        export TABLE_METRICS=./configs/olmo/eval_debug_main_table.txt
        ;;
    "single")
        export TASKS="$SINGLE_TASK"
        export TABLE_METRICS="$SINGLE_TASK"
        export WANDB_PROJECT="${WANDB_PROJECT}-single"
        ;;
    "custom")
        ;;
esac

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

# --- Few-shot override ---
[[ -n "$FEWSHOT_FLAG" ]] && export NUM_FEWSHOT="$FEWSHOT_FLAG"

# --- Harness limit override ---
[[ -n "$HARNESS_LIMIT" ]] && export HARNESS_LIMIT="$HARNESS_LIMIT"

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
    echo "  W&B:    $WANDB_ENTITY/$WANDB_PROJECT"
    echo "======================================"

    # Build a single-model checkpoint array and source the runner
    declare -A MODEL_CHECKPOINTS=(
        ["$MODEL_NAME"]="$MODEL_PATH"
    )
    source runners/hf_base_runner.sh "model"

elif [[ -n "$SCRIPT_PATH" ]]; then
    # ===== MODE 2: Run a model-list script =====
    if [[ ! -f "$SCRIPT_PATH" ]]; then
        echo "Error: Script not found: $SCRIPT_PATH"
        exit 1
    fi

    [[ -n "$CHAT_TEMPLATE_OVERRIDE" ]] && export APPLY_CHAT_TEMPLATE="$CHAT_TEMPLATE_OVERRIDE"
    [[ -n "$CUSTOM_TOKENIZER" ]] && export TOKENIZER="$CUSTOM_TOKENIZER"
    [[ -n "$BOS_FLAG" ]] && export BOS="$BOS_FLAG"
    [[ -n "$BACKEND_FLAG" ]] && export LM_EVAL_BACKEND="$BACKEND_FLAG"

    echo "  Script: $SCRIPT_PATH"
    echo "  W&B:    $WANDB_ENTITY/$WANDB_PROJECT"
    echo "======================================"

    bash "$SCRIPT_PATH"

else
    # ===== MODE 3: Default EVALUATION_SCRIPTS array =====
    [[ -n "$CHAT_TEMPLATE_OVERRIDE" ]] && export APPLY_CHAT_TEMPLATE="$CHAT_TEMPLATE_OVERRIDE"
    [[ -n "$CUSTOM_TOKENIZER" ]] && export TOKENIZER="$CUSTOM_TOKENIZER"
    [[ -n "$BOS_FLAG" ]] && export BOS="$BOS_FLAG"
    [[ -n "$BACKEND_FLAG" ]] && export LM_EVAL_BACKEND="$BACKEND_FLAG"

    # Edit this array to select which model-list scripts to run
    EVALUATION_SCRIPTS=(
        "runners/hf_eval_multiple_apertus_base_models.sh"
        # "runners/hf_eval_multiple_apertus_models.sh"
        # "runners/hf_eval_multiple_other_base_models.sh"
        # "runners/hf_eval_multiple_other_models.sh"
    )

    echo "  Scripts:"
    for script in "${EVALUATION_SCRIPTS[@]}"; do
        echo "    - $script"
    done
    echo "  W&B:    $WANDB_ENTITY/$WANDB_PROJECT"
    echo "======================================"

    for script in "${EVALUATION_SCRIPTS[@]}"; do
        echo ""
        echo "Launching: $script"
        echo "----------------------------------------"
        bash "$script"
    done
fi
