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
#   --harness-branch B   - Install lm-evaluation-harness from branch/ref B (default: repo default branch)
#   --standalone-edf E   - CSCS CE Environment Definition File for standalone jobs
#   --sandbox-backend B  - Sandbox provider hint for standalone benchmarks
#   --container-cache-backend B - Container cache backend for standalone jobs
#   --local-registry-home P - Path to local-registry checkout
#   --local-registry-dir P  - Local registry data directory for this job
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
HARNESS_BRANCH=""
STANDALONE_TASKS=""
STANDALONE_EDF=""
SANDBOX_BACKEND_FLAG=""
CONTAINER_CACHE_BACKEND_FLAG=""
LOCAL_REGISTRY_HOME_FLAG=""
LOCAL_REGISTRY_DIR_FLAG=""

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
        --harness-branch) HARNESS_BRANCH="$2";        shift 2 ;;
        --standalone-edf) STANDALONE_EDF="$2"; shift 2 ;;
        --sandbox-backend) SANDBOX_BACKEND_FLAG="$2"; shift 2 ;;
        --container-cache-backend) CONTAINER_CACHE_BACKEND_FLAG="$2"; shift 2 ;;
        --local-registry-home) LOCAL_REGISTRY_HOME_FLAG="$2"; shift 2 ;;
        --local-registry-dir) LOCAL_REGISTRY_DIR_FLAG="$2"; shift 2 ;;
        *)
            echo "Error: Unknown option '$1'"
            echo "Run with no arguments for usage."
            exit 1
            ;;
    esac
done

# --- Validate mode ---
VALID_MODES=("default" "multi-lingual" "apertus-previous" "pretrain" "olmo-easy" "olmo-main" "olmo-heldout" "olmo-safety" "olmo-longcontext" "olmo-complete" "eval-debug" "non-gated" "single" "claritas")
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
export STANDALONE_SBATCH_SCRIPT=${STANDALONE_SBATCH_SCRIPT:-scripts/evaluate_standalone.sbatch}
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
    "claritas")
        export TASKS=./configs/apertus/tasks_claritas.txt
        export TABLE_METRICS=./configs/olmo/eval_debug_main_table.txt
        export WANDB_PROJECT="claritas-benchmarks"
        ;;
    "single")
        export TASKS="$SINGLE_TASK"
        export TABLE_METRICS="$SINGLE_TASK"
        export WANDB_PROJECT="${WANDB_PROJECT}-single"
        ;;
esac

read_task_list() {
    local source="$1"
    if [[ -f "$source" ]]; then
        grep -v '^\s*#' "$source" | grep -v '^\s*$'
    else
        echo "$source" | tr ',' '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | grep -v '^$'
    fi
}

standalone_registry_names() {
    for cfg in configs/standalone/benchmarks/*.toml; do
        [[ -f "$cfg" ]] || continue
        sed -n 's/^name[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$cfg" | head -n 1
    done
}

is_standalone_task() {
    local task="$1"
    local registered
    for registered in "${REGISTERED_STANDALONE_TASKS[@]}"; do
        [[ "$task" == "$registered" ]] && return 0
    done
    return 1
}

join_by_comma() {
    local IFS=,
    echo "$*"
}

standalone_metric_specs() {
    local task="$1"
    local cfg metric
    for cfg in configs/standalone/benchmarks/*.toml; do
        [[ -f "$cfg" ]] || continue
        if [[ "$(sed -n 's/^name[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$cfg" | head -n 1)" != "$task" ]]; then
            continue
        fi
        while IFS= read -r metric; do
            [[ -n "$metric" ]] && echo "$task/$metric"
        done < <(awk '
            /^\[\[metrics\]\]/ { in_metric=1; next }
            /^\[/ && $0 !~ /^\[\[metrics\]\]/ { in_metric=0 }
            in_metric && /^name[[:space:]]*=/ {
                gsub(/"/, "", $3)
                print $3
            }
        ' "$cfg")
    done
}

standalone_table_metrics_file() {
    if [[ "${STANDALONE_TASKS:-}" == *"swebench_verified"* ]]; then
        echo configs/standalone/swebench_main_table.txt
    else
        echo configs/standalone/smoke_main_table.txt
    fi
}

configure_standalone_runtime_defaults() {
    if [[ -z "$STANDALONE_EDF" ]]; then
        if [[ "${BACKEND_FLAG:-${LM_EVAL_BACKEND:-vllm}}" == "vllm" ]]; then
            export STANDALONE_EDF=./containers/env_vllm.toml
        else
            export STANDALONE_EDF=./containers/env.toml
        fi
    else
        export STANDALONE_EDF
    fi
    export STANDALONE_LIMIT="$HARNESS_LIMIT"
    export SANDBOX_BACKEND=${SANDBOX_BACKEND_FLAG:-none}
    export CONTAINER_CACHE_BACKEND=${CONTAINER_CACHE_BACKEND_FLAG:-none}
    [[ -n "$LOCAL_REGISTRY_HOME_FLAG" ]] && export LOCAL_REGISTRY_HOME="$LOCAL_REGISTRY_HOME_FLAG"
    [[ -n "$LOCAL_REGISTRY_DIR_FLAG" ]] && export LOCAL_REGISTRY_DIR="$LOCAL_REGISTRY_DIR_FLAG"
    return 0
}

LM_EVAL_TASK_ITEMS=()
STANDALONE_TASK_ITEMS=()
LM_EVAL_TASK_COUNT=0
STANDALONE_TASK_COUNT=0

REGISTERED_STANDALONE_TASKS=()
while IFS= read -r registered_task; do
    [[ -n "$registered_task" ]] && REGISTERED_STANDALONE_TASKS+=("$registered_task")
done < <(standalone_registry_names)

while IFS= read -r task_name; do
    if is_standalone_task "$task_name"; then
        STANDALONE_TASK_ITEMS+=("$task_name")
        STANDALONE_TASK_COUNT=$((STANDALONE_TASK_COUNT + 1))
    else
        LM_EVAL_TASK_ITEMS+=("$task_name")
        LM_EVAL_TASK_COUNT=$((LM_EVAL_TASK_COUNT + 1))
    fi
done < <(read_task_list "$TASKS")

if (( STANDALONE_TASK_COUNT > 0 )); then
    export RUN_STANDALONE=true
    STANDALONE_TASKS=$(join_by_comma "${STANDALONE_TASK_ITEMS[@]}")
    export STANDALONE_TASKS
    configure_standalone_runtime_defaults
else
    export RUN_STANDALONE=false
    if [[ -n "$STANDALONE_EDF" || -n "$SANDBOX_BACKEND_FLAG" || -n "$CONTAINER_CACHE_BACKEND_FLAG" || -n "$LOCAL_REGISTRY_HOME_FLAG" || -n "$LOCAL_REGISTRY_DIR_FLAG" ]]; then
        echo "Error: standalone options were provided, but no standalone tasks were requested."
        exit 1
    fi
fi

if (( LM_EVAL_TASK_COUNT > 0 )); then
    export RUN_LM_EVAL=true
    if (( STANDALONE_TASK_COUNT > 0 )); then
        PARTITION_DIR=logs/task_partitions
        mkdir -p "$PARTITION_DIR"
        LM_TASKS_FILE="$PARTITION_DIR/${EVAL_MODE}_lm_eval_$$_tasks.txt"
        printf "%s\n" "${LM_EVAL_TASK_ITEMS[@]}" > "$LM_TASKS_FILE"
        export TASKS="$LM_TASKS_FILE"
    fi
else
    export RUN_LM_EVAL=false
    export TASKS=""
fi

if [[ "$RUN_STANDALONE" == "true" && ( "$RUN_LM_EVAL" == "false" || -z "${TABLE_METRICS:-}" ) ]]; then
    TABLE_METRICS=$(standalone_table_metrics_file)
    export TABLE_METRICS
fi

if [[ "$RUN_LM_EVAL" == "true" && "$RUN_STANDALONE" == "true" ]]; then
    PARTITION_DIR=logs/task_partitions
    mkdir -p "$PARTITION_DIR"
    COMBINED_TABLE_METRICS="$PARTITION_DIR/${EVAL_MODE}_mixed_$$_main_table.txt"
    if [[ -f "$TABLE_METRICS" ]]; then
        grep -v '^\s*#' "$TABLE_METRICS" | grep -v '^\s*$' > "$COMBINED_TABLE_METRICS"
    else
        echo "$TABLE_METRICS" | tr ' ' '\n' | grep -v '^$' > "$COMBINED_TABLE_METRICS"
    fi
    for standalone_task in "${STANDALONE_TASK_ITEMS[@]}"; do
        standalone_metric_specs "$standalone_task" >> "$COMBINED_TABLE_METRICS"
    done
    export TABLE_METRICS="$COMBINED_TABLE_METRICS"
fi

# --- Validate split count vs task count ---
if [[ "$RUN_LM_EVAL" == "false" && "$RUN_STANDALONE" == "true" && "$NUM_SPLITS" -gt 1 ]]; then
    echo "Error: standalone-only launches do not support --splits yet. Use benchmark-native sharding once a runner exposes it."
    exit 1
fi

if [[ "$RUN_LM_EVAL" == "true" && "$NUM_SPLITS" -gt 1 ]]; then
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
if [[ "$RUN_STANDALONE" == "true" ]]; then
    [[ -n "${STANDALONE_TASKS:-}" ]] && echo "  Standalone tasks: $STANDALONE_TASKS"
    [[ "$RUN_LM_EVAL" == "true" ]] && echo "  LM tasks: $LM_EVAL_TASK_COUNT"
    echo "  EDF:    $STANDALONE_EDF"
    echo "  Sandbox: $SANDBOX_BACKEND"
    echo "  Cache:  $CONTAINER_CACHE_BACKEND"
    [[ -n "${LOCAL_REGISTRY_HOME:-}" ]] && echo "  Local registry home: $LOCAL_REGISTRY_HOME"
    [[ -n "${LOCAL_REGISTRY_DIR:-}" ]] && echo "  Local registry dir: $LOCAL_REGISTRY_DIR"
fi
echo "  Splits: $NUM_SPLITS"

# --- Few-shot override ---
[[ -n "$FEWSHOT_FLAG" ]] && export NUM_FEWSHOT="$FEWSHOT_FLAG"

# --- Harness limit override ---
[[ -n "$HARNESS_LIMIT" ]] && export HARNESS_LIMIT="$HARNESS_LIMIT"
[[ -n "$HARNESS_BRANCH" ]] && export LM_EVAL_HARNESS_BRANCH="$HARNESS_BRANCH"

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
    declare -A MODEL_CHECKPOINTS
    MODEL_CHECKPOINTS["$MODEL_NAME"]="$MODEL_PATH"
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
    [[ -n "$HARNESS_BRANCH" ]] && echo "  Harness branch: $HARNESS_BRANCH"
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
    [[ -n "$HARNESS_BRANCH" ]] && echo "  Harness branch: $HARNESS_BRANCH"
    echo "  W&B:    $WANDB_ENTITY/$WANDB_PROJECT"
    echo "======================================"

    for script in "${EVALUATION_SCRIPTS[@]}"; do
        echo ""
        echo "Launching: $script"
        echo "----------------------------------------"
        bash "$script"
    done
fi
