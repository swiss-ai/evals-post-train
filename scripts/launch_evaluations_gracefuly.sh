#!/bin/bash

set -eo pipefail

# ==========================================
# CACHE REDIRECTION
# ==========================================
export CACHE_ROOT="${SCRATCH:-/tmp}/.cache"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$CACHE_ROOT/huggingface/datasets"
export NLTK_DATA="$CACHE_ROOT/nltk_data"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export WANDB_DIR="$CACHE_ROOT/wandb"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb_cache"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export TIKTOKEN_CACHE_DIR="$CACHE_ROOT/tiktoken"
export PYTHONUSERBASE="$CACHE_ROOT/python_userbase"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$NLTK_DATA" "$TRITON_CACHE_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$MPLCONFIGDIR" "$TIKTOKEN_CACHE_DIR" "$PYTHONUSERBASE"

# --- Defaults ---
EVAL_PREFIX="$SCRATCH/eval_logs_start/apertus/apertus-1.5-post-training-v0.0/"
ACCOUNT="infra01"
RESERVATION="SD-69241-apertus-1-5-0"
WANDB_ENTITY="apertus"
WANDB_PROJECT="apertus-1.5-post-training-v0.0"
TABLE_METRICS=""
DEBUG=0
MERGE_ONLY=0
GROUP_SIZE=1
TASK_FILE=""
MODEL=""
FORCE_TASKS=""
TOKENIZER=""
RUN_NAME=""
# Thinking flags are forwarded verbatim to the inner `launch_evaluations.sh single` runs (which
# load the model), never to the --merge_only aggregator re-invocation (which loads none).
declare -a THINKING_ARGS=()
WANTS_THINK=0   # see run-name isolation below

# --- Argument Parsing ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task_file) TASK_FILE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --eval_prefix) EVAL_PREFIX="$2"; shift 2 ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --reservation) RESERVATION="$2"; shift 2 ;;
        --wandb_entity) WANDB_ENTITY="$2"; shift 2 ;;
        --wandb_project) WANDB_PROJECT="$2"; shift 2 ;;
        --table_metrics) TABLE_METRICS="$2"; shift 2 ;;
        --group_size) GROUP_SIZE="$2"; shift 2 ;;
        --debug) DEBUG=1; shift 1 ;;
        --merge_only) MERGE_ONLY=1; shift 1 ;;
        --force_tasks) FORCE_TASKS="$2"; shift 2 ;;
        --tokenizer) TOKENIZER="$2"; shift 2 ;;
        --name) RUN_NAME="$2"; shift 2 ;;
        --thinking|--enable-thinking) THINKING_ARGS+=("$1"); WANTS_THINK=1; shift 1 ;;
        --no-enable-thinking)        THINKING_ARGS+=("$1"); shift 1 ;;
        --autodetect-think-tokens)   THINKING_ARGS+=("$1"); shift 1 ;;
        --no-track-thinking-metrics) THINKING_ARGS+=("$1"); shift 1 ;;
        --log-length-metrics)        THINKING_ARGS+=("$1"); shift 1 ;;
        --think-end-token)           THINKING_ARGS+=("$1" "$2"); shift 2 ;;
        --think-start-token)         THINKING_ARGS+=("$1" "$2"); shift 2 ;;
        --track-thinking-metrics)    THINKING_ARGS+=("$1" "$2"); shift 2 ;;
        *) echo "Error: Unknown argument '$1'"; exit 1 ;;
    esac
done

if [[ -z "$TASK_FILE" || -z "$MODEL" ]]; then
    echo "Error: --task_file and --model are required."
    exit 1
fi

MODEL_BASENAME=$(basename "${MODEL%/}")

# Run-name isolation: a reasoning run must not collide with the same model's non-thinking eval,
# in the COMPLETED_MAP scan or the W&B run. Suffix "-think" when the run actually reasons
# (--thinking / --enable-thinking); --name overrides the whole run name.
RUN_BASENAME="$MODEL_BASENAME"
(( WANTS_THINK )) && RUN_BASENAME="${MODEL_BASENAME}-think"
[[ -n "$RUN_NAME" ]] && RUN_BASENAME="$RUN_NAME"
# Non-default run names are threaded to the per-task jobs and the aggregator; default names
# keep the launcher's auto-derived name.
if [[ "$RUN_BASENAME" != "$MODEL_BASENAME" ]]; then NAME_ISOLATED=1; else NAME_ISOLATED=0; fi

MAIN_HARNESS_DIR="$EVAL_PREFIX/$RUN_BASENAME/harness"
SINGLE_EVAL_PREFIX="${EVAL_PREFIX/$WANDB_PROJECT/${WANDB_PROJECT}-single}"
SINGLE_HARNESS_DIR="$SINGLE_EVAL_PREFIX/$RUN_BASENAME/harness"

declare -a ORDERED_TASKS
while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"        # strip inline comments
    line=$(echo "$line" | xargs)
    if [[ -n "$line" ]]; then
        ORDERED_TASKS+=("$line")
    fi
done < "$TASK_FILE"

declare -A COMPLETED_MAP

scan_harness_dir() {
    local target_dir="$1"
    
    if [[ $DEBUG -eq 1 ]]; then 
        echo -e "\n[DEBUG] --- Scanning Directory ---"
        echo "[DEBUG] Target: $target_dir"
    fi

    if [[ -d "$target_dir" ]]; then
        while IFS= read -r edir; do
            if [[ "$(basename "$edir")" == *"eval_merged"* ]]; then continue; fi
            
            while IFS= read -r res_file; do
                if [[ $DEBUG -eq 1 ]]; then echo "[DEBUG] Found JSON: $res_file"; fi
                for task in "${ORDERED_TASKS[@]}"; do
                    # Safe grep search without '|| true' causing false positives
                    if grep -E -q "\"$task\"[[:space:]]*:" "$res_file" 2>/dev/null; then
                        COMPLETED_MAP["$task"]="$edir"
                        if [[ $DEBUG -eq 1 ]]; then echo "[DEBUG]   -> Marked complete: $task"; fi
                    fi
                done
            done < <(find "$edir" -type f -name "results_*.json")
        done < <(find "$target_dir" -mindepth 1 -maxdepth 1 -type d -name "eval_*" | sort)
    fi
}

scan_harness_dir "$MAIN_HARNESS_DIR"
scan_harness_dir "$SINGLE_HARNESS_DIR"

rebuild_split_markers() {
    local split_dir="$MAIN_HARNESS_DIR/split_markers"
    if [[ $DEBUG -eq 1 ]]; then
        echo -e "\n[DEBUG] Would wipe and rebuild $split_dir"
    else
        echo -e "\nRebuilding $split_dir mapping..."
        rm -rf "$split_dir"
        mkdir -p "$split_dir"
        for i in "${!ORDERED_TASKS[@]}"; do
            local task="${ORDERED_TASKS[$i]}"
            if [[ -n "${COMPLETED_MAP[$task]:-}" ]]; then
                echo "${COMPLETED_MAP[$task]}" > "$split_dir/split_${i}.txt"
            fi
        done
    fi
}

submit_aggregator() {
    export WANDB_ENTITY="$WANDB_ENTITY"
    export WANDB_PROJECT="$WANDB_PROJECT"
    export NUM_SPLITS="${#ORDERED_TASKS[@]}"
    
    # Export metrics file safely
    if [[ -n "$TABLE_METRICS" ]]; then
        export TABLE_METRICS=$(realpath "$TABLE_METRICS")
    fi
    
    # Clean the path so the aggregator can rebuild it properly
    local CLEAN_PREFIX="${EVAL_PREFIX%/}"
    CLEAN_PREFIX="${CLEAN_PREFIX%/$WANDB_PROJECT}"
    CLEAN_PREFIX="${CLEAN_PREFIX%/$WANDB_ENTITY}"
    export LOGS_ROOT="$CLEAN_PREFIX"
    
    # Pass positional arguments to aggregate_splits.sbatch
    local agg_cmd=("sbatch" "--account" "$ACCOUNT")
    if [[ -n "$RESERVATION" ]]; then agg_cmd+=("--reservation" "$RESERVATION"); fi
    agg_cmd+=("scripts/aggregate_splits.sbatch" "$MODEL" "$RUN_BASENAME")
    
    if [[ $DEBUG -eq 1 ]]; then
        echo -e "\n[DEBUG] Would submit aggregator: ${agg_cmd[*]}"
    else
        echo -e "\nSubmitting aggregator: ${agg_cmd[*]}"
        "${agg_cmd[@]}"
    fi
}

# Force re-evaluation: remove specified tasks from COMPLETED_MAP
# FORCE_TASKS is a comma-separated list of task substrings (e.g. "mbpp_instruct,humaneval_instruct")
if [[ -n "$FORCE_TASKS" ]]; then
    IFS=',' read -ra FORCE_LIST <<< "$FORCE_TASKS"
    for task in "${ORDERED_TASKS[@]}"; do
        for force_pat in "${FORCE_LIST[@]}"; do
            if [[ "$task" == *"$force_pat"* ]]; then
                if [[ -n "${COMPLETED_MAP[$task]:-}" ]]; then
                    echo "Forcing re-evaluation of: $task (matched pattern '$force_pat')"
                    unset "COMPLETED_MAP[$task]"
                fi
            fi
        done
    done
fi

if [[ $MERGE_ONLY -eq 1 ]]; then
    echo "--- Running Post-Eval Cleanup for $RUN_BASENAME ---"
    rebuild_split_markers
    submit_aggregator
    exit 0
fi

declare -a MISSING_TASKS

for task in "${ORDERED_TASKS[@]}"; do
    if [[ -z "${COMPLETED_MAP[$task]:-}" ]]; then
        MISSING_TASKS+=("$task")
    fi
done

echo -e "\nModel: $MODEL_BASENAME (run name: $RUN_BASENAME)"
echo "Total expected tasks: ${#ORDERED_TASKS[@]}"
echo "Successfully completed tasks: ${#COMPLETED_MAP[@]}"
echo "Missing tasks: ${#MISSING_TASKS[@]}"
echo "Completed benchmarks: ${!COMPLETED_MAP[*]}"

if [[ ${#MISSING_TASKS[@]} -eq 0 ]]; then
    echo -e "\nAll tasks completed! Triggering marker rebuild and aggregation."
    rebuild_split_markers
    submit_aggregator
    exit 0
fi

declare -a JOB_IDS

# Group missing tasks into batches of GROUP_SIZE
declare -a TASK_GROUPS
num_missing=${#MISSING_TASKS[@]}
for (( i=0; i<num_missing; i+=GROUP_SIZE )); do
    group=""
    for (( j=i; j<i+GROUP_SIZE && j<num_missing; j++ )); do
        if [[ -n "$group" ]]; then
            group="${group},${MISSING_TASKS[$j]}"
        else
            group="${MISSING_TASKS[$j]}"
        fi
    done
    TASK_GROUPS+=("$group")
done

num_groups=${#TASK_GROUPS[@]}
echo -e "\nLaunching $num_missing missing tasks in $num_groups groups (group_size=$GROUP_SIZE):"
for group in "${TASK_GROUPS[@]}"; do
    launch_cmd=("bash" "scripts/launch_evaluations.sh" "single" "--task" "$group" "--model" "$MODEL" "--chat-template")
    if [[ ${#THINKING_ARGS[@]} -gt 0 ]]; then launch_cmd+=("${THINKING_ARGS[@]}"); fi
    (( NAME_ISOLATED )) && launch_cmd+=("--name" "$RUN_BASENAME")
    if [[ -n "$TOKENIZER" ]]; then launch_cmd+=("--tokenizer" "$TOKENIZER"); fi

    if [[ $DEBUG -eq 1 ]]; then
        echo "[DEBUG] Would launch: WANDB_MODE=disabled SBATCH_ACCOUNT=$ACCOUNT SBATCH_RESERVATION=$RESERVATION ${launch_cmd[*]}"
        JOB_IDS+=("999999")
        continue
    fi

    set +e
    output=$(export WANDB_MODE=disabled && export SBATCH_ACCOUNT="$ACCOUNT" && { [[ -n "$RESERVATION" ]] && export SBATCH_RESERVATION="$RESERVATION" || unset SBATCH_RESERVATION; } && export WANDB_ENTITY="$WANDB_ENTITY" && export WANDB_PROJECT="$WANDB_PROJECT" && "${launch_cmd[@]}" 2>&1)
    rc=$?
    set -e

    job_id=$(echo "$output" | grep -oE 'Submitted batch job [0-9]+' | awk '{print $4}' || true)

    if [[ -n "$job_id" ]]; then
        JOB_IDS+=("$job_id")
        echo " -> Submitted [$group] (Job ID: $job_id)"
    else
        echo " -> Failed to submit [$group]:"
        echo "$output"
    fi
done

if [[ ${#JOB_IDS[@]} -gt 0 ]]; then
    dep_str="afterok"
    for id in "${JOB_IDS[@]}"; do
        dep_str="${dep_str}:${id}"
    done
    
    SELF_PATH=$(realpath "$0")
    
    WRAP_CMD="bash $SELF_PATH --task_file \"$TASK_FILE\" --model \"$MODEL\" --eval_prefix \"$EVAL_PREFIX\" --account \"$ACCOUNT\" --wandb_entity \"$WANDB_ENTITY\" --wandb_project \"$WANDB_PROJECT\" --group_size \"$GROUP_SIZE\""
    if [[ -n "$RESERVATION" ]]; then
        WRAP_CMD="$WRAP_CMD --reservation \"$RESERVATION\""
    fi
    
    if [[ -n "$TABLE_METRICS" ]]; then
        WRAP_CMD="$WRAP_CMD --table_metrics \"$TABLE_METRICS\""
    fi

    if [[ -n "$FORCE_TASKS" ]]; then
        WRAP_CMD="$WRAP_CMD --force_tasks \"$FORCE_TASKS\""
    fi

    if [[ -n "$TOKENIZER" ]]; then
        WRAP_CMD="$WRAP_CMD --tokenizer \"$TOKENIZER\""
    fi

    # The aggregator loads no model, so thinking flags aren't forwarded -- but it must still
    # scan/aggregate under the isolated run name.
    (( NAME_ISOLATED )) && WRAP_CMD="$WRAP_CMD --name \"$RUN_BASENAME\""

    WRAP_CMD="$WRAP_CMD --merge_only"
    
    merge_launch_cmd=("sbatch" "--account" "$ACCOUNT")
    if [[ -n "$RESERVATION" ]]; then merge_launch_cmd+=("--reservation" "$RESERVATION"); fi
    merge_launch_cmd+=("--dependency=$dep_str" "--wrap" "$WRAP_CMD")
    
    if [[ $DEBUG -eq 1 ]]; then
        echo -e "\n[DEBUG] Would queue the marker reconstructor and aggregator using:"
        echo "[DEBUG] ${merge_launch_cmd[*]}"
    else
        echo -e "\nQueueing marker reconstruction and aggregation to run after tasks complete..."
        "${merge_launch_cmd[@]}"
    fi
fi