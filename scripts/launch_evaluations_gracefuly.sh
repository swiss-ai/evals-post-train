#!/bin/bash

# Backward-compatible wrapper. New code should call launch_evaluations.sh with
# --failure-policy resume and --chunk-size directly.
set -euo pipefail

EVAL_PREFIX="${SCRATCH:-/tmp}/eval_logs_start/apertus/apertus-1.5-post-training-v0.0/"
ACCOUNT="infra01"
RESERVATION=""
WANDB_ENTITY="apertus"
WANDB_PROJECT="apertus-1.5-post-training-v0.0"
TABLE_METRICS=""
TASK_FILE=""
MODEL=""
CHUNK_SIZE=1
FORCE_TASKS=""
TOKENIZER=""
RUN_NAME=""
declare -a FORWARDED_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task_file|--task-file) TASK_FILE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --eval_prefix|--eval-prefix) EVAL_PREFIX="$2"; shift 2 ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --reservation) RESERVATION="$2"; shift 2 ;;
        --wandb_entity|--wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
        --wandb_project|--wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
        --table_metrics|--table-metrics) TABLE_METRICS="$2"; shift 2 ;;
        --group_size|--group-size|--chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
        --force_tasks|--force-tasks) FORCE_TASKS="$2"; shift 2 ;;
        --tokenizer) TOKENIZER="$2"; shift 2 ;;
        --name) RUN_NAME="$2"; shift 2 ;;
        --merge_only|--merge-only) FORWARDED_ARGS+=(--merge-only); shift ;;
        --debug) FORWARDED_ARGS+=(--debug); shift ;;
        --thinking|--enable-thinking|--no-enable-thinking|--autodetect-think-tokens|--no-track-thinking-metrics|--log-length-metrics)
            FORWARDED_ARGS+=("$1"); shift ;;
        --think-end-token|--think-start-token|--track-thinking-metrics)
            FORWARDED_ARGS+=("$1" "$2"); shift 2 ;;
        *) echo "Error: Unknown argument '$1'" >&2; exit 1 ;;
    esac
done

[[ -n "$TASK_FILE" && -n "$MODEL" ]] || {
    echo "Error: --task_file and --model are required" >&2
    exit 1
}

# The legacy prefix ended at <entity>/<project>; the unified launcher accepts
# the root above those two components.
LOGS_ROOT=${EVAL_PREFIX%/}
LOGS_ROOT=${LOGS_ROOT%/$WANDB_PROJECT}
LOGS_ROOT=${LOGS_ROOT%/$WANDB_ENTITY}

command=(bash scripts/launch_evaluations.sh custom
    --task-file "$TASK_FILE"
    --model "$MODEL"
    --chat-template
    --failure-policy resume
    --chunk-size "$CHUNK_SIZE"
    --logs-root "$LOGS_ROOT"
    --wandb-entity "$WANDB_ENTITY"
    --wandb-project "$WANDB_PROJECT"
    --account "$ACCOUNT")

[[ -n "$TABLE_METRICS" ]] && command+=(--table-metrics "$TABLE_METRICS")
[[ -n "$RESERVATION" ]] && command+=(--reservation "$RESERVATION")
[[ -n "$FORCE_TASKS" ]] && command+=(--force-tasks "$FORCE_TASKS")
[[ -n "$TOKENIZER" ]] && command+=(--tokenizer "$TOKENIZER")
[[ -n "$RUN_NAME" ]] && command+=(--name "$RUN_NAME")
if (( ${#FORWARDED_ARGS[@]} > 0 )); then
    command+=("${FORWARDED_ARGS[@]}")
fi

echo "WARNING: launch_evaluations_gracefuly.sh is deprecated; forwarding to launch_evaluations.sh" >&2
exec "${command[@]}"
