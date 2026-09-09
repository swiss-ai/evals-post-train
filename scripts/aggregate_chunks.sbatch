#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=16
#SBATCH --job-name=eval-aggregate
#SBATCH --mem=64000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=normal
#SBATCH --time=00:30:00

# aggregate_chunks.sbatch - Merge completed chunk outputs and upload to W&B
# Usage: EVAL_DIRS_FILE=<file> sbatch aggregate_chunks.sbatch <model> <name>

set -e
echo "START TIME: $(date)"

if (( $# != 2 )); then
    echo "Usage: sbatch aggregate_chunks.sbatch <model> <name>"
    exit 1
fi
MODEL=$1
NAME=$2

# These must be passed as env vars from the launcher
LOGS_ROOT=${LOGS_ROOT:-/iopsstor/scratch/cscs/dmelikidze/eval_logs_start}
WANDB_ENTITY=${WANDB_ENTITY:-apertus}
WANDB_PROJECT=${WANDB_PROJECT:-apertus-1.5-post-training-v0.0}
TABLE_METRICS=${TABLE_METRICS:-""}
: "${EVAL_DIRS_FILE:?EVAL_DIRS_FILE is required}"
EVAL_INCOMPLETE_TASKS_FILE=${EVAL_INCOMPLETE_TASKS_FILE:-""}

export WANDB_API_KEY=${WANDB_API_KEY:-$(cat ./scripts/wandb_api_key.txt)}
export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/ymetz/huggingface}

if [ -f "$TABLE_METRICS" ]; then
    echo "Reading table metrics from file: $TABLE_METRICS"
    TABLE_METRICS=$(grep -v '^\s*#' "$TABLE_METRICS" | grep -v '^\s*$' | paste -sd' ' -)
fi

RUN_ROOT=$LOGS_ROOT/$WANDB_ENTITY/$WANDB_PROJECT/$NAME
HARNESS_DIR=$RUN_ROOT/harness
mapfile -t SPLIT_DIRS < <(sed '/^[[:space:]]*$/d' "$EVAL_DIRS_FILE")
(( ${#SPLIT_DIRS[@]} > 0 )) || { echo "ERROR: no evaluation directories" >&2; exit 1; }
echo "Aggregating ${#SPLIT_DIRS[@]} completed chunk outputs for $NAME"
for split_dir in "${SPLIT_DIRS[@]}"; do
    [[ -d "$split_dir" ]] || { echo "ERROR: result directory does not exist: $split_dir" >&2; exit 1; }
    echo "  $split_dir"
done
if [[ -n "$EVAL_INCOMPLETE_TASKS_FILE" && -s "$EVAL_INCOMPLETE_TASKS_FILE" ]]; then
    echo "WARNING: uploading partial results; failed tasks:"
    sed 's/^/  - /' "$EVAL_INCOMPLETE_TASKS_FILE"
fi

# Create merged output directory
MERGED_DIR=$HARNESS_DIR/eval_merged_$(date +%Y%m%d_%H%M%S)_$SLURM_JOBID
mkdir -p "$MERGED_DIR"
if [[ -f "${EVAL_RUN_CONFIG:-}" ]]; then
    cp "$EVAL_RUN_CONFIG" "$MERGED_DIR/.eval-run-config.json"
fi

echo "Merging results into: $MERGED_DIR"

# Merge using Python script
MERGE_CMD="cd $PWD && python -m scripts.alignment.merge_split_results --split_dirs_file '$EVAL_DIRS_FILE' --output_dir '$MERGED_DIR'"
if [[ -n "$EVAL_INCOMPLETE_TASKS_FILE" && -s "$EVAL_INCOMPLETE_TASKS_FILE" ]]; then
    MERGE_CMD="$MERGE_CMD --exclude_tasks_file '$EVAL_INCOMPLETE_TASKS_FILE'"
fi
echo "Merge command: $MERGE_CMD"

srun -ul --mpi=pmix --environment=./containers/env.toml bash -c "$MERGE_CMD"

# Sum evaluation-only runtimes from completed chunks. An explicit value remains
# available for recovery of legacy result directories that predate duration files.
TOTAL_DURATION=0
MISSING_DURATIONS=0
for SPLIT_DIR in "${SPLIT_DIRS[@]}"; do
    DURATION_FILE="$SPLIT_DIR/eval_duration_seconds.txt"
    if [[ -f "$DURATION_FILE" ]]; then
        CHUNK_DURATION=$(tr -d '[:space:]' < "$DURATION_FILE")
        if [[ "$CHUNK_DURATION" =~ ^[0-9]+$ ]]; then
            TOTAL_DURATION=$((TOTAL_DURATION + CHUNK_DURATION))
            continue
        fi
    fi
    MISSING_DURATIONS=$((MISSING_DURATIONS + 1))
done
DURATION=${EVAL_DURATION:-$TOTAL_DURATION}
if (( MISSING_DURATIONS > 0 )); then
    echo "WARNING: $MISSING_DURATIONS result directories had no valid duration; aggregated duration is partial ($DURATION seconds)" >&2
fi

echo "Uploading merged results to wandb"
WANDB_CMD="cd $PWD && python -m scripts.alignment.update_wandb_alignment --entity $WANDB_ENTITY --project $WANDB_PROJECT --logs_root $MERGED_DIR --name $NAME --main_metrics $TABLE_METRICS --eval_duration $DURATION"
echo "Running: $WANDB_CMD"
srun -ul --mpi=pmix --environment=./containers/env.toml bash -c " \
    $WANDB_CMD
"

echo "Aggregation complete"
echo "END TIME: $(date)"
