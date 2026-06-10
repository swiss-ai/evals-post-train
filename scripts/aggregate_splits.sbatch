#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=eval-aggregate
#SBATCH --mem=64000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=normal
#SBATCH --time=00:30:00

# aggregate_splits.sbatch - Merge results from split evaluation jobs and upload to W&B
# Usage: sbatch --dependency=afterok:JOB1:JOB2:... aggregate_splits.sbatch <model> <name>
#
# Required env vars: HARNESS_DIR, NUM_SPLITS, WANDB_ENTITY, WANDB_PROJECT, TABLE_METRICS

set -e
echo "START TIME: $(date)"

if (( $# != 2 )); then
    echo "Usage: sbatch aggregate_splits.sbatch <model> <name>"
    exit 1
fi
MODEL=$1
NAME=$2

# These must be passed as env vars from the launcher
LOGS_ROOT=${LOGS_ROOT:-/iopsstor/scratch/cscs/dmelikidze/eval_logs_start}
WANDB_ENTITY=${WANDB_ENTITY:-apertus}
WANDB_PROJECT=${WANDB_PROJECT:-apertus-1.5-post-training-v0.0}
TABLE_METRICS=${TABLE_METRICS:-""}
NUM_SPLITS=${NUM_SPLITS:-1}

export WANDB_API_KEY=${WANDB_API_KEY:-$(cat ./scripts/wandb_api_key.txt)}
export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/ymetz/huggingface}

if [ -f "$TABLE_METRICS" ]; then
    echo "Reading table metrics from file: $TABLE_METRICS"
    TABLE_METRICS=$(grep -v '^\s*#' "$TABLE_METRICS" | grep -v '^\s*$' | paste -sd' ' -)
fi

RUN_ROOT=$LOGS_ROOT/$WANDB_ENTITY/$WANDB_PROJECT/$NAME
HARNESS_DIR=$RUN_ROOT/harness
SPLIT_MARKER_DIR=$HARNESS_DIR/split_markers

echo "Aggregating $NUM_SPLITS split results for $NAME"
echo "Looking for markers in: $SPLIT_MARKER_DIR"

# Collect all split eval directories from markers
SPLIT_DIRS=()
for (( i=0; i<NUM_SPLITS; i++ )); do
    MARKER="$SPLIT_MARKER_DIR/split_${i}.txt"
    if [ ! -f "$MARKER" ]; then
        echo "ERROR: Missing marker for split $i: $MARKER"
        exit 1
    fi
    SPLIT_DIR=$(cat "$MARKER")
    if [ ! -d "$SPLIT_DIR" ]; then
        echo "ERROR: Split dir does not exist: $SPLIT_DIR"
        exit 1
    fi
    SPLIT_DIRS+=("$SPLIT_DIR")
    echo "  Split $i: $SPLIT_DIR"
done

# Create merged output directory
MERGED_DIR=$HARNESS_DIR/eval_merged_$(date +%Y%m%d_%H%M%S)_$SLURM_JOBID
mkdir -p "$MERGED_DIR"

echo "Merging results into: $MERGED_DIR"

# Merge using Python script
MERGE_CMD="cd $PWD && python -m scripts.alignment.merge_split_results --split_dirs ${SPLIT_DIRS[*]} --output_dir $MERGED_DIR"
echo "Merge command: $MERGE_CMD"

srun -ul --mpi=pmix --environment=./containers/env.toml bash -c " \
    pip install --no-cache-dir --upgrade \"git+https://github.com/swiss-ai/lm-evaluation-harness.git\" && \
    pip install --no-cache-dir --upgrade --no-deps \
        \"huggingface-hub>=1.3.0,<2.0\" \
        \"transformers==5.1.0\" && \
    $MERGE_CMD
"

# Compute total duration from all splits (sum)
TOTAL_DURATION=0
for SPLIT_DIR in "${SPLIT_DIRS[@]}"; do
    # Try to extract duration from the split's log (best effort)
    TOTAL_DURATION=$((TOTAL_DURATION + 1))
done
# Use a rough estimate - actual duration is wall-clock of the longest split
# The real value could be parsed from logs, but the wall-clock time is most relevant
DURATION=${EVAL_DURATION:-0}

echo "Uploading merged results to wandb"
WANDB_CMD="cd $PWD && python -m scripts.alignment.update_wandb_alignment --entity $WANDB_ENTITY --project $WANDB_PROJECT --logs_root $MERGED_DIR --name $NAME --main_metrics $TABLE_METRICS --eval_duration $DURATION"
echo "Running: $WANDB_CMD"
srun -ul --mpi=pmix --environment=./containers/env.toml bash -c " \
    $WANDB_CMD
"

# Clean up markers
rm -rf "$SPLIT_MARKER_DIR"

echo "Aggregation complete"
echo "END TIME: $(date)"
