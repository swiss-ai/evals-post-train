#!/bin/bash

# hf_base_runner.sh - Generic script to run evaluation jobs for multiple models
# Usage: hf_base_runner.sh <model_type_description>
#
# This script expects MODEL_CHECKPOINTS associative array to be defined before calling
# and optionally WANDB_ENTITY, WANDB_PROJECT, APPLY_CHAT_TEMPLATE, NUM_SPLITS environment variables
#
# When NUM_SPLITS > 1, each model's tasks are split across NUM_SPLITS parallel sbatch jobs.
# A dependency-chained aggregation job merges results and uploads to W&B.

# Get model type description from argument (for display purposes)
MODEL_TYPE_DESC=${1:-"models"}

# Set default values for optional environment variables
export WANDB_ENTITY=${WANDB_ENTITY:-apertus}
export WANDB_PROJECT=${WANDB_PROJECT:-swissai-evals}
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-false}
NUM_SPLITS=${NUM_SPLITS:-1}

# Allow overriding the sbatch scripts.
SBATCH_SCRIPT=${SBATCH_SCRIPT:-scripts/evaluate.sbatch}
STANDALONE_SBATCH_SCRIPT=${STANDALONE_SBATCH_SCRIPT:-scripts/evaluate_standalone.sbatch}
RUN_LM_EVAL=${RUN_LM_EVAL:-true}
RUN_STANDALONE=${RUN_STANDALONE:-false}

# Launch evaluation jobs for each model
echo "Launching evaluation jobs for ${#MODEL_CHECKPOINTS[@]} ${MODEL_TYPE_DESC}..."
echo "WANDB Project: ${WANDB_PROJECT}"
echo "Apply Chat Template: ${APPLY_CHAT_TEMPLATE}"
if [[ "$RUN_LM_EVAL" == "true" ]]; then
    echo "lm-eval sbatch script: ${SBATCH_SCRIPT}"
fi
if [[ "$RUN_STANDALONE" == "true" ]]; then
    echo "Standalone sbatch script: ${STANDALONE_SBATCH_SCRIPT}"
    echo "Standalone tasks: ${STANDALONE_TASKS:-}"
fi
if [[ "$RUN_LM_EVAL" == "true" && "$NUM_SPLITS" -gt 1 ]]; then
    echo "Task splits: ${NUM_SPLITS} parallel nodes per model"
fi
echo ""

job_count=0
HAS_MODEL_ITERATIONS=0
if declare -p MODEL_ITERATIONS >/dev/null 2>&1; then
    HAS_MODEL_ITERATIONS=1
fi

for MODEL in "${!MODEL_CHECKPOINTS[@]}"; do
    CKPT_PATH="${MODEL_CHECKPOINTS[$MODEL]}"
    # Priority: model-specific override > global override > latest
    CKPT_ITER="${CKPT_ITERATION:-latest}"
    if (( HAS_MODEL_ITERATIONS )) && [[ -n "${MODEL_ITERATIONS["${MODEL}-iter"]+x}" ]]; then
        CKPT_ITER="${MODEL_ITERATIONS["${MODEL}-iter"]}"
    fi
    job_count=$((job_count + 1))

    echo "Launching job $job_count/${#MODEL_CHECKPOINTS[@]}: $MODEL"
    echo "  Checkpoint path: $CKPT_PATH"
    echo "  Checkpoint iter: $CKPT_ITER (only applies to local Megatron checkpoints)"

    if [[ "$RUN_LM_EVAL" == "true" && "$NUM_SPLITS" -le 1 ]]; then
        # Single-node execution (original behavior)
        sbatch --job-name eval-$MODEL \
            --export=ALL,CKPT_ITER=$CKPT_ITER \
            "$SBATCH_SCRIPT" "$CKPT_PATH" "$MODEL"
    elif [[ "$RUN_LM_EVAL" == "true" ]]; then
        # Submit K split jobs, then one aggregation job with dependency
        SPLIT_JOB_IDS=()
        for (( i=0; i<NUM_SPLITS; i++ )); do
            JOB_ID=$(sbatch --parsable \
                --job-name "eval-${MODEL}-split${i}" \
                --export=ALL,NUM_SPLITS=$NUM_SPLITS,SPLIT_INDEX=$i,CKPT_ITER=$CKPT_ITER \
                "$SBATCH_SCRIPT" "$CKPT_PATH" "$MODEL")
            SPLIT_JOB_IDS+=("$JOB_ID")
            echo "  Split $((i+1))/$NUM_SPLITS submitted: job $JOB_ID"
            sleep 1
        done

        # Build dependency string: afterok:job1:job2:...
        DEP_STRING=$(IFS=':'; echo "${SPLIT_JOB_IDS[*]}")

        # Submit aggregation job to merge results and upload to W&B
        AGG_JOB_ID=$(sbatch --parsable \
            --job-name "eval-${MODEL}-aggregate" \
            --dependency="afterok:${DEP_STRING}" \
            --export=ALL,NUM_SPLITS=$NUM_SPLITS \
            scripts/aggregate_splits.sbatch "$CKPT_PATH" "$MODEL")
        echo "  Aggregation job submitted: job $AGG_JOB_ID (depends on splits)"
    fi

    if [[ "$RUN_STANDALONE" == "true" ]]; then
        sbatch --job-name standalone-eval-$MODEL \
            --export=ALL,CKPT_ITER=$CKPT_ITER \
            "$STANDALONE_SBATCH_SCRIPT" "$CKPT_PATH" "$MODEL"
    fi

    # Add a small delay between submissions to avoid overwhelming the scheduler
    sleep 1
done
