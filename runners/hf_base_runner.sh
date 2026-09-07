#!/bin/bash

# hf_base_runner.sh - Generic script to run evaluation jobs for multiple models
# Usage: hf_base_runner.sh <model_type_description>
#
# This script expects MODEL_CHECKPOINTS associative array to be defined before calling
# and optionally WANDB_ENTITY, WANDB_PROJECT, and APPLY_CHAT_TEMPLATE environment variables.
# Task chunking, retries, environment preparation, and aggregation are delegated
# to scripts/evaluation_orchestrator.sh.

# Get model type description from argument (for display purposes)
MODEL_TYPE_DESC=${1:-"models"}

# Set default values for optional environment variables
export WANDB_ENTITY=${WANDB_ENTITY:-apertus}
export WANDB_PROJECT=${WANDB_PROJECT:-swissai-evals}
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-false}
export LOGS_ROOT=${LOGS_ROOT:-${SCRATCH:-/tmp}/eval_logs_start}
export EVAL_CHUNK_SIZE=${EVAL_CHUNK_SIZE:-8}
export EVAL_MAX_PARALLEL=${EVAL_MAX_PARALLEL:-}
export EVAL_MAX_RETRIES=${EVAL_MAX_RETRIES:-1}
export EVAL_FAILURE_POLICY=${EVAL_FAILURE_POLICY:-resume}
export EVAL_FORCE_TASKS=${EVAL_FORCE_TASKS:-}
export EVAL_MERGE_ONLY=${EVAL_MERGE_ONLY:-false}
export EVAL_DRY_RUN=${EVAL_DRY_RUN:-false}

# Allow overriding the sbatch script (e.g. evaluate.sbatch)
SBATCH_SCRIPT=${SBATCH_SCRIPT:-scripts/evaluate.sbatch}
export SBATCH_SCRIPT

source scripts/evaluation_orchestrator.sh

# Launch evaluation jobs for each model
echo "Launching evaluation jobs for ${#MODEL_CHECKPOINTS[@]} ${MODEL_TYPE_DESC}..."
echo "WANDB Project: ${WANDB_PROJECT}"
echo "Apply Chat Template: ${APPLY_CHAT_TEMPLATE}"
echo "Sbatch script: ${SBATCH_SCRIPT}"
echo "Failure policy: ${EVAL_FAILURE_POLICY:-resume}"
if [[ "${EVAL_FAILURE_POLICY:-resume}" == "resume" ]]; then
    echo "Task chunk size: ${EVAL_CHUNK_SIZE:-8}"
    echo "Maximum parallel chunks: ${EVAL_MAX_PARALLEL:-all}"
fi
echo ""

EVAL_JOB_IDS=()
job_count=0
HAS_MODEL_ITERATIONS=0
if declare -p MODEL_ITERATIONS >/dev/null 2>&1; then
    HAS_MODEL_ITERATIONS=1
fi

for MODEL in "${!MODEL_CHECKPOINTS[@]}"; do
    CKPT_PATH="${MODEL_CHECKPOINTS[$MODEL]}"
    # Append optional suffix for variant runs (e.g., "-weighted", "-style-control")
    MODEL="${MODEL}${EVAL_NAME_SUFFIX:-}"
    # Priority: model-specific override > global override > latest
    CKPT_ITER="${CKPT_ITERATION:-latest}"
    if (( HAS_MODEL_ITERATIONS )) && [[ -n "${MODEL_ITERATIONS["${MODEL}-iter"]+x}" ]]; then
        CKPT_ITER="${MODEL_ITERATIONS["${MODEL}-iter"]}"
    fi
    job_count=$((job_count + 1))

    echo "Launching job $job_count/${#MODEL_CHECKPOINTS[@]}: $MODEL"
    echo "  Checkpoint path: $CKPT_PATH"
    echo "  Checkpoint iter: $CKPT_ITER (only applies to local Megatron checkpoints)"

    export CKPT_ITER
    submit_evaluation "$CKPT_PATH" "$MODEL"
    EVAL_JOB_IDS+=("$ORCHESTRATION_FINAL_JOB_ID")

    # Add a small delay between submissions to avoid overwhelming the scheduler
    sleep 1
done
