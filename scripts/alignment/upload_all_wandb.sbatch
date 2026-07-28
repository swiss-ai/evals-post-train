#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --time=1:00:00
#SBATCH --output=logs/upload-results_%j.out
#SBATCH --error=logs/upload-results_%j.err

# upload_all_wandb.sbatch - Upload all model evaluation results to W&B
# This script is designed to be submitted after all evaluation jobs complete

echo "START TIME: $(date)"
echo "Evaluation jobs finished, uploading results to wandb..."

# Set default values (same as evaluate.sbatch)
WANDB_ENTITY=${WANDB_ENTITY:-apertus}
WANDB_PROJECT=${WANDB_PROJECT:-swissai-evals-v0.0.1}
LOGS_ROOT=${LOGS_ROOT:-/iopsstor/scratch/cscs/ihakimi/eval-logs/$WANDB_ENTITY/$WANDB_PROJECT}

echo "Sleeping for 1 minute to ensure all files are written..."
sleep 60

echo "Running upload script..."
echo "Entity: $WANDB_ENTITY"
echo "Project: $WANDB_PROJECT"
echo "Logs root: $LOGS_ROOT"
echo ""

# Run the Python script with container environment
srun -ul --environment=./containers/env.toml bash -c " \
    cd $PWD && python -m scripts.alignment.update_wandb_all_models \
        --entity '$WANDB_ENTITY' \
        --project '$WANDB_PROJECT' \
        --logs_root '$LOGS_ROOT'
"

echo "Upload completed!"
echo "END TIME: $(date)"
