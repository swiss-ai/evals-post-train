#!/bin/bash
#SBATCH --job-name=helpfulness-regression
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
: "${EVAL_REPO_ROOT:?Set the staged checkout}"
: "${EVAL_REGRESSION_ROOT:?Set the private permanent run directory}"
: "${EVAL_CONTAINER_CONFIG:?Set the EDF}"
exec srun --ntasks=1 --environment="$EVAL_CONTAINER_CONFIG" --export=ALL \
    bash "$EVAL_REPO_ROOT/scripts/helpfulness/prepare_regression.sh"
