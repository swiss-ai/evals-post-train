#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=16
#SBATCH --job-name=eval-env
#SBATCH --mem=64000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=normal
#SBATCH --time=00:30:00

set -euo pipefail

: "${EVAL_CONTAINER_CONFIG:?EVAL_CONTAINER_CONFIG is required}"
: "${EVAL_REPO_ROOT:?EVAL_REPO_ROOT is required}"
echo "Preparing evaluation environment with $EVAL_CONTAINER_CONFIG"

srun -n 1 --mpi=pmix -ul --environment="$EVAL_CONTAINER_CONFIG" \
    --export=ALL bash "$EVAL_REPO_ROOT/scripts/build_eval_env.sh"
