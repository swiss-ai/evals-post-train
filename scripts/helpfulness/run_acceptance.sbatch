#!/bin/bash
#SBATCH --job-name=helpfulness-acceptance
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
: "${EVAL_REPO_ROOT:?Set the absolute cluster evals-post-train checkout path}"
: "${EVAL_CONTAINER_CONFIG:?Set the approved EDF file or environment name}"
[[ -d "$EVAL_REPO_ROOT" ]] || { echo "Missing checkout: $EVAL_REPO_ROOT" >&2; exit 2; }

# The checkout and all evidence/output/runtime paths must be identity-mounted.
exec srun --ntasks=1 --environment="$EVAL_CONTAINER_CONFIG" --export=ALL \
    bash "$EVAL_REPO_ROOT/scripts/helpfulness/run_acceptance.sh" "$@"
