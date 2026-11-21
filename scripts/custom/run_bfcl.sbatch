#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=8
#SBATCH --job-name=bfcl
#SBATCH --mem=32000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=normal
#SBATCH --time=11:59:00

# run_bfcl.sbatch - SLURM wrapper for scripts/custom/run_bfcl.sh on Clariden.
#
# A CPU-only job: BFCL's multi-turn loop and checkers run here and call the model under test at
# --api-base-url (a served model, e.g. the one the suite's launch block starts). No container:
# run_bfcl.sh builds its locked venv into the cache (or uses BFCL_VENV in the prebuilt image).
#
# Usage:
#   sbatch [-p debug -t 1:00:00] scripts/custom/run_bfcl.sbatch <run_bfcl.sh args...>
# e.g. a smoke test:
#   sbatch -p debug -t 1:00:00 scripts/custom/run_bfcl.sbatch --version v3 --model m --api-base-url http://host:8000/v1 --num-tasks 2

set -e

if (( $# == 0 )); then
    echo "Usage: sbatch scripts/custom/run_bfcl.sbatch <run_bfcl.sh args...>" >&2
    exit 1
fi

echo "START TIME: $(date)"
echo "Using nodes: $SLURM_JOB_NODELIST"
export PATH="$HOME/.local/bin:$PATH"
STATUS=0
bash scripts/custom/run_bfcl.sh "$@" || STATUS=$?
echo "END TIME: $(date)"
exit $STATUS
