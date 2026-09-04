#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=16
#SBATCH --job-name=evaluation-api
#SBATCH --mem=64000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=normal
#SBATCH --time=11:59:00

# CPU-only allocation wrapper for the OpenAI-compatible backend. The shared
# evaluation implementation is run as a regular shell script, so its GPU SBATCH
# directives are comments within this already allocated job.
set -euo pipefail

if [[ "${LM_EVAL_BACKEND:-}" != "openai" ]]; then
    echo "scripts/evaluate_api.sbatch requires LM_EVAL_BACKEND=openai" >&2
    exit 2
fi

bash scripts/evaluate.sbatch "$@"
