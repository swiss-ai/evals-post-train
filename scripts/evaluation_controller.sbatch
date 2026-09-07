#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=4
#SBATCH --job-name=eval-controller
#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=normal
#SBATCH --time=00:15:00

set -euo pipefail

bash scripts/evaluation_orchestrator.sh controller "$@"
