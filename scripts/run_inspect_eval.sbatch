#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=8
#SBATCH --job-name=inspect-eval
#SBATCH --mem=32000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=normal
#SBATCH --time=11:59:00

# run_inspect_eval.sbatch - SLURM wrapper for scripts/run_inspect_eval.sh.
#
# The benchmarks run_inspect_eval.sh targets evaluate an already-running served endpoint
# (CSCS serving, `vllm serve`, ...) or a hosted API model -- no model is loaded in this job, so
# it requests CPU-only resources, unlike evaluate.sbatch's GPU allocation for the
# lm-eval-harness backends. If a future benchmark needs Inspect to load a model in-job (its own
# hf/vllm provider), bump --cpus-per-task/--mem and add a --gres=gpu request above.
#
# Usage:
#   sbatch scripts/run_inspect_eval.sbatch <run_inspect_eval.sh args...>
# All arguments are forwarded verbatim to scripts/run_inspect_eval.sh -- see
# `bash scripts/run_inspect_eval.sh --help` for the full argument list, e.g.:
#   sbatch scripts/run_inspect_eval.sbatch --task tau2_retail --model Qwen/Qwen3-8B \
#     --api-base-url http://nid001234:8000 --model-role user=openai/gpt-4.1-2025-04-14
#
# Credentials are read the same way evaluate.sbatch reads them: from the environment, else
# scripts/<name>_key.txt. HF_TOKEN/CSCS_SERVING_API/WANDB_API_KEY fall back to their key files;
# other provider keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, TARGET_API_KEY, ...) must already be
# exported -- there's no fixed set of providers to special-case here, unlike the harness's single
# backend. WANDB_API_KEY is only actually needed if you pass --wandb-entity/--wandb-project (or
# set WANDB_ENTITY/WANDB_PROJECT) to opt into uploading results.

set -e

usage() {
    echo "Usage: sbatch scripts/run_inspect_eval.sbatch <run_inspect_eval.sh args...>"
    echo "Forwards all arguments to scripts/run_inspect_eval.sh inside a SLURM job/container."
    echo "See 'bash scripts/run_inspect_eval.sh --help' for the argument list."
}

if (( $# == 0 )); then
    usage
    echo "Invalid usage: no arguments given." >&2
    exit 1
fi

echo "START TIME: $(date)"
echo "Using nodes: $SLURM_JOB_NODELIST"

export HF_TOKEN="${HF_TOKEN:-$(tr -d '\r\n' < ./scripts/hf_token.txt)}"
export CSCS_SERVING_API="${CSCS_SERVING_API:-$(tr -d '\r\n' < ./scripts/cscs_serving_api_key.txt)}"
if [[ -z "${OPENAI_API_KEY:-}" && -f ./scripts/openai_api_key.txt ]]; then
    export OPENAI_API_KEY="$(tr -d '\r\n' < ./scripts/openai_api_key.txt)"
fi
if [[ -z "${WANDB_API_KEY:-}" && -f ./scripts/wandb_api_key.txt ]]; then
    export WANDB_API_KEY="$(tr -d '\r\n' < ./scripts/wandb_api_key.txt)"
fi

# Forward whatever the caller already exported for other providers/roles (e.g.
# ANTHROPIC_API_KEY for a --model-role grader=anthropic/... , or TARGET_API_KEY for
# --api-base-url) -- run_inspect_eval.sh applies its own CSCS_SERVING_API fallback for the
# latter, so an empty default here is fine.
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
export TARGET_API_KEY="${TARGET_API_KEY:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

srun -n 1 --mpi=pmix -ul --environment=./containers/env.toml \
    --export=ALL,HF_TOKEN,CSCS_SERVING_API,OPENAI_API_KEY,ANTHROPIC_API_KEY,TARGET_API_KEY,WANDB_API_KEY \
    bash -c '
        set -euo pipefail
        cd "$SLURM_SUBMIT_DIR"
        unset SSL_CERT_FILE
        export HF_DATASETS_TRUST_REMOTE_CODE=1
        exec bash scripts/run_inspect_eval.sh "$@"
    ' -- "$@"

echo "END TIME: $(date)"
