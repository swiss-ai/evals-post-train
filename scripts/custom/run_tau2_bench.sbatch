#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=8
#SBATCH --job-name=tau2-bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=normal
#SBATCH --time=23:59:00

# run_tau2_bench.sbatch - SLURM wrapper for scripts/custom/run_tau2_bench.sh.
#
# Like run_inspect_eval.sbatch: tau2-bench evaluates an already-running served endpoint, no model is
# loaded in this job, so it asks for CPU only. No --mem: on highprio Slurm cancels such small jobs a few seconds
# after the start when they set their own (the default memory of the 8 CPUs is enough). The job runs in the prebuilt eval image when
# EVAL_PREBUILT_ENV=1 (tau2-bench baked in, nothing installed), else in containers/env.toml.
#
# Usage:
#   sbatch scripts/custom/run_tau2_bench.sbatch <run_tau2_bench.sh args...>
# e.g.
#   sbatch scripts/custom/run_tau2_bench.sbatch --domain retail --model chonk \
#     --api-base-url https://<server>/v1 --num-tasks 2
# All arguments are forwarded verbatim; `bash scripts/custom/run_tau2_bench.sh --help` lists them.
# Without --workdir the results go to logs/tau2/<job id>/ in the checkout (the container's /tmp is lost
# with the job): <workdir>/tau2-data/simulations/<run>/results.json.
#
# Keys come from the environment, else scripts/secrets.env (scripts/common/secrets.sh):
# TARGET_API_KEY (default CSCS_SERVING_API) for the endpoint, OPENAI_API_KEY for retail's NL-assertion
# judge (hardcoded to OpenAI in tau2) or a user that litellm reaches at OpenAI.

set -e

if (( $# == 0 )); then
    echo "Usage: sbatch scripts/custom/run_tau2_bench.sbatch <run_tau2_bench.sh args...>" >&2
    echo "See 'bash scripts/custom/run_tau2_bench.sh --help' for the argument list." >&2
    exit 1
fi

echo "START TIME: $(date)"
echo "Using nodes: $SLURM_JOB_NODELIST"

# shellcheck source=scripts/common/secrets.sh
source scripts/common/secrets.sh
export CSCS_SERVING_API="${CSCS_SERVING_API:-}" TARGET_API_KEY="${TARGET_API_KEY:-}"
[[ -n "${OPENAI_API_KEY:-}" ]] && export OPENAI_API_KEY

environment=./containers/env.toml
if [[ "${EVAL_PREBUILT_ENV:-0}" == "1" && -n "${EVAL_PREBUILT_CONTAINER_CONFIG:-}" ]]; then
    environment=$EVAL_PREBUILT_CONTAINER_CONFIG
fi

srun -n 1 --mpi=pmix -ul --environment="$environment" \
    --export=ALL,CSCS_SERVING_API,TARGET_API_KEY,OPENAI_API_KEY \
    bash -c '
        set -euo pipefail
        cd "$SLURM_SUBMIT_DIR"
        unset SSL_CERT_FILE
        exec bash scripts/custom/run_tau2_bench.sh --workdir "logs/tau2/$SLURM_JOB_ID" "$@"
    ' -- "$@"

echo "END TIME: $(date)"
