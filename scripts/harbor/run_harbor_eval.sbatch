#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=8
#SBATCH --job-name=harbor-eval
#SBATCH --mem=32000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=normal
#SBATCH --time=11:59:00

# run_harbor_eval.sbatch - SLURM wrapper for scripts/harbor/run_harbor_eval.sh on Clariden.
#
# Clariden's nodes are aarch64 and have no Docker, while the task images are linux/amd64: the task
# containers run as pods on a Kubernetes cluster (--sandbox k8s, the default here), e.g. Breithorn's
# rob-poc namespace, and this CPU-only job runs Harbor's orchestration and the agent loop, calling
# the model under test at --api-base-url. No container: Harbor needs Python >= 3.12, which
# run_harbor_eval.sh installs with uv.
#
# Kubernetes access is your kubeconfig on Clariden (~/.kube/config). For Breithorn that is an OIDC
# login through kubelogin, which a batch job cannot renew: log in first (kubectl get pods works on
# the login node) and keep the run within the token's lifetime.
#
# Usage:
#   sbatch [-p debug -t 1:00:00] scripts/harbor/run_harbor_eval.sbatch <run_harbor_eval.sh args...>
# e.g. a smoke test:
#   sbatch -p debug -t 1:00:00 scripts/harbor/run_harbor_eval.sbatch --benchmark swe_bench,terminal_bench_2 --gold --limit 1
# Defaults here: SANDBOX_K8S_NAMESPACE=rob-poc, memory request 4g, CPU request 1 (see below);
# other SANDBOX_K8S_* settings pass through.

set -e

if (( $# == 0 )); then
    echo "Usage: sbatch scripts/harbor/run_harbor_eval.sbatch <run_harbor_eval.sh args...>" >&2
    exit 1
fi

echo "START TIME: $(date)"
echo "Using nodes: $SLURM_JOB_NODELIST"

export PATH="$HOME/.local/bin:$PATH"
export SANDBOX_K8S_NAMESPACE="${SANDBOX_K8S_NAMESPACE:-rob-poc}"
export SANDBOX_K8S_RUN_LABEL="${SANDBOX_K8S_RUN_LABEL:-harbor-$SLURM_JOB_ID}"
# Breithorn's amd64 nodes are mostly full on reservations, not on use: reserve less than the
# tasks' limits so their pods schedule (they can still use up to the limit).
export SANDBOX_K8S_MEMORY_REQUEST="${SANDBOX_K8S_MEMORY_REQUEST:-4g}"
export SANDBOX_K8S_CPU_REQUEST="${SANDBOX_K8S_CPU_REQUEST:-1}"
command -v kubectl >/dev/null && kubectl -n "$SANDBOX_K8S_NAMESPACE" auth can-i create pods >/dev/null ||
    echo "warning: cannot confirm pod access in $SANDBOX_K8S_NAMESPACE (expired kubelogin token?)" >&2

ARGS=("$@")
[[ " ${ARGS[*]} " == *" --sandbox "* ]] || ARGS+=(--sandbox k8s)

STATUS=0
srun -n 1 bash scripts/harbor/run_harbor_eval.sh "${ARGS[@]}" || STATUS=$?
echo "Leftover pods, if the run died: kubectl -n $SANDBOX_K8S_NAMESPACE delete pods -l sandbox-run=$SANDBOX_K8S_RUN_LABEL"
echo "END TIME: $(date)"
exit $STATUS
