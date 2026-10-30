#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=8
#SBATCH --job-name=evalscope-eval
#SBATCH --mem=32000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=normal
#SBATCH --time=11:59:00
#SBATCH --signal=B:USR1@180

# run_evalscope_eval.sbatch - SLURM wrapper for scripts/evalscope/run_evalscope_eval.sh on Clariden.
#
# Clariden's nodes are aarch64 and have no Docker, while the SWE-bench images are linux/amd64: the
# sandboxes run as pods on a Kubernetes cluster (--sandbox k8s, the default here), e.g. Breithorn's
# rob-poc namespace, and this CPU-only job runs EvalScope and the agent loop, calling the model
# under test at --api-base-url. No container: run_evalscope_eval.sh installs EvalScope's venv.
#
# Kubernetes access is your kubeconfig on Clariden (~/.kube/config). For Breithorn that is an OIDC
# login through kubelogin, which a batch job cannot renew: log in first (kubectl get pods works on
# the login node) and keep the run within the token's lifetime.
#
# Pods are labelled sandbox-run=evalscope-<job id>. Three minutes before the time limit the job
# stops the run (EvalScope writes summary.json only at the end, so a stopped run has none), and at
# the end it deletes whatever pods with that label are left. With a SANDBOX_K8S_RUN_LABEL set by
# hand it only prints the command, since other runs may share that label.
#
# Usage:
#   sbatch [-p debug -t 1:00:00] scripts/evalscope/run_evalscope_eval.sbatch <run_evalscope_eval.sh args...>
# e.g. a smoke test:
#   sbatch -p debug -t 1:00:00 scripts/evalscope/run_evalscope_eval.sbatch --benchmark swe_bench,swe_bench_pro --gold --limit 2
# Defaults here: SANDBOX_K8S_NAMESPACE=rob-poc, CPU request 1 (see below); other SANDBOX_K8S_*
# settings pass through.

set -e

if (( $# == 0 )); then
    echo "Usage: sbatch scripts/evalscope/run_evalscope_eval.sbatch <run_evalscope_eval.sh args...>" >&2
    exit 1
fi

echo "START TIME: $(date)"
echo "Using nodes: $SLURM_JOB_NODELIST"

export PATH="$HOME/.local/bin:$PATH"
export SANDBOX_K8S_NAMESPACE="${SANDBOX_K8S_NAMESPACE:-rob-poc}"
export SANDBOX_K8S_RUN_LABEL="${SANDBOX_K8S_RUN_LABEL:-evalscope-$SLURM_JOB_ID}"
# Breithorn's amd64 nodes are mostly full on reservations, not on use: reserve less CPU than the
# sandboxes' limit so their pods schedule (they can still use up to the limit). The memory
# request already sits below the limit (scripts/evalscope/run.py).
export SANDBOX_K8S_CPU_REQUEST="${SANDBOX_K8S_CPU_REQUEST:-1}"
command -v kubectl >/dev/null && kubectl -n "$SANDBOX_K8S_NAMESPACE" auth can-i create pods >/dev/null ||
    echo "warning: cannot confirm pod access in $SANDBOX_K8S_NAMESPACE (expired kubelogin token?)" >&2

ARGS=("$@")
[[ " ${ARGS[*]} " == *" --sandbox "* ]] || ARGS+=(--sandbox k8s)

# In the background so the USR1 trap (--signal above) runs while the run is still going. Not
# through srun: srun takes SIGTERM as "abort the step" and Slurm then SIGKILLs it after KillWait
# (30 s on Clariden), before the sandboxes have been stopped.
bash scripts/evalscope/run_evalscope_eval.sh "${ARGS[@]}" &
RUN_PID=$!
# "|| true": the signal can come after the run ended, and set -e must not skip the cleanup below.
trap 'echo "Time limit near: stopping the run"; kill -TERM "$RUN_PID" 2>/dev/null || true' USR1
STATUS=0
wait "$RUN_PID" || STATUS=$?
# wait returns early (status > 128) when the trap fires; wait for the run to finish stopping.
while kill -0 "$RUN_PID" 2>/dev/null; do wait "$RUN_PID" || STATUS=$?; done

CLEANUP="kubectl -n $SANDBOX_K8S_NAMESPACE delete pods -l sandbox-run=$SANDBOX_K8S_RUN_LABEL"
if [[ " ${ARGS[*]} " == *" --sandbox docker "* ]] || ! command -v kubectl >/dev/null; then
    :
elif [[ "$SANDBOX_K8S_RUN_LABEL" != "evalscope-$SLURM_JOB_ID" ]]; then
    # A label set by hand may be shared with other runs: their pods must not go with this job.
    echo "Not cleaning up (custom SANDBOX_K8S_RUN_LABEL); leftover pods, if any: $CLEANUP" >&2
else
    # Bounded: with an expired kubelogin token, kubectl waits for a browser login.
    timeout 120 kubectl -n "$SANDBOX_K8S_NAMESPACE" delete pods \
        -l "sandbox-run=$SANDBOX_K8S_RUN_LABEL" --ignore-not-found --wait=false ||
        echo "warning: could not clean up; run: $CLEANUP" >&2
fi
echo "END TIME: $(date)"
exit $STATUS
