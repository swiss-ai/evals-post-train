#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=8
#SBATCH --job-name=gaia2-eval
#SBATCH --mem=32000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=normal
#SBATCH --time=11:59:00
#SBATCH --signal=B:USR1@180

# run_gaia2_eval.sbatch - SLURM wrapper for scripts/gaia2/run_gaia2_eval.sh on Clariden.
#
# A CPU-only job: the scenarios run as Kata microVMs on the OpenSandbox server (only
# OPENSANDBOX_API_KEY), the agent inside each calls the model at --api-base-url, and this job
# runs gaia2-runner's orchestration. So the model must be served where the sandboxes reach it
# (the SwissAI gateway), not in this or another Clariden job. No container: gaia2-runner needs
# Python >= 3.12, which run_gaia2_eval.sh installs with uv.
#
# Sandboxes are labelled sandbox-run=gaia2-<job id>. Three minutes before the time limit the job
# stops the run (summary.json keeps the scenarios that finished), and the runner deletes the
# sandboxes left with its label.
#
# Usage:
#   sbatch [-p debug -t 1:00:00] scripts/gaia2/run_gaia2_eval.sbatch <run_gaia2_eval.sh args...>
# e.g. a check of the sandboxes and the judge:
#   sbatch -p debug -t 1:00:00 scripts/gaia2/run_gaia2_eval.sbatch --gold --language deu_Latn --limit 2 --no-wandb

set -e

if (( $# == 0 )); then
    echo "Usage: sbatch scripts/gaia2/run_gaia2_eval.sbatch <run_gaia2_eval.sh args...>" >&2
    exit 1
fi

echo "START TIME: $(date)"
echo "Using nodes: $SLURM_JOB_NODELIST"

export PATH="$HOME/.local/bin:$PATH"
export OPENSANDBOX_RUN_LABEL="${OPENSANDBOX_RUN_LABEL:-gaia2-$SLURM_JOB_ID}"

# In the background so the USR1 trap (--signal above) runs while the run is still going; not
# through srun, which would turn SIGTERM into a kill of the step before summary.json is written.
bash scripts/gaia2/run_gaia2_eval.sh "$@" &
RUN_PID=$!
trap 'echo "Time limit near: stopping the run"; kill -TERM "$RUN_PID" 2>/dev/null || true' USR1
STATUS=0
wait "$RUN_PID" || STATUS=$?
while kill -0 "$RUN_PID" 2>/dev/null; do wait "$RUN_PID" || STATUS=$?; done
echo "END TIME: $(date)"
exit $STATUS
