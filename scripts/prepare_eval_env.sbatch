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
: "${EVAL_ENV_MANIFEST:?EVAL_ENV_MANIFEST is required}"
echo "Preparing evaluation environment with $EVAL_CONTAINER_CONFIG"

EVAL_ENV_CACHE_ROOT=${EVAL_ENV_CACHE_ROOT:-${SCRATCH:-/tmp}/eval-envs}
mkdir -p "$EVAL_ENV_CACHE_ROOT" "$(dirname "$EVAL_ENV_MANIFEST")"
EVAL_CONTAINER_CACHE_PROBE=$(mktemp "$EVAL_ENV_CACHE_ROOT/.container-mount-probe.XXXXXX")
EVAL_CONTAINER_MANIFEST_PROBE=$(mktemp "$(dirname "$EVAL_ENV_MANIFEST")/.container-mount-probe.XXXXXX")
printf 'host-visible\n' > "$EVAL_CONTAINER_CACHE_PROBE"
printf 'host-visible\n' > "$EVAL_CONTAINER_MANIFEST_PROBE"
export EVAL_ENV_CACHE_ROOT EVAL_CONTAINER_CACHE_PROBE EVAL_CONTAINER_MANIFEST_PROBE
cleanup_mount_probes() {
    rm -f "$EVAL_CONTAINER_CACHE_PROBE" "$EVAL_CONTAINER_MANIFEST_PROBE"
}
trap cleanup_mount_probes EXIT

srun -n 1 --mpi=pmix -ul --environment="$EVAL_CONTAINER_CONFIG" \
    --export=ALL bash "$EVAL_REPO_ROOT/scripts/build_eval_env.sh"

grep -qx 'container-visible' "$EVAL_CONTAINER_CACHE_PROBE" \
    || { echo "Container did not update cache mount probe: $EVAL_CONTAINER_CACHE_PROBE" >&2; exit 1; }
grep -qx 'container-visible' "$EVAL_CONTAINER_MANIFEST_PROBE" \
    || { echo "Container did not update manifest mount probe: $EVAL_CONTAINER_MANIFEST_PROBE" >&2; exit 1; }
echo "Container path preflight passed for cache and manifest directories"
