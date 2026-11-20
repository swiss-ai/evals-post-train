#!/bin/bash
#
# job.sh - Run scripts/direct_serving/job.py (docs/direct-serving.md) from its own locked venv: the cached
# venv of requirements/direct-serving.lock, which holds the pinned model-launch it renders its serving job
# from (Python >= 3.11, which uv finds or fetches; see scripts/common/locked_venv.sh).
# Every argument goes to job.py. Run it on a Clariden login node:
#
#   scripts/direct_serving/job.sh --suite apertus2/chonk_sft --setup <setup> --eval '<command>' --submit
#
# DIRECT_SERVING_VENV  a venv to use instead (e.g. one with another model-launch); nothing is installed.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

die() { echo "$*" >&2; exit 1; }

if [[ -n "${DIRECT_SERVING_VENV:-}" ]]; then
    PY="$DIRECT_SERVING_VENV/bin/python"
    [[ -x "$PY" ]] || die "DIRECT_SERVING_VENV=$DIRECT_SERVING_VENV has no bin/python"
else
    # shellcheck source=scripts/common/locked_venv.sh
    source scripts/common/locked_venv.sh
    locked_venv requirements/direct-serving.lock 3.11 || die "Installing requirements/direct-serving.lock failed."
    PY="$LOCKED_VENV/bin/python"
fi
exec "$PY" -m scripts.direct_serving.job "$@"
