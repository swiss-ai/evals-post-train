#!/bin/bash
#
# lock_requirements.sh - Regenerate requirements/*.lock from requirements/*.txt.
#
# The .txt files are what you edit; the .lock files pin every package and dependency, and are what
# gets installed. Run this after changing a .txt file and commit both. Locked versions only move
# when a .txt file requires it, or when asked to, e.g.:
#   scripts/common/lock_requirements.sh --upgrade-package openai
# Any arguments are passed to every `uv pip compile`. CI checks that the locks are up to date.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# lock <name> <minimum Python> [uv pip compile options...]
lock() {
    local name=$1 python=$2
    shift 2
    uv pip compile "requirements/$name.txt" --output-file "requirements/$name.lock" \
        --universal --python-version "$python" --quiet \
        --custom-compile-command scripts/common/lock_requirements.sh "$@"
}

# lm-eval-runtime.txt also feeds in-job backends, which take torch from their own images and so install
# the .txt (build_eval_env.sh). This lock is for the openai backend: no model is loaded, so CPU torch.
lock lm-eval-runtime 3.11 --generate-hashes --torch-backend cpu "$@"
lock inspect-runtime 3.11 --generate-hashes "$@"
lock evalscope-runtime 3.11 --generate-hashes "$@"
lock harbor-runtime 3.12 --generate-hashes "$@"
# No hashes: the adapter is a git install, which cannot be hashed. The lock pins its commit instead.
lock harbor-adapters 3.12 "$@"
