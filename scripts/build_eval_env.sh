#!/bin/bash

# Build immutable, shared evaluation environments. This script runs inside the
# same container used by the evaluation backend.
set -euo pipefail

: "${EVAL_ENV_MANIFEST:?EVAL_ENV_MANIFEST is required}"
: "${EVAL_HARNESS_REPO:?EVAL_HARNESS_REPO is required}"
: "${EVAL_CONTAINER_CONFIG:?EVAL_CONTAINER_CONFIG is required}"

EVAL_HARNESS_REF=${EVAL_HARNESS_REF:-HEAD}
EVAL_ENV_CACHE_ROOT=${EVAL_ENV_CACHE_ROOT:-${SCRATCH:-/tmp}/eval-envs}
EVAL_RUNTIME_REQUIREMENTS=${EVAL_RUNTIME_REQUIREMENTS:-${EVAL_REPO_ROOT:-$PWD}/requirements/eval-runtime.txt}
EVAL_BACKEND=${LM_EVAL_BACKEND:-vllm}
HARNESS_URL="https://github.com/${EVAL_HARNESS_REPO}.git"

mkdir -p "$EVAL_ENV_CACHE_ROOT/base" "$EVAL_ENV_CACHE_ROOT/harness" \
    "$EVAL_ENV_CACHE_ROOT/locks" "$(dirname "$EVAL_ENV_MANIFEST")"

if [[ "$EVAL_HARNESS_REF" =~ ^[0-9a-fA-F]{40}$ ]]; then
    HARNESS_COMMIT=${EVAL_HARNESS_REF,,}
else
    if [[ "$EVAL_HARNESS_REF" == "HEAD" ]]; then
        HARNESS_COMMIT=$(git ls-remote "$HARNESS_URL" HEAD | awk 'NR == 1 {print $1}')
    else
        HARNESS_COMMIT=$(git ls-remote "$HARNESS_URL" \
            "$EVAL_HARNESS_REF" "refs/heads/$EVAL_HARNESS_REF" \
            "refs/tags/$EVAL_HARNESS_REF^{}" | awk 'NR == 1 {print $1}')
    fi
fi
if [[ ! "$HARNESS_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "Unable to resolve $EVAL_HARNESS_REPO ref '$EVAL_HARNESS_REF'" >&2
    exit 1
fi

hash_stream() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum | awk '{print $1}'
    else
        shasum -a 256 | awk '{print $1}'
    fi
}

BASE_KEY=$(
    {
        printf '%s\n' "$EVAL_BACKEND"
        python -VV
        printf '%s\n' "--- requirements ---"
        cat "$EVAL_RUNTIME_REQUIREMENTS"
        printf '%s\n' "--- container ---"
        cat "$EVAL_CONTAINER_CONFIG"
    } | hash_stream
)
OVERLAY_KEY=$(printf '%s\n%s\n%s\n' "$BASE_KEY" "$EVAL_HARNESS_REPO" "$HARNESS_COMMIT" | hash_stream)

BASE_ENV="$EVAL_ENV_CACHE_ROOT/base/${BASE_KEY:0:20}"
HARNESS_OVERLAY="$EVAL_ENV_CACHE_ROOT/harness/${OVERLAY_KEY:0:20}"

base_is_ready() {
    [[ -f "$BASE_ENV/.complete" && -x "$BASE_ENV/bin/python" ]] || return 1
    "$BASE_ENV/bin/python" -c "import accelerate, datasets, transformers, wandb" \
        >/dev/null 2>&1
}

overlay_is_ready() {
    [[ -f "$HARNESS_OVERLAY/.complete" ]] || return 1
    PYTHONPATH="$HARNESS_OVERLAY" "$BASE_ENV/bin/python" -c "import lm_eval" \
        >/dev/null 2>&1
}

build_base() {
    exec 9>"$EVAL_ENV_CACHE_ROOT/locks/base-${BASE_KEY}.lock"
    flock 9
    if base_is_ready; then
        return
    fi

    # A venv embeds its creation path in console-script shebangs, so it must be
    # built at its final location. Preserve an interrupted or partially purged
    # build for diagnosis.
    if [[ -e "$BASE_ENV" ]]; then
        mv "$BASE_ENV" "${BASE_ENV}.incomplete.$(date +%s).$$"
    fi
    python -m venv --system-site-packages "$BASE_ENV"
    "$BASE_ENV/bin/python" -m pip install --upgrade pip
    "$BASE_ENV/bin/python" -m pip install -r "$EVAL_RUNTIME_REQUIREMENTS"
    if [[ "$EVAL_BACKEND" == "megatron_lm" ]]; then
        "$BASE_ENV/bin/python" -m pip install --no-deps megatron-core
    fi
    "$BASE_ENV/bin/python" -c "import accelerate, datasets, transformers, wandb"
    touch "$BASE_ENV/.complete"
}

build_overlay() {
    local tmp_dir
    exec 8>"$EVAL_ENV_CACHE_ROOT/locks/harness-${OVERLAY_KEY}.lock"
    flock 8
    if overlay_is_ready; then
        return
    fi

    if [[ -e "$HARNESS_OVERLAY" ]]; then
        mv "$HARNESS_OVERLAY" \
            "${HARNESS_OVERLAY}.incomplete.$(date +%s).$$"
    fi

    tmp_dir=$(mktemp -d "$EVAL_ENV_CACHE_ROOT/harness/.build-${OVERLAY_KEY:0:12}.XXXXXX")
    trap 'rm -rf "${tmp_dir:-}"' RETURN
    "$BASE_ENV/bin/python" -m pip install --no-deps --target "$tmp_dir" \
        "git+${HARNESS_URL}@${HARNESS_COMMIT}"
    PYTHONPATH="$tmp_dir" "$BASE_ENV/bin/python" -c "import lm_eval"
    touch "$tmp_dir/.complete"
    mv "$tmp_dir" "$HARNESS_OVERLAY"
    trap - RETURN
}

build_base
build_overlay

manifest_tmp="${EVAL_ENV_MANIFEST}.tmp.$$"
{
    printf 'EVAL_BASE_ENV=%q\n' "$BASE_ENV"
    printf 'EVAL_HARNESS_OVERLAY=%q\n' "$HARNESS_OVERLAY"
    printf 'EVAL_RESOLVED_HARNESS_REPO=%q\n' "$EVAL_HARNESS_REPO"
    printf 'EVAL_RESOLVED_HARNESS_COMMIT=%q\n' "$HARNESS_COMMIT"
    printf 'EVAL_RESOLVED_BASE_KEY=%q\n' "$BASE_KEY"
} > "$manifest_tmp"
mv "$manifest_tmp" "$EVAL_ENV_MANIFEST"

echo "Evaluation environment ready"
echo "  Base:    $BASE_ENV"
echo "  Harness: $EVAL_HARNESS_REPO@$HARNESS_COMMIT"
echo "  Overlay: $HARNESS_OVERLAY"
