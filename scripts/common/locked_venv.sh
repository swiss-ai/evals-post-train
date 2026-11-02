# shellcheck shell=bash
# locked_venv.sh - Source this for locked_venv: the cached venv of a requirements lock.
#
#   locked_venv <lock file> <minimum Python>   sets LOCKED_VENV; non-zero on failure
#
# Standalone runs get their packages this way; the prebuilt image (containers/Dockerfile.eval) bakes
# in the same locks. Venvs live in ${EVAL_ENV_CACHE_ROOT:-$SCRATCH/eval-envs} (~/.cache/evals-post-train
# without $SCRATCH), keyed by the lock and the Python, since a venv only works with the Python that
# built it. Deleting one only costs time: the lock rebuilds the same versions.

locked_venv() {
    local lock=$1 min_python=$2 python root name key venv
    if command -v uv >/dev/null 2>&1; then
        python=$(uv python find ">=$min_python" 2>/dev/null) ||
            { uv python install --quiet "$min_python" && python=$(uv python find ">=$min_python"); } ||
            return 1
    else
        python=python3
    fi

    root=${EVAL_ENV_CACHE_ROOT:-${SCRATCH:-${XDG_CACHE_HOME:-$HOME/.cache}/evals-post-train}/eval-envs}
    name=$(basename "$lock" .lock)
    key=$({ cat "$lock" && "$python" -VV && uname -m; } | { sha256sum 2>/dev/null || shasum -a 256; } | cut -c1-20) ||
        return 1
    venv="$root/$name/$key"
    mkdir -p "$root/$name" "$root/locks" || return 1

    # Simultaneous jobs wait for the first builder. flock is missing on macOS, where runs rarely race.
    exec 9>"$root/locks/$name-$key.lock" || return 1
    if command -v flock >/dev/null 2>&1; then flock 9 || return 1; fi
    if [[ ! -f "$venv/.complete" || ! -x "$venv/bin/python" ]]; then
        echo "Installing $lock into $venv"
        # A venv embeds its path, so it is built in place. Keep an interrupted build for diagnosis.
        if [[ -e "$venv" ]]; then mv "$venv" "$venv.incomplete.$(date +%s).$$" || return 1; fi
        if command -v uv >/dev/null 2>&1; then
            uv venv --quiet --python "$python" "$venv" &&
                uv pip install --quiet --python "$venv/bin/python" -r "$lock" || return 1
        else
            "$python" -m venv "$venv" &&
                "$venv/bin/python" -m pip install --quiet --no-cache-dir -r "$lock" || return 1
        fi
    fi
    # Also refreshes a reused venv for age-based scratch retention.
    touch "$venv/.complete" || return 1
    exec 9>&-
    LOCKED_VENV=$venv
}
