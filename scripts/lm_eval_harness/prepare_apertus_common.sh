#!/bin/bash
#
# prepare_apertus_common.sh [requirements file] - Build the apertus-common wheel that
# requirements/apertus2.txt (or the file given, e.g. requirements/apertus2-vllm-plugin.txt) pins.
#
# apertus-common is a private repository, and the prebuilt image has neither a credential nor an ssh
# client. So the launcher (which runs where your git access is) clones the pinned commit and builds a
# wheel once; the job only installs that wheel, offline (run_lm_eval.sbatch). The wheel is kept in
# ${EVAL_ENV_CACHE_ROOT:-$SCRATCH/eval-envs}/apertus-common/<commit>/, which the job's container must
# mount (like the rest of EVAL_ENV_CACHE_ROOT). Prints that directory; EVAL_DRY_RUN=true only prints it.
#
# Cloning uses the URL in that file through your own git configuration, e.g. an ssh rewrite:
#   git config --global url."git@github.com:".insteadOf https://github.com/

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

req=${1:-requirements/apertus2.txt}
pin=$(grep -E '^[[:space:]]*apertus-common[[:space:]]*@[[:space:]]*git\+' "$req") \
    || { echo "$req has no 'apertus-common @ git+<url>@<commit>' line" >&2; exit 1; }
url=${pin#*git+}
url=${url%@*}
commit=${pin##*@}
commit=${commit//[[:space:]]/}
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] \
    || { echo "$req does not pin apertus-common to a full commit sha" >&2; exit 1; }

cache_root=${EVAL_ENV_CACHE_ROOT:-${SCRATCH:-/tmp}/eval-envs}
wheel_dir="$cache_root/apertus-common/$commit"

if [[ "${EVAL_DRY_RUN:-false}" != "true" && ! -f "$wheel_dir/.complete" ]]; then
    mkdir -p "$cache_root/apertus-common"
    work=$(mktemp -d "$cache_root/apertus-common/.build-${commit:0:12}.XXXXXX")
    trap 'rm -rf -- "$work"' EXIT
    echo "Building apertus-common@${commit:0:12} from $url" >&2
    git init -q "$work/src"
    GIT_TERMINAL_PROMPT=0 git -C "$work/src" fetch -q --depth 1 "$url" "$commit" >&2 \
        || { echo "Could not fetch $url@$commit: this needs read access to the repository (see the header of this script)." >&2; exit 1; }
    git -C "$work/src" checkout -q FETCH_HEAD
    [[ "$(git -C "$work/src" rev-parse HEAD)" == "$commit" ]] \
        || { echo "Fetched a different commit than the pinned $commit" >&2; exit 1; }
    uv build --wheel --quiet --out-dir "$work/wheel" "$work/src" >&2
    touch "$work/wheel/.complete"
    # Another launch may have built it meanwhile: keep that one.
    [[ -f "$wheel_dir/.complete" ]] || mv "$work/wheel" "$wheel_dir"
fi

printf '%s\n' "$wheel_dir"
