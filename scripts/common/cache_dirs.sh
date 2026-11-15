# shellcheck shell=bash
# cache_dirs.sh - Where every runner keeps its Hugging Face caches: on scratch, not in HOME.
# Sourced next to secrets.sh by the lm-eval, Inspect and aggregation jobs, so a dataset is downloaded once
# for all of them and a job in the prebuilt image needs no home mount. HF_HOME (hub and models) can be
# set by the caller; EVAL_CACHE_ROOT is also where run_lm_eval.sbatch keeps its other caches.

EVAL_CACHE_ROOT="${SCRATCH:-/tmp}/.cache"
export HF_HOME=${HF_HOME:-${SCRATCH:-/tmp}/huggingface}
export HF_DATASETS_CACHE="$EVAL_CACHE_ROOT/huggingface/datasets"
mkdir -p "$HF_DATASETS_CACHE"

# XDG data and cache on scratch too. Inspect writes its trace log (XDG_DATA_HOME/inspect_ai) and its
# dataset downloads (XDG_CACHE_HOME/inspect_evals) there. In the prebuilt image HOME is on the container's
# fuse-overlayfs root, and Inspect jobs hung forever in a FUSE request on it (swiss-ai/evals-post-train#158).
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$EVAL_CACHE_ROOT}
export XDG_DATA_HOME=${XDG_DATA_HOME:-${SCRATCH:-/tmp}/.local/share}
mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME"
