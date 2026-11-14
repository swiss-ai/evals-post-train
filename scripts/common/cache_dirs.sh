# shellcheck shell=bash
# cache_dirs.sh - Where every runner keeps its Hugging Face caches: on scratch, not in HOME.
# Sourced next to secrets.sh by the lm-eval, Inspect and aggregation jobs, so a dataset is downloaded once
# for all of them and a job in the prebuilt image needs no home mount. HF_HOME (hub and models) can be
# set by the caller; EVAL_CACHE_ROOT is also where run_lm_eval.sbatch keeps its other caches.

EVAL_CACHE_ROOT="${SCRATCH:-/tmp}/.cache"
export HF_HOME=${HF_HOME:-${SCRATCH:-/tmp}/huggingface}
export HF_DATASETS_CACHE="$EVAL_CACHE_ROOT/huggingface/datasets"
mkdir -p "$HF_DATASETS_CACHE"
