#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --job-name=convert-megatron-hf
#SBATCH --mem=230000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=normal
#SBATCH --time=02:00:00

# Converts a Megatron distributed checkpoint to an Apertus Hugging Face checkpoint
# using swiss-ai/Megatron-LM's tools/checkpoint/convert.py (loader=core, saver=swissai_hf).
# That saver understands Apertus's xIELU activation, QK-norm, and GQA layout, and is
# the tool that produced the released swiss-ai/Apertus-* Hugging Face checkpoints.
#
# Usage: sbatch convert_megatron_checkpoint.sbatch <megatron_load_dir> <hf_save_dir>
#
# Env vars (all optional):
#   HF_TOKENIZER              Tokenizer id/path baked into the converted checkpoint
#                             (default: alehc/swissai-tokenizer)
#   FORCE_CONVERT             'true' to reconvert even if hf_save_dir already has a
#                             config.json (default: false)
#   SWISSAI_MEGATRON_LM_PATH  Path to the swiss-ai/Megatron-LM checkout
#                             (default: $PWD/swiss-ai-Megatron-LM)
#   CHECK_EQ_HF               Path/repo-id of a reference HF checkpoint to diff the
#                             conversion output against (loader's --check-eq-hf);
#                             useful to validate against a known-good conversion.

set -euo pipefail
mkdir -p logs

usage() {
    echo "Usage: sbatch convert_megatron_checkpoint.sbatch <megatron_load_dir> <hf_save_dir>"
}

MEGATRON_LOAD_DIR="${1:?$(usage)}"
HF_SAVE_DIR="${2:?$(usage)}"
HF_TOKENIZER="${HF_TOKENIZER:-alehc/swissai-tokenizer}"
FORCE_CONVERT="${FORCE_CONVERT:-false}"
SWISSAI_MEGATRON_LM_PATH="${SWISSAI_MEGATRON_LM_PATH:-$PWD/swiss-ai-Megatron-LM}"
CHECK_EQ_HF="${CHECK_EQ_HF:-}"

[[ -d "$MEGATRON_LOAD_DIR" ]] \
    || { echo "Megatron checkpoint directory not found: $MEGATRON_LOAD_DIR" >&2; exit 1; }
[[ -d "$SWISSAI_MEGATRON_LM_PATH" ]] \
    || { echo "swiss-ai/Megatron-LM checkout not found at $SWISSAI_MEGATRON_LM_PATH (git clone https://github.com/swiss-ai/Megatron-LM.git)" >&2; exit 1; }

if [[ "$FORCE_CONVERT" != "true" && -f "$HF_SAVE_DIR/config.json" ]]; then
    echo "Converted checkpoint already exists at $HF_SAVE_DIR (config.json found); skipping conversion."
    echo "Set FORCE_CONVERT=true to reconvert."
    exit 0
fi
mkdir -p "$HF_SAVE_DIR"

# Megatron's loader resolves the checkpoint iteration from a
# latest_checkpointed_iteration.txt tracker file in --load-dir (this loader plugin
# exposes no --ckpt-step passthrough), and exits treating the checkpoint as missing
# without one. Our source trees are laid out as sibling iter_NNNNNNN snapshots with
# no such tracker, so stage a tiny directory with the tracker file plus a symlink to
# the real checkpoint instead of writing into the (shared, not ours) source tree.
ITER_DIRNAME=$(basename "$MEGATRON_LOAD_DIR")
[[ "$ITER_DIRNAME" =~ ^iter_([0-9]+)$ ]] \
    || { echo "Expected $MEGATRON_LOAD_DIR to be named iter_<digits> (Megatron's checkpoint tracker needs the plain iteration number)" >&2; exit 1; }
MEGATRON_ITERATION=$((10#${BASH_REMATCH[1]}))

STAGING_DIR="${SCRATCH:-/tmp}/megatron-convert-staging/$(basename "$HF_SAVE_DIR")"
mkdir -p "$STAGING_DIR"
echo "$MEGATRON_ITERATION" > "$STAGING_DIR/latest_checkpointed_iteration.txt"
ln -sfn "$MEGATRON_LOAD_DIR" "$STAGING_DIR/$ITER_DIRNAME"

# Reuse the same cached-venv builder the megatron_lm eval backend uses (it installs
# megatron-core into a venv keyed by container + requirements hash) so this doesn't
# need its own environment-caching logic.
EVAL_ENV_CACHE_ROOT="${EVAL_ENV_CACHE_ROOT:-${SCRATCH:-/tmp}/eval-envs}"
EVAL_ENV_MANIFEST="$EVAL_ENV_CACHE_ROOT/convert-environment.${SLURM_JOB_ID:-$$}.sh"
mkdir -p "$(dirname "$EVAL_ENV_MANIFEST")"

srun -n 1 --mpi=pmix -ul --environment=./containers/env_vllm.toml --export=ALL bash -lc "
    set -euo pipefail
    # Megatron requires this whenever tensor/context parallelism > 1, which this
    # checkpoint was trained with.
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export EVAL_ENV_CACHE_ROOT='$EVAL_ENV_CACHE_ROOT'
    export EVAL_ENV_MANIFEST='$EVAL_ENV_MANIFEST'
    export EVAL_HARNESS_REPO=\${EVAL_HARNESS_REPO:-swiss-ai/lm-evaluation-harness}
    export EVAL_CONTAINER_CONFIG='$PWD/containers/env_vllm.toml'
    export EVAL_REPO_ROOT='$PWD'
    export EVAL_RUNTIME_REQUIREMENTS='$PWD/requirements/eval-runtime.txt'
    export LM_EVAL_BACKEND=megatron_lm
    bash '$PWD/scripts/lm_eval_harness/build_eval_env.sh'
    source '$EVAL_ENV_MANIFEST'
    source \"\$EVAL_BASE_ENV/bin/activate\"

    export PYTHONPATH='$SWISSAI_MEGATRON_LM_PATH'\${PYTHONPATH:+:\$PYTHONPATH}
    cd '$SWISSAI_MEGATRON_LM_PATH/tools/checkpoint'
    echo Starting checkpoint conversion: \$(date -Is)
    python convert.py \
        --model-type GPT \
        --loader core \
        --saver swissai_hf \
        --load-dir '$STAGING_DIR' \
        --save-dir '$HF_SAVE_DIR' \
        --megatron-path '$SWISSAI_MEGATRON_LM_PATH' \
        --hf-tokenizer '$HF_TOKENIZER' \
        --use-plain-tensor-loading \
        ${CHECK_EQ_HF:+--check-eq-hf '$CHECK_EQ_HF'}
    echo Checkpoint conversion finished: \$(date -Is)
"

[[ -f "$HF_SAVE_DIR/config.json" ]] \
    || { echo "Conversion finished but $HF_SAVE_DIR/config.json is missing" >&2; exit 1; }
echo "Converted checkpoint written to $HF_SAVE_DIR"
