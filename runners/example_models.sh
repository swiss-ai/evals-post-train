#!/bin/bash
# example_models.sh - Model list for launch_evaluations.sh --script: copy it, list your models.
#
#   bash scripts/launch_evaluations.sh posttrain --script runners/example_models.sh
#
# Per-run settings come from the launcher flags, e.g. base models: --no-chat-template, plus
# --tokenizer alehc/swissai-tokenizer --bos for Apertus base checkpoints. Optional per-model
# Megatron iterations: declare -A MODEL_ITERATIONS=(["<name>-iter"]=8926).

# name -> HF repo id or checkpoint path
declare -A MODEL_CHECKPOINTS=(
    ["Apertus-8B-sft-mixture-8e-aligned"]="/capstor/store/cscs/swissai/infra01/swiss-alignment/checkpoints/Apertus-8B-sft-mixture-8e-aligned"
    ["Apertus-70B-sft-mixture-8e-aligned"]="/capstor/store/cscs/swissai/infra01/swiss-alignment/checkpoints/Apertus-70B-sft-mixture-8e-aligned"
    ["Llama-3.1-8B-Instruct"]="meta-llama/Llama-3.1-8B-Instruct"
)

export WANDB_ENTITY=${WANDB_ENTITY:-apertus}
export WANDB_PROJECT=${WANDB_PROJECT:-swissai-evals}
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-true}

source runners/hf_base_runner.sh "models"
