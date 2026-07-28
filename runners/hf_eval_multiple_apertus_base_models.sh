#!/bin/bash

export TOKENIZER=alehc/swissai-tokenizer
export BOS=true

# Define MODEL:CKPT_PATH pairs using an associative array
declare -A MODEL_CHECKPOINTS=(    
   
    # # from slack canvas
    # #["Apertus8B-tokens10.2T-it2059810-newcooldown"]="/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/Apertus8B-tokens10.2T-it2059810-newcooldown"

    # ["Apertus8B-tokens15T-longcontext64k"]="/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/Apertus8B-tokens15T-longcontext64k"
    ["Apertus8B-tokens15T-it2627139"]="/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/Apertus8B-tokens15T-it2627139"
    
    # ["Apertus70B-tokens15T-it1155828"]=["/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/Apertus70B-tokens15T-it1155828"]
    # ["Apertus70B-tokens15T-long-context-64k"]="/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/Apertus70B-tokens15T-longcontext64k

    # aligned (only for uler)
    ["Apertus-8B-sft-mixture-8e-aligned"]="/capstor/store/cscs/swissai/infra01/swiss-alignment/checkpoints/Apertus-8B-sft-mixture-8e-aligned"
    ["Apertus-70B-sft-mixture-8e-aligned"]="/capstor/store/cscs/swissai/infra01/swiss-alignment/checkpoints/Apertus-70B-sft-mixture-8e-aligned"
)

export WANDB_ENTITY=${WANDB_ENTITY:-apertus}
export WANDB_PROJECT=${WANDB_PROJECT:-swissai-evals}

# Base model configurations
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-false}

# Call the common runner script
source runners/hf_base_runner.sh "Apertus base models"
