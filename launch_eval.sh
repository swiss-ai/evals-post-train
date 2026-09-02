#!/bin/bash
# RULER long-context sweep, same invocation as evals-post-train used for these runs.

MODELS=(
    /capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200_corr
)

export WANDB_ENTITY=danitamayo-adhoc
export MAX_NEW_TOKENS=128
export HF_TOKEN=$(tr -d '\r\n' < ./scripts/hf_token.txt)

for model in ${MODELS[@]}; do

    export CONTEXT_LEN=4096,8192,16384,32768,65536
    CUSTOM_TYPE=olmo3_ruler_large
    bash scripts/launch_evaluations.sh $CUSTOM_TYPE --model $model --backend vllm --chat-template --splits 13

    export CONTEXT_LEN=128000
    bash scripts/launch_evaluations.sh $CUSTOM_TYPE --model $model --backend vllm --chat-template --splits 13

    export CONTEXT_LEN=261000
    bash scripts/launch_evaluations.sh $CUSTOM_TYPE --model $model --backend vllm --chat-template --splits 13

done
