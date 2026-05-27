#!/bin/bash
#SBATCH --account=a127
#SBATCH --cpus-per-task=200
#SBATCH --gres=gpu:4
#SBATCH --job-name=evaluation
#SBATCH --mem=460000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --time=11:59:00
#SBATCH --exclude=nid006784
#SBATCH --exclude=nid007014
#SBATCH --exclude=nid007127
#SBATCH --exclude=nid007348
#SBATCH --exclude=nid007449
#SBATCH --exclude=nid007548
#SBATCH --exclude=nid007587
#SBATCH --exclude=nid006679
#SBATCH --exclude=nid006975
#SBATCH --exclude=nid007620

# Aux functions.
usage() {
	echo "Usage: sbatch evaluate.sbatch <model> <name>"
	echo "Runs evaluation of the specified model. The 'model' should a huggingface path/name."
	echo "The 'name' shall be used as an unique identifier."
	echo "In addition of these three necessary positional arguments, you can specify the following bash environment variables. Unless explicitly stated, all of these are optional:"
	echo " TOKENIZER: A huggingface tokenizer path/name. Needed if 'model' is a megatron checkpoint, otherwise the default will be set as the same value as 'model'."
	echo " BS: Batch size."
	echo " REVISION: Only used in huggingface models. If set, this revision of the model will be evaluated."
	echo " SIZE: The (approximate) size of the model in billions of parameters. Used to set model parallelism (needs to be set in larger models)."
	echo " MAX_LENGTH: The maximum length of the input sequence to the model. By default, it is unset."
	echo " MAX_NEW_TOKENS: The maximum number of new tokens to generate. By default, it is unset."
	echo " BOS: Set this to 'true' if you wish to prepend the BOS token when evaluating models."
	echo " LOGS_ROOT: Where are your evaluation wandb&harness logs going to."
	echo " TASKS: Tasks to run with lm eval harness."
	echo " WANDB_ENTITY: WandB entity name for uploading results (default: apertus)."
	echo " WANDB_PROJECT: WandB project name for uploading results (default: swissai-evals)."
	echo " LM_EVAL_BACKEND: Backend to use for lm-eval (default: hf). - options are: hf, vllm, megatron_lm"
    echo " CKPT_ITER: Megatron checkpoint iteration (integer). Required when LM_EVAL_BACKEND=megatron_lm."
    echo " HARNESS_LIMIT: Optional argument to pass as --limit to the lm-evaluation-harness, to limit the number of samples per task (default: no limit)."
    echo " LM_EVAL_HARNESS_BRANCH: Branch/ref to install from swiss-ai/lm-evaluation-harness (default: repository default branch)."
	echo "For more information see the README: https://github.com/swiss-ai/evals?tab=readme-ov-file."
}
die() {
	echo "$*" >& 2
	exit 1
}

# Wakeup logs.
set -e
echo "START TIME: $(date)"
echo "Using nodes: $SLURM_JOB_NODELIST"

# Grab variables and arguments.
if (( $# != 2 )); then
	usage
	die "Invalid usage: Invalid argument count"
fi
MODEL=$1
NAME=$2
CKPT_ITER=${CKPT_ITER:-"latest"}
BS=${BS:-"auto:20"}
SIZE=${SIZE:-1}
BOS=${BOS:-false}
LOGS_ROOT=${LOGS_ROOT:-/capstor/store/cscs/swissai/infra01/eval-logs}
TASKS=${TASKS:-./configs/apertus/tasks_default.txt}
TABLE_METRICS=${TABLE_METRICS:-./configs/apertus/tasks_default_main_table.txt}
WANDB_ENTITY=${WANDB_ENTITY:-apertus}
WANDB_PROJECT=${WANDB_PROJECT:-swissai-evals-test}
LM_EVAL_BACKEND=${LM_EVAL_BACKEND:-vllm}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-true}
HARNESS_LIMIT=${HARNESS_LIMIT:-""}
LM_EVAL_HARNESS_BRANCH=${LM_EVAL_HARNESS_BRANCH:-""}

export HF_TOKEN="${HF_TOKEN:-$(tr -d '\r\n' < ./scripts/hf_token.txt)}"
export WANDB_API_KEY="${WANDB_API_KEY:-$(tr -d '\r\n' < ./scripts/wandb_api_key.txt)}"
export CSCS_SERVING_API="${CSCS_SERVING_API:-$(tr -d '\r\n' < ./scripts/cscs_serving_api_key.txt)}"

warn_if_empty() {
  local var="$1"
  local msg="$2"

  if [[ -z "${!var:-}" ]]; then
    echo "WARNING: $var is not set. $msg" >&2
  fi
}

warn_if_empty HF_TOKEN \
  "This may cause Hugging Face rate-limit issues or prevent access to gated models/datasets."

warn_if_empty WANDB_API_KEY \
  "Weights & Biases logging is disabled. Results will NOT be uploaded."

warn_if_empty CSCS_SERVING_API \
  "Optional key missing. LLM-as-a-judge benchmarks via CSCS serving API will fail."

# optional eval arguments
MAX_LENGTH=${MAX_LENGTH:-4096}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
NUM_FEWSHOT=${NUM_FEWSHOT:-""}

# Task splitting support: set NUM_SPLITS and SPLIT_INDEX to distribute tasks across jobs
NUM_SPLITS=${NUM_SPLITS:-1}
SPLIT_INDEX=${SPLIT_INDEX:-0}

if [ -f "$TASKS" ]; then
    echo "Reading task list from file: $TASKS"
    # Filter out comments and blank lines, then join with commas
    TASKS=$(grep -v '^\s*#' "$TASKS" | grep -v '^\s*$' | paste -sd, -)

    # Split tasks across jobs if NUM_SPLITS > 1
    if (( NUM_SPLITS > 1 )); then
        IFS=',' read -ra TASK_ARRAY <<< "$TASKS"
        TOTAL_TASKS=${#TASK_ARRAY[@]}
        CHUNK_SIZE=$(( (TOTAL_TASKS + NUM_SPLITS - 1) / NUM_SPLITS ))
        START=$(( SPLIT_INDEX * CHUNK_SIZE ))
        END=$(( START + CHUNK_SIZE ))
        if (( END > TOTAL_TASKS )); then END=$TOTAL_TASKS; fi
        SPLIT_TASKS=()
        for (( i=START; i<END; i++ )); do
            SPLIT_TASKS+=("${TASK_ARRAY[$i]}")
        done
        TASKS=$(IFS=','; echo "${SPLIT_TASKS[*]}")
        echo "Split $((SPLIT_INDEX+1))/$NUM_SPLITS: running ${#SPLIT_TASKS[@]}/$TOTAL_TASKS tasks (indices $START-$((END-1)))"
    fi
fi

if [ -f "$TABLE_METRICS" ]; then
	echo "Reading table metrics from file: $TABLE_METRICS"
	TABLE_METRICS=$(grep -v '^\s*#' "$TABLE_METRICS" | grep -v '^\s*$' | paste -sd' ' -)
fi

# export HF_HOME=${HF_HOME:-/capstor/store/cscs/swissai/infra01/hf_home/}
export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/$USER/huggingface}
GPUS_PER_NODE=4

# Print configuration.
echo "Configuration set:"
printf "MODEL=$MODEL\nNAME=$NAME\nTOKENIZER=$TOKENIZER\nBS=$BS\nREVISION=$REVISION\nSIZE=$SIZE\nHARNESS_LIMIT=$HARNESS_LIMIT\nLM_EVAL_HARNESS_BRANCH=$LM_EVAL_HARNESS_BRANCH\nBOS=$BOS\nTASKS=$TASKS\nMAX_LENGTH=$MAX_LENGTH\nWANDB_ENTITY=$WANDB_ENTITY\nWANDB_PROJECT=$WANDB_PROJECT\nLM_EVAL_BACKEND=$LM_EVAL_BACKEND\nLHF_HOME=$HF_HOME\n\n"

# Generic envs.
export MASTER_ADDR=$(hostname)
export MASTER_PORT=25678
export CUDA_DEVICE_MAX_CONNECTIONS=1

export WORLD_SIZE=$GPUS_PER_NODE

# export HF_ALLOW_CODE_EVAL=1

echo "Huggingface checkpoint detected!"
if [ ! -z ${REVISION+x} ]; then
    echo "Using revision=$REVISION"
    MAYBE_REVISION=",revision=$REVISION"
    MAYBE_DOWNLOAD_REVISION="--revision=$REVISION"
fi
if [ -z ${TOKENIZER+x} ]; then
    echo "TOKENIZER not set, using TOKENIZER=$MODEL"
    TOKENIZER=$MODEL
fi


# Use only model parallel because of vLLM
MP=$GPUS_PER_NODE
echo "Using model size ~${SIZE}B, model parallel size of $MP"

# Prepare logs root.
RUN_ROOT=$LOGS_ROOT/$WANDB_ENTITY/$WANDB_PROJECT/$NAME
HARNESS_DIR=$RUN_ROOT/harness
export WANDB_DIR=$RUN_ROOT
mkdir -p $HARNESS_DIR

# export HF_HUB_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
# export PERSPECTIVE_API_KEY="" # Uncomment if you want to use perspective API, needed for realtoxicityprompts task

# Finally, run the actual evaluations.
HF_CHECKPOINT_PATH=$MODEL
DP=$(( GPUS_PER_NODE / MP ))
COMMON_MODEL_ARGS="dtype=bfloat16"
if [[ $BOS = true ]]; then
	echo "Adding BOS token."
	COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,add_bos_token=True"
fi

# Add vLLM specific arguments.
export OMP_NUM_THREADS=$(( 8 * $GPUS_PER_NODE ))
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_DISABLE_COMPILE_CACHE=1

#export VLLM_TORCH_COMPILE=0
#export NCCL_P2P_DISABLE=1

# export NCCL_IB_DISABLE=1
# export NCCL_NET="Socket"
# export NCCL_DEBUG=WARN
# Reduce memory fraction to avoid OOM during initialization
VLLM_MEMORY_FRACTION=0.75
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1

if [[ $LM_EVAL_BACKEND == "vllm" ]]; then
    COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,pretrained=$HF_CHECKPOINT_PATH,tokenizer=$TOKENIZER$MAYBE_REVISION,data_parallel_size=$DP,tensor_parallel_size=$MP,gpu_memory_utilization=$VLLM_MEMORY_FRACTION,enable_thinking=False"
elif [[ $LM_EVAL_BACKEND == "megatron_lm" ]]; then
    COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,load=$MODEL,tokenizer_type=HuggingFaceTokenizer,tokenizer_model=$TOKENIZER$MAYBE_REVISION,devices=4"
    if [[ "$CKPT_ITER" =~ ^[0-9]+$ ]]; then
        COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,ckpt_step=$CKPT_ITER"
    elif [[ "$CKPT_ITER" == "latest" ]]; then
        echo "CKPT_ITER=latest: omitting ckpt_step from Megatron model args."
    else
        die "CKPT_ITER must be an integer or 'latest' when LM_EVAL_BACKEND=megatron_lm (got '$CKPT_ITER')."
    fi
elif [[ $LM_EVAL_BACKEND == "hf" ]]; then
    COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,pretrained=$HF_CHECKPOINT_PATH,tokenizer=$TOKENIZER$MAYBE_REVISION"
else
    die "Unsupported LM_EVAL_BACKEND: $LM_EVAL_BACKEND"
fi
#,tokenizer_mode=slow"

HARNESS_EVAL_DIR=$HARNESS_DIR/eval_$(date +%Y%m%d_%H%M%S)_$SLURM_JOBID

COMMON_EVAL_ARGS=(
	--trust_remote_code
	--batch_size $BS
	--tasks $TASKS
	--output_path $HARNESS_EVAL_DIR
	--max_batch_size 32
	--log_samples
	--write_out
	--confirm_run_unsafe_code
    # --cache_requests true - causes problems for long model names due to file name length limits
    --gen_kwargs max_gen_toks=$MAX_NEW_TOKENS
)

if [[ -n "${HARNESS_LIMIT}" ]]; then
    COMMON_EVAL_ARGS+=(--limit $HARNESS_LIMIT)
fi

if [[ -n "${NUM_FEWSHOT}" ]]; then
	COMMON_EVAL_ARGS+=(--num_fewshot $NUM_FEWSHOT)
fi

if [[ $APPLY_CHAT_TEMPLATE = true ]]; then
	COMMON_EVAL_ARGS+=(--apply_chat_template)
	COMMON_EVAL_ARGS+=(--fewshot_as_multiturn)
fi

# Detect long context length only when 'ruler' is among tasks.
CONTEXT_LEN=""
if echo "$TASKS" | grep -q "ruler"; then
    if echo "$TASKS" | grep -qE "/[0-9]+"; then
        CONTEXT_LEN=$(echo "$TASKS" | grep -oE '/[0-9]+' | head -n1 | tr -d '/')
    fi
    # Fallback: support underscore or dash pattern like ruler_8192 or ruler-8192
    if [[ -z "$CONTEXT_LEN" ]]; then
        if echo "$TASKS" | grep -qE "ruler[_-][0-9]+"; then
            CONTEXT_LEN=$(echo "$TASKS" | grep -oE 'ruler[_-][0-9]+' | head -n1 | grep -oE '[0-9]+')
        fi
    fi
    if [[ -n "$CONTEXT_LEN" ]]; then
        export MAX_LENGTH="$CONTEXT_LEN"
        # Only add metadata and model arg if CONTEXT_LEN is numeric
        if [[ "$CONTEXT_LEN" =~ ^[0-9]+$ ]]; then
			COMMON_EVAL_ARGS+=( --metadata \'{\"max_seq_lengths\":[${CONTEXT_LEN}]}\' )

            MAX_MODEL_LEN_WITH_MARGIN=$(( CONTEXT_LEN + 600 ))
            export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
            COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,max_model_len=$MAX_MODEL_LEN_WITH_MARGIN"
            echo "Detected ruler context length: $CONTEXT_LEN (set MAX_LENGTH, max_model_len=$MAX_MODEL_LEN_WITH_MARGIN and lm-eval metadata)"
        else
            echo "Warning: Ruler detected but no numeric context length found; skipping metadata and max_model_len."
        fi
    fi
fi

LM_EVAL_HARNESS_PIP_SPEC="git+https://github.com/ymetz/lm-evaluation-harness.git"
if [[ -n "$LM_EVAL_HARNESS_BRANCH" ]]; then
    LM_EVAL_HARNESS_PIP_SPEC="${LM_EVAL_HARNESS_PIP_SPEC}@${LM_EVAL_HARNESS_BRANCH}"
fi

INSTALL_CMD="pip install --no-cache-dir --upgrade \"$LM_EVAL_HARNESS_PIP_SPEC\" && \
pip install --no-cache-dir --upgrade \
    \"huggingface-hub>=1.3.0,<2.0\" \
    \"transformers==5.6.2\" \
    \"accelerate>=1.1.0\" \
    \"antlr4-python3-runtime==4.11\" \
    datasketch \
    segtok \
    \"tokenizers==0.22.2\" \
    mistral-common \
    sympy \
    spacy \
    math_verify \
    latex2sympy2_extended \
    wonderwords \
    nltk \
    immutabledict \
    langdetect \
    hf-transfer \
    subset2evaluate \
    \"alpaca-eval>=0.6.2\" \
    \"peft>=0.18.0\"  "

if [[ $LM_EVAL_BACKEND == "megatron_lm" ]]; then
  INSTALL_CMD="$INSTALL_CMD && pip install megatron-core --no-deps && export MEGATRON_PATH=$(pwd)/Megatron-LM"
fi

if [[ $LM_EVAL_BACKEND == "vllm" ]]; then
    CMD="lm_eval --model $LM_EVAL_BACKEND --model_args=$COMMON_MODEL_ARGS ${COMMON_EVAL_ARGS[@]}"
elif [[ $LM_EVAL_BACKEND == "hf" ]]; then
    CMD="accelerate launch --num_processes=4  -m lm_eval --model $LM_EVAL_BACKEND --model_args=$COMMON_MODEL_ARGS ${COMMON_EVAL_ARGS[@]}"
elif [[ $LM_EVAL_BACKEND == "megatron_lm" ]]; then
    CMD="torchrun --nproc-per-node=4 -m lm_eval --model megatron_lm --model_args=$COMMON_MODEL_ARGS ${COMMON_EVAL_ARGS[@]}"
fi

echo "Installation command: $INSTALL_CMD"
echo "Final command: $CMD"

# Start timing for lines 182-189
echo "Starting timing for installation and evaluation..."
START_TIME=$(date +%s)

if [[ $LM_EVAL_BACKEND == "megatron_lm" ]]; then
  srun -n 1 --mpi=pmix -ul --environment=./containers/env_vllm.toml bash -lc "
    set -euo pipefail
    unset SSL_CERT_FILE

    $INSTALL_CMD
    # Make torch CUDA libs discoverable for vllm._C import
    export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:\${LD_LIBRARY_PATH:-}
    $CMD
    "
elif [[ $LM_EVAL_BACKEND == "vllm" ]]; then
  # vLLM needs to run with a single process that manages all GPUs, so we run the install and eval in the same command to avoid issues with environment persistence across srun calls
  srun -n 1 --mpi=pmix -ul --environment=./containers/env_vllm.toml bash -lc "
    set -euo pipefail
    unset SSL_CERT_FILE

    $INSTALL_CMD
    # Make torch CUDA libs discoverable for vllm._C import
    export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:\${LD_LIBRARY_PATH:-}
    export SWISS_AI_CHARTER_JUDGE_MODEL=meta-llama/Llama-3.3-70B-Instruct
    $CMD
    "
else
  # hf (via accelerate)
  srun -n 1 --mpi=pmix -ul --environment=./containers/env.toml --export=ALL,HF_TOKEN,HF_HOME,WANDB_API_KEY bash -lc "
    set -euo pipefail
    unset SSL_CERT_FILE

    $INSTALL_CMD

    # Make torch CUDA libs discoverable for vllm._C import - is just ignored for Huggingface
    export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:\${LD_LIBRARY_PATH:-}

    $CMD
    "
fi

# End timing and calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Time taken for installation and evaluation: ${DURATION} seconds ($(($DURATION / 60)) minutes and $(($DURATION % 60)) seconds)"

# Goodbye.
echo "Evaluation finished"

if (( NUM_SPLITS > 1 )); then
    # When running as a split job, write a marker file with the eval dir path
    # so the aggregation job can find all split results
    SPLIT_MARKER_DIR=$HARNESS_DIR/split_markers
    mkdir -p $SPLIT_MARKER_DIR
    echo "$HARNESS_EVAL_DIR" > "$SPLIT_MARKER_DIR/split_${SPLIT_INDEX}.txt"
    echo "Split job $((SPLIT_INDEX+1))/$NUM_SPLITS done. Skipping W&B upload (aggregation job will handle it)."
    echo "Results in: $HARNESS_EVAL_DIR"
    echo "Marker written to: $SPLIT_MARKER_DIR/split_${SPLIT_INDEX}.txt"
else
    echo "Uploading results to wandb"
    WANDB_CMD="cd $PWD && python -m scripts.alignment.update_wandb_alignment --entity $WANDB_ENTITY --project $WANDB_PROJECT --logs_root $HARNESS_EVAL_DIR --name $NAME --main_metrics $TABLE_METRICS --eval_duration $DURATION"
    echo "Running command to upload results to wandb:"
    echo "$WANDB_CMD"
    srun -ul --mpi=pmix --environment=./containers/env.toml bash -c " \
        $WANDB_CMD
    "
fi
echo "END TIME: $(date)"
