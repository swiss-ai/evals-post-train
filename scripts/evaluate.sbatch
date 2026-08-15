#!/bin/bash
#SBATCH --account=infra01
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
	echo " SIZE: Legacy/informational approximate model size in billions."
	echo " MAX_LENGTH: Optional backend context limit (vLLM max_model_len; Megatron seq_length)."
	echo " MAX_NEW_TOKENS: Thinking-run default for tasks without a task-local max_gen_toks (default: 8192 when thinking)."
	echo " VLLM_TP_SIZE / VLLM_DP_SIZE: Explicit vLLM topology. Defaults to TP=4/DP=1, or TP=1/DP=4 when a model name unambiguously indicates <30B."
	echo " VLLM_MEMORY_FRACTION: vLLM/SGLang GPU memory fraction (default: 0.8)."
	echo " MEGATRON_MICRO_BATCH_SIZE: Megatron micro-batch override; otherwise uses numeric BS or 1 when BS=auto."
	echo " BOS: Set this to 'true' if you wish to prepend the BOS token when evaluating models."
	echo " LOGS_ROOT: Where are your evaluation wandb&harness logs going to."
	echo " TASKS: Tasks to run with lm eval harness."
	echo " WANDB_ENTITY: WandB entity name for uploading results (default: apertus)."
	echo " WANDB_PROJECT: WandB project name for uploading results (default: swissai-evals-test)."
	echo " LM_EVAL_BACKEND: Backend to use for lm-eval (default: vllm). - options are: hf, vllm, sglang, megatron_lm, openai"
    echo " CKPT_ITER: Megatron checkpoint iteration (integer). Required when LM_EVAL_BACKEND=megatron_lm."
    echo " OpenAI-compatible API backend (LM_EVAL_BACKEND=openai; no model is loaded in the job):"
    echo "  API_BASE_URL: Endpoint to evaluate against. Required. Accepts a bare host (http://host:8000), a /v1 root, or a full /v1/(chat/)completions URL."
    echo "  API_CHAT_ENDPOINT: 'true' to use /v1/chat/completions (server-side chat template; generative tasks ONLY). Default: /v1/completions, which serves mixed suites."
    echo "  API_MODEL_NAME: 'model' field sent in requests (default: the <model> positional argument)."
    echo "  API_NUM_CONCURRENT: Concurrent requests (default: 8)."
    echo "  API_MAX_RETRIES: Retries per request (default: 3)."
    echo "  OPENAI_API_KEY: Bearer token for the endpoint (default: CSCS_SERVING_API)."
    echo " HARNESS_LIMIT: Optional argument to pass as --limit to the lm-evaluation-harness, to limit the number of samples per task (default: no limit)."
    echo " LM_EVAL_HARNESS_BRANCH: Branch/ref resolved once by the environment-preparation job."
    echo " EVAL_ENV_MANIFEST: Prepared base environment and harness overlay (required; set by launcher)."
    echo " EVAL_VERIFY_ENV_IMPORTS: 'true' to repeat the preparation job's deep Python import checks inside the evaluation job (default: false)."
    echo " EVAL_CHUNKS_FILE: Resumable array chunk manifest (set automatically by launcher)."
    echo " Thinking / reasoning metrics (hf and vllm backends only; megatron_lm is unsupported):"
    echo "  ENABLE_THINKING: 'true'/'false' chat-template argument deciding whether the model reasons (default: false)."
    echo "  AUTODETECT_THINK_TOKENS: 'true' to read the reasoning open/close tokens from the chat template (default: false)."
    echo "  THINK_END_TOKEN: Force the reasoning close token, e.g. '</think>'. A known close token arms the strip AND the metrics."
    echo "  THINK_START_TOKEN: Force the reasoning open token, e.g. '<think>'. Needed for thinking_format_has_open."
    echo "  TRACK_THINKING_METRICS: 'true'/'false' to force the thinking metrics on/off (default: on iff a close token is known)."
    echo "  LOG_LENGTH_METRICS: 'true' to aggregate response_length_*/thinking_length_* (thinking_format_* is always aggregated)."
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
LOGS_ROOT=${LOGS_ROOT:-$SCRATCH/eval_logs_start/}
TASKS=${TASKS:-./configs/apertus/tasks_default.txt}
TABLE_METRICS=${TABLE_METRICS:-./configs/apertus/tasks_default_main_table.txt}
WANDB_ENTITY=${WANDB_ENTITY:-apertus}
WANDB_PROJECT=${WANDB_PROJECT:-swissai-evals-test}
LM_EVAL_BACKEND=${LM_EVAL_BACKEND:-vllm}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-true}
HARNESS_LIMIT=${HARNESS_LIMIT:-""}
LM_EVAL_HARNESS_BRANCH=${LM_EVAL_HARNESS_BRANCH:-""}
EVAL_ENV_MANIFEST=${EVAL_ENV_MANIFEST:-""}
EVAL_CHUNKS_FILE=${EVAL_CHUNKS_FILE:-""}
EVAL_CHUNKED=${EVAL_CHUNKED:-false}
EVAL_VERIFY_ENV_IMPORTS=${EVAL_VERIFY_ENV_IMPORTS:-false}

case "${EVAL_VERIFY_ENV_IMPORTS,,}" in
    true|1|yes) EVAL_VERIFY_ENV_IMPORTS=true ;;
    false|0|no|"") EVAL_VERIFY_ENV_IMPORTS=false ;;
    *) die "EVAL_VERIFY_ENV_IMPORTS must be true or false, got: $EVAL_VERIFY_ENV_IMPORTS" ;;
esac

if [[ -n "$EVAL_CHUNKS_FILE" ]]; then
    [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || die "EVAL_CHUNKS_FILE requires a Slurm array task"
    [[ -f "$EVAL_CHUNKS_FILE" ]] || die "Chunk file not found: $EVAL_CHUNKS_FILE"
    TASKS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$EVAL_CHUNKS_FILE")
    [[ -n "$TASKS" ]] || die "No task chunk at array index $SLURM_ARRAY_TASK_ID"
    echo "Array chunk $SLURM_ARRAY_TASK_ID: $TASKS"
fi

[[ -n "$EVAL_ENV_MANIFEST" ]] || die "EVAL_ENV_MANIFEST is required; launch through scripts/launch_evaluations.sh"

load_prepared_eval_env() {
    EVAL_ENV_CHECK_ERROR=""
    unset EVAL_BASE_ENV EVAL_HARNESS_OVERLAY EVAL_RESOLVED_HARNESS_REPO \
        EVAL_RESOLVED_HARNESS_COMMIT EVAL_RESOLVED_BASE_KEY
    if [[ ! -f "$EVAL_ENV_MANIFEST" ]]; then
        EVAL_ENV_CHECK_ERROR="manifest is missing: $EVAL_ENV_MANIFEST"
        return 1
    fi
    # The manifest is generated by scripts/build_eval_env.sh and contains only
    # shell-escaped paths plus the resolved repository and commit.
    if ! source "$EVAL_ENV_MANIFEST"; then
        EVAL_ENV_CHECK_ERROR="manifest could not be sourced: $EVAL_ENV_MANIFEST"
        return 1
    fi
    # Do not test EVAL_BASE_ENV/bin/python with -x here. This shell runs outside
    # the container, while venv's python may be an absolute symlink to a path
    # that exists only inside it (for example /usr/local/bin/python). The
    # in-container preflight below performs the executable check.
    [[ -n "${EVAL_BASE_ENV:-}" ]] \
        || { EVAL_ENV_CHECK_ERROR="manifest has no EVAL_BASE_ENV"; return 1; }
    [[ -n "${EVAL_HARNESS_OVERLAY:-}" ]] \
        || { EVAL_ENV_CHECK_ERROR="manifest has no EVAL_HARNESS_OVERLAY"; return 1; }
    [[ -n "${EVAL_RESOLVED_HARNESS_REPO:-}" ]] \
        || { EVAL_ENV_CHECK_ERROR="manifest has no EVAL_RESOLVED_HARNESS_REPO"; return 1; }
    [[ -n "${EVAL_RESOLVED_HARNESS_COMMIT:-}" ]] \
        || { EVAL_ENV_CHECK_ERROR="manifest has no EVAL_RESOLVED_HARNESS_COMMIT"; return 1; }
    [[ -d "$EVAL_BASE_ENV" ]] \
        || { EVAL_ENV_CHECK_ERROR="base environment directory is missing: $EVAL_BASE_ENV"; return 1; }
    [[ -f "$EVAL_BASE_ENV/.complete" ]] \
        || { EVAL_ENV_CHECK_ERROR="base completion marker is missing: $EVAL_BASE_ENV/.complete"; return 1; }
    [[ -d "$EVAL_HARNESS_OVERLAY" ]] \
        || { EVAL_ENV_CHECK_ERROR="harness overlay directory is missing: $EVAL_HARNESS_OVERLAY"; return 1; }
    [[ -f "$EVAL_HARNESS_OVERLAY/.complete" ]] \
        || { EVAL_ENV_CHECK_ERROR="harness completion marker is missing: $EVAL_HARNESS_OVERLAY/.complete"; return 1; }
}

if ! load_prepared_eval_env; then
    echo "Prepared evaluation environment is missing or incomplete ($EVAL_ENV_CHECK_ERROR); rebuilding it now."
    : "${EVAL_CONTAINER_CONFIG:?EVAL_CONTAINER_CONFIG is required for environment repair}"
    : "${EVAL_REPO_ROOT:?EVAL_REPO_ROOT is required for environment repair}"
    : "${EVAL_HARNESS_REPO:?EVAL_HARNESS_REPO is required for environment repair}"
    # If the manifest was readable, keep this run pinned to its already-resolved
    # harness commit. Otherwise two simultaneous repairs of a moving branch
    # could observe different commits.
    if [[ "${EVAL_RESOLVED_HARNESS_COMMIT:-}" =~ ^[0-9a-fA-F]{40}$ ]]; then
        export EVAL_HARNESS_REF="$EVAL_RESOLVED_HARNESS_COMMIT"
        echo "Repairing the environment at pinned harness commit $EVAL_HARNESS_REF"
    fi
    srun -n 1 --mpi=pmix -ul --environment="$EVAL_CONTAINER_CONFIG" \
        --export=ALL bash "$EVAL_REPO_ROOT/scripts/build_eval_env.sh"
fi
if ! load_prepared_eval_env; then
    die "Evaluation environment repair did not produce a complete environment: $EVAL_ENV_CHECK_ERROR"
fi
# If a moving branch has advanced since preparation, a last-resort repair must
# still recreate this run's exact environment rather than silently changing it.
export EVAL_HARNESS_REF="$EVAL_RESOLVED_HARNESS_COMMIT"

# OpenAI-compatible API backend (LM_EVAL_BACKEND=openai).
API_BASE_URL=${API_BASE_URL:-""}
API_MODEL_NAME=${API_MODEL_NAME:-""}
API_NUM_CONCURRENT=${API_NUM_CONCURRENT:-8}
API_MAX_RETRIES=${API_MAX_RETRIES:-3}

export HF_TOKEN="${HF_TOKEN:-$(tr -d '\r\n' < ./scripts/hf_token.txt)}"
export WANDB_API_KEY="${WANDB_API_KEY:-$(tr -d '\r\n' < ./scripts/wandb_api_key.txt)}"
export CSCS_SERVING_API="${CSCS_SERVING_API:-$(tr -d '\r\n' < ./scripts/cscs_serving_api_key.txt)}"
if [[ -z "${OPENAI_API_KEY:-}" && -f ./scripts/openai_api_key.txt ]]; then
    export OPENAI_API_KEY="$(tr -d '\r\n' < ./scripts/openai_api_key.txt)"
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-${CSCS_SERVING_API:-}}"

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

# Optional eval arguments. An unset MAX_LENGTH leaves the backend/model default intact.
MAX_LENGTH=${MAX_LENGTH:-""}
if [[ -n "$MAX_LENGTH" && ! "$MAX_LENGTH" =~ ^[1-9][0-9]*$ ]]; then
    die "MAX_LENGTH must be a positive integer when set (got '$MAX_LENGTH')."
fi
# Capture explicitness before defaulting: thinking raises the default, but a caller value wins.
if [[ -n "${MAX_NEW_TOKENS:+x}" ]]; then MAX_NEW_TOKENS_EXPLICIT=1; else MAX_NEW_TOKENS_EXPLICIT=0; fi
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
NUM_FEWSHOT=${NUM_FEWSHOT:-""}

# --- Thinking / reasoning metrics ---
# Capture explicitness before defaulting: hf forwards enable_thinking only when set (unset lets
# the template default apply). ':+' treats an exported-but-empty value as unset.
if [[ -n "${ENABLE_THINKING:+x}" ]]; then ENABLE_THINKING_EXPLICIT=1; else ENABLE_THINKING_EXPLICIT=0; fi
ENABLE_THINKING=${ENABLE_THINKING:-false}
AUTODETECT_THINK_TOKENS=${AUTODETECT_THINK_TOKENS:-false}
THINK_END_TOKEN=${THINK_END_TOKEN:-""}
THINK_START_TOKEN=${THINK_START_TOKEN:-""}
TRACK_THINKING_METRICS=${TRACK_THINKING_METRICS:-""}   # "" | true | false
LOG_LENGTH_METRICS=${LOG_LENGTH_METRICS:-false}

# Validate booleans (don't coerce): a typo'd 'ture' read as false would waste a GPU job.
# Assign directly so a bad value aborts under set -e.
_norm_bool() {  # <name> <value> -> echoes True/False, dies otherwise
	local name="$1" val="$2"
	val="${val#"${val%%[![:space:]]*}"}"   # strip leading whitespace
	val="${val%"${val##*[![:space:]]}"}"   # strip trailing whitespace
	case "$val" in
		true|True|TRUE|1|yes)   echo "True" ;;
		false|False|FALSE|0|no) echo "False" ;;
		*) die "$name must be true or false (got '$2')." ;;
	esac
}
ENABLE_THINKING=$(_norm_bool ENABLE_THINKING "$ENABLE_THINKING")
AUTODETECT_THINK_TOKENS=$(_norm_bool AUTODETECT_THINK_TOKENS "$AUTODETECT_THINK_TOKENS")
LOG_LENGTH_METRICS=$(_norm_bool LOG_LENGTH_METRICS "$LOG_LENGTH_METRICS")
[[ -z "$TRACK_THINKING_METRICS" ]] || TRACK_THINKING_METRICS=$(_norm_bool TRACK_THINKING_METRICS "$TRACK_THINKING_METRICS")

# Thinking emits a trace before the answer; raise the budget unless the caller set it.
if [[ "$ENABLE_THINKING" == "True" ]] && (( ! MAX_NEW_TOKENS_EXPLICIT )); then
    MAX_NEW_TOKENS=16384
    echo "ENABLE_THINKING=True: raising default MAX_NEW_TOKENS to $MAX_NEW_TOKENS"
fi

# lm_eval splits --model_args on commas, so a comma in a token would inject an extra key.
_reject_comma() {
	[[ "$2" != *,* ]] || die "$1 must not contain a comma (got '$2'): it would inject an extra model_args key."
}
_reject_comma THINK_END_TOKEN "$THINK_END_TOKEN"
_reject_comma THINK_START_TOKEN "$THINK_START_TOKEN"
_reject_comma API_BASE_URL "$API_BASE_URL"
_reject_comma API_MODEL_NAME "$API_MODEL_NAME"

# Reasoning model_args shared by hf and vllm. Only non-defaults are emitted (an unset flag
# keeps the harness default).
THINK_ARGS=""
[[ -n "$THINK_END_TOKEN" ]] && THINK_ARGS="${THINK_ARGS},think_end_token=${THINK_END_TOKEN}"
[[ -n "$THINK_START_TOKEN" ]] && THINK_ARGS="${THINK_ARGS},think_start_token=${THINK_START_TOKEN}"
[[ "$AUTODETECT_THINK_TOKENS" == "True" ]] && THINK_ARGS="${THINK_ARGS},autodetect_think_tokens=True"
[[ -n "$TRACK_THINKING_METRICS" ]] && THINK_ARGS="${THINK_ARGS},track_thinking_metrics=${TRACK_THINKING_METRICS}"

if [ -f "$TASKS" ]; then
    echo "Reading task list from file: $TASKS"
    # Filter out comments and blank lines, then join with commas
    TASKS=$(grep -v '^\s*#' "$TASKS" | grep -v '^\s*$' | paste -sd, -)
fi

if [ -f "$TABLE_METRICS" ]; then
	echo "Reading table metrics from file: $TABLE_METRICS"
	TABLE_METRICS=$(grep -v '^\s*#' "$TABLE_METRICS" | grep -v '^\s*$' | paste -sd' ' -)
fi

# export HF_HOME=${HF_HOME:-/capstor/store/cscs/swissai/infra01/hf_home/}
export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/$USER/huggingface}

# Redirect all caches to SCRATCH to avoid filling up HOME
CACHE_ROOT="${SCRATCH:-/tmp}/.cache"
export HF_DATASETS_CACHE="$CACHE_ROOT/huggingface/datasets"
export NLTK_DATA="$CACHE_ROOT/nltk_data"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TIKTOKEN_CACHE_DIR="$CACHE_ROOT/tiktoken"
export PYTHONUSERBASE="$CACHE_ROOT/python_userbase"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export SPACY_DATA="$CACHE_ROOT/spacy"
export XDG_CACHE_HOME="$CACHE_ROOT"
mkdir -p "$HF_DATASETS_CACHE" "$NLTK_DATA" "$TRITON_CACHE_DIR" "$TIKTOKEN_CACHE_DIR" "$PYTHONUSERBASE" "$MPLCONFIGDIR" "$SPACY_DATA"

GPUS_PER_NODE=4

# Print configuration.
echo "Configuration set:"
# Values go in printf's arguments, not the format string: a '%' would be read as a conversion.
printf '%s\n' "MODEL=$MODEL" "NAME=$NAME" "TOKENIZER=$TOKENIZER" "BS=$BS" "REVISION=$REVISION" \
	"SIZE=$SIZE" "HARNESS_LIMIT=$HARNESS_LIMIT" "LM_EVAL_HARNESS_BRANCH=$LM_EVAL_HARNESS_BRANCH" \
	"BOS=$BOS" "TASKS=$TASKS" "MAX_LENGTH=$MAX_LENGTH" "WANDB_ENTITY=$WANDB_ENTITY" \
	"WANDB_PROJECT=$WANDB_PROJECT" "LM_EVAL_BACKEND=$LM_EVAL_BACKEND" "HF_HOME=$HF_HOME" ""
printf '%s\n' "ENABLE_THINKING=$ENABLE_THINKING (explicit=$ENABLE_THINKING_EXPLICIT)" \
	"AUTODETECT_THINK_TOKENS=$AUTODETECT_THINK_TOKENS" "THINK_START_TOKEN=$THINK_START_TOKEN" \
	"THINK_END_TOKEN=$THINK_END_TOKEN" "TRACK_THINKING_METRICS=$TRACK_THINKING_METRICS" \
	"LOG_LENGTH_METRICS=$LOG_LENGTH_METRICS" ""

# Generic envs.
export MASTER_ADDR=$(hostname)
export MASTER_PORT=25678

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


# Preserve the historically compatible TP=4/DP=1 default. For an unambiguous
# sub-30B model name, prefer four independent replicas for throughput. Explicit
# VLLM_TP_SIZE/VLLM_DP_SIZE values always win; either side may be omitted and is
# derived from the four allocated GPUs.
MP=$GPUS_PER_NODE
DP=1
DETECTED_MODEL_SIZE_B=""
if [[ "$LM_EVAL_BACKEND" == "vllm" ]]; then
    VLLM_TP_SIZE=${VLLM_TP_SIZE:-""}
    VLLM_DP_SIZE=${VLLM_DP_SIZE:-""}
    VLLM_AUTO_PARALLELISM=${VLLM_AUTO_PARALLELISM:-true}
    VLLM_AUTO_PARALLELISM=$(_norm_bool VLLM_AUTO_PARALLELISM "$VLLM_AUTO_PARALLELISM")

    if [[ -n "$VLLM_TP_SIZE" || -n "$VLLM_DP_SIZE" ]]; then
        [[ -z "$VLLM_TP_SIZE" || "$VLLM_TP_SIZE" =~ ^[1-9][0-9]*$ ]] \
            || die "VLLM_TP_SIZE must be a positive integer (got '$VLLM_TP_SIZE')."
        [[ -z "$VLLM_DP_SIZE" || "$VLLM_DP_SIZE" =~ ^[1-9][0-9]*$ ]] \
            || die "VLLM_DP_SIZE must be a positive integer (got '$VLLM_DP_SIZE')."
        if [[ -z "$VLLM_TP_SIZE" ]]; then
            (( GPUS_PER_NODE % VLLM_DP_SIZE == 0 )) \
                || die "VLLM_DP_SIZE=$VLLM_DP_SIZE does not divide $GPUS_PER_NODE GPUs."
            VLLM_TP_SIZE=$(( GPUS_PER_NODE / VLLM_DP_SIZE ))
        elif [[ -z "$VLLM_DP_SIZE" ]]; then
            (( GPUS_PER_NODE % VLLM_TP_SIZE == 0 )) \
                || die "VLLM_TP_SIZE=$VLLM_TP_SIZE does not divide $GPUS_PER_NODE GPUs."
            VLLM_DP_SIZE=$(( GPUS_PER_NODE / VLLM_TP_SIZE ))
        fi
        (( VLLM_TP_SIZE * VLLM_DP_SIZE == GPUS_PER_NODE )) \
            || die "VLLM_TP_SIZE * VLLM_DP_SIZE must equal $GPUS_PER_NODE GPUs."
        MP=$VLLM_TP_SIZE
        DP=$VLLM_DP_SIZE
        echo "Using explicit vLLM topology: TP=$MP DP=$DP"
    elif [[ "$VLLM_AUTO_PARALLELISM" == "True" ]]; then
        DETECTED_MODEL_SIZE_B=$(python3 -m scripts.infer_model_size "$MODEL" 2>/dev/null || true)
        if [[ -n "$DETECTED_MODEL_SIZE_B" ]] \
            && awk -v size="$DETECTED_MODEL_SIZE_B" 'BEGIN { exit !(size < 30) }'; then
            MP=1
            DP=$GPUS_PER_NODE
            echo "Detected ${DETECTED_MODEL_SIZE_B}B model from path; using faster vLLM topology TP=$MP DP=$DP"
        else
            echo "No unambiguous <30B model size detected; using compatible vLLM topology TP=$MP DP=$DP"
        fi
    else
        echo "vLLM topology auto-detection disabled; using TP=$MP DP=$DP"
    fi
else
    echo "Using backend topology TP=$MP DP=$DP"
fi

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
COMMON_MODEL_ARGS="dtype=bfloat16"
if [[ $BOS = true ]]; then
	echo "Adding BOS token."
	COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,add_bos_token=True"
fi

# Add vLLM specific arguments.
export OMP_NUM_THREADS=$(( 8 * $GPUS_PER_NODE ))
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 # might crash on some nodes

#export VLLM_TORCH_COMPILE=0
#export NCCL_P2P_DISABLE=1

# export NCCL_IB_DISABLE=1
# export NCCL_NET="Socket"
# export NCCL_DEBUG=WARN
# Keep headroom for node-to-node variation while exposing more KV-cache space.
VLLM_MEMORY_FRACTION=${VLLM_MEMORY_FRACTION:-0.8}
awk -v fraction="$VLLM_MEMORY_FRACTION" \
    'BEGIN { exit !(fraction > 0 && fraction <= 1) }' \
    || die "VLLM_MEMORY_FRACTION must be in (0, 1] (got '$VLLM_MEMORY_FRACTION')."
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1

# The harness only implements the reasoning producers for hf/vllm/sglang; fail loudly otherwise.
if [[ $LM_EVAL_BACKEND == "megatron_lm" || $LM_EVAL_BACKEND == "openai" ]]; then
    if [[ -n "$THINK_ARGS" ]] || (( ENABLE_THINKING_EXPLICIT )) || [[ "$LOG_LENGTH_METRICS" == "True" ]]; then
        die "Thinking/reasoning metrics are unsupported with LM_EVAL_BACKEND=$LM_EVAL_BACKEND."
    fi
fi

THINKING_DEFAULT_MODEL_ARG=""
if [[ "$ENABLE_THINKING" == "True" ]]; then
    # This is a backend default, not a CLI --gen_kwargs override: task-local
    # generation_kwargs therefore retain priority.
    THINKING_DEFAULT_MODEL_ARG=",max_gen_toks=$MAX_NEW_TOKENS"
fi

if [[ $LM_EVAL_BACKEND == "vllm" ]]; then
    COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,pretrained=$HF_CHECKPOINT_PATH,tokenizer=$TOKENIZER$MAYBE_REVISION,language_model_only=True,data_parallel_size=$DP,tensor_parallel_size=$MP,gpu_memory_utilization=$VLLM_MEMORY_FRACTION,enable_thinking=${ENABLE_THINKING}${THINK_ARGS}${THINKING_DEFAULT_MODEL_ARG},disable_flashinfer_allreduce_fusion=True"
elif [[ $LM_EVAL_BACKEND == "sglang" ]]; then
    COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,pretrained=$HF_CHECKPOINT_PATH,tokenizer_path=$TOKENIZER$MAYBE_REVISION,dp_size=$DP,tp_size=$MP,mem_fraction_static=$VLLM_MEMORY_FRACTION,dtype=auto${THINKING_DEFAULT_MODEL_ARG}"
elif [[ $LM_EVAL_BACKEND == "megatron_lm" ]]; then
    MEGATRON_MICRO_BATCH_SIZE=${MEGATRON_MICRO_BATCH_SIZE:-""}
    if [[ -z "$MEGATRON_MICRO_BATCH_SIZE" ]]; then
        if [[ "$BS" =~ ^[1-9][0-9]*$ ]]; then
            MEGATRON_MICRO_BATCH_SIZE=$BS
        else
            MEGATRON_MICRO_BATCH_SIZE=1
            echo "Megatron does not support BS=$BS auto-detection; using micro_batch_size=1 (override with MEGATRON_MICRO_BATCH_SIZE or numeric BS)."
        fi
    fi
    [[ "$MEGATRON_MICRO_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] \
        || die "MEGATRON_MICRO_BATCH_SIZE must be a positive integer (got '$MEGATRON_MICRO_BATCH_SIZE')."
    BS=$MEGATRON_MICRO_BATCH_SIZE
    COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,load=$MODEL,tokenizer_type=HuggingFaceTokenizer,tokenizer_model=$TOKENIZER$MAYBE_REVISION,devices=4,micro_batch_size=$MEGATRON_MICRO_BATCH_SIZE${THINKING_DEFAULT_MODEL_ARG}"
    echo "Megatron effective batch size: $BS"
    if [[ "$CKPT_ITER" =~ ^[0-9]+$ ]]; then
        COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,ckpt_step=$CKPT_ITER"
    elif [[ "$CKPT_ITER" == "latest" ]]; then
        echo "CKPT_ITER=latest: omitting ckpt_step from Megatron model args."
    else
        die "CKPT_ITER must be an integer or 'latest' when LM_EVAL_BACKEND=megatron_lm (got '$CKPT_ITER')."
    fi
elif [[ $LM_EVAL_BACKEND == "hf" ]]; then
    COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,pretrained=$HF_CHECKPOINT_PATH,tokenizer=$TOKENIZER$MAYBE_REVISION"
    # Forward enable_thinking only when explicitly set; unset lets the template default apply.
    (( ENABLE_THINKING_EXPLICIT )) && COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,enable_thinking=${ENABLE_THINKING}"
    COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS${THINK_ARGS}"
    if [[ "$ENABLE_THINKING" == "True" ]]; then
        echo "HF backend: preserving task-local max_gen_toks; no model-level thinking default is available."
    fi
elif [[ $LM_EVAL_BACKEND == "openai" ]]; then
    [[ -n "$API_BASE_URL" ]] || die "API_BASE_URL must be set when LM_EVAL_BACKEND=openai (e.g. http://<host>:8000/v1)."
    API_MODEL_NAME=${API_MODEL_NAME:-$MODEL}
    # Always the completions endpoint: it serves BOTH generative and loglikelihood/MC
    # requests, so mixed suites work (chat/completions raises NotImplementedError on
    # loglikelihood). With the chat template on, the harness renders the model's template
    # client-side via the HF tokenizer. For purely generative suites the server-side
    # template path can be forced with API_CHAT_ENDPOINT=true.
    API_CHAT_ENDPOINT=${API_CHAT_ENDPOINT:-false}
    if [[ "$API_CHAT_ENDPOINT" == "true" ]]; then
        [[ $APPLY_CHAT_TEMPLATE = true ]] || die "API_CHAT_ENDPOINT=true requires the chat template (chat/completions takes messages)."
        API_LM_EVAL_MODEL="local-chat-completions"
        API_ENDPOINT_SUFFIX="chat/completions"
    else
        API_LM_EVAL_MODEL="local-completions"
        API_ENDPOINT_SUFFIX="completions"
    fi
    # Accept a bare host, a /v1 root, or an already-full endpoint URL.
    if [[ "$API_BASE_URL" != */chat/completions && "$API_BASE_URL" != */completions ]]; then
        API_BASE_URL="${API_BASE_URL%/}"
        [[ "$API_BASE_URL" == */v1 ]] || API_BASE_URL="$API_BASE_URL/v1"
        API_BASE_URL="$API_BASE_URL/$API_ENDPOINT_SUFFIX"
    fi
    # Rebuilt from scratch: no model is loaded in-job, so dtype etc. do not apply.
    COMMON_MODEL_ARGS="model=$API_MODEL_NAME,base_url=$API_BASE_URL,num_concurrent=$API_NUM_CONCURRENT,max_retries=$API_MAX_RETRIES"
    if [[ $API_LM_EVAL_MODEL == "local-completions" ]]; then
        # Client-side tokenization: loglikelihood scoring needs token-aligned context/continuation.
        COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,tokenizer_backend=huggingface,tokenizer=$TOKENIZER"
        [[ $BOS = true ]] && COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,add_bos_token=True"
    else
        COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,tokenized_requests=False"
    fi
    # API models take an integer batch size; throughput comes from num_concurrent instead.
    if [[ "$BS" == auto* ]]; then
        BS=1
        echo "openai backend: batch size 'auto' unsupported, using BS=$BS (concurrency: API_NUM_CONCURRENT=$API_NUM_CONCURRENT)"
    fi
    echo "openai backend: $API_LM_EVAL_MODEL against $API_BASE_URL (model=$API_MODEL_NAME)"
else
    die "Unsupported LM_EVAL_BACKEND: $LM_EVAL_BACKEND"
fi

HARNESS_EVAL_DIR=$HARNESS_DIR/eval_$(date +%Y%m%d_%H%M%S)_$SLURM_JOBID

# Generation kwargs supplied through the CLI update every task's YAML. Deliberately
# omit max_gen_toks here so task-local budgets retain priority. Thinking's longer
# fallback is passed as a model default above and is used only when a task has no
# local max_gen_toks.
GEN_KWARGS=""
if [[ "$ENABLE_THINKING" == "True" ]]; then
    THINK_TEMPERATURE=${THINK_TEMPERATURE:-0.6}
    THINK_TOP_P=${THINK_TOP_P:-0.95}
    GEN_KWARGS="do_sample=true,temperature=${THINK_TEMPERATURE},top_p=${THINK_TOP_P}"
    # Optional knob for degenerate looping (e.g. 1.05); unset leaves vLLM's default (1.0).
    if [[ -n "${THINK_REPETITION_PENALTY:-}" ]]; then
        GEN_KWARGS="${GEN_KWARGS},repetition_penalty=${THINK_REPETITION_PENALTY}"
    fi
    echo "Thinking sampling: do_sample=true temperature=$THINK_TEMPERATURE top_p=$THINK_TOP_P repetition_penalty=${THINK_REPETITION_PENALTY:-<unset>} (task-local max_gen_toks takes priority; backend default=$MAX_NEW_TOKENS)"
elif [[ -n "${NOTHINK_TEMPERATURE:-}" ]]; then
    # Optional no-think sampling (inert unless NOTHINK_TEMPERATURE is set): greedy-vs-sampling ablation.
    NOTHINK_TOP_P=${NOTHINK_TOP_P:-0.95}
    GEN_KWARGS="do_sample=true,temperature=${NOTHINK_TEMPERATURE},top_p=${NOTHINK_TOP_P}"
    echo "No-think sampling: do_sample=true temperature=$NOTHINK_TEMPERATURE top_p=$NOTHINK_TOP_P"
fi

MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-""}

COMMON_EVAL_ARGS=(
	--trust_remote_code
	--batch_size $BS
	--tasks $TASKS
	--output_path $HARNESS_EVAL_DIR
	--log_samples
	--write_out
	--confirm_run_unsafe_code
	--cache_requests true # was disabled until recently, but we can enable it because the upstream file length problem was patched
)

if [[ -n "$GEN_KWARGS" ]]; then
    COMMON_EVAL_ARGS+=(--gen_kwargs "$GEN_KWARGS")
fi

if [[ -n "$MAX_BATCH_SIZE" ]]; then
    COMMON_EVAL_ARGS+=(--max_batch_size "$MAX_BATCH_SIZE")
fi

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

# Aggregates response_length_*/thinking_length_* into results.json and W&B (thinking_format_*
# is aggregated whenever tracking is on, flag or not).
if [[ "$LOG_LENGTH_METRICS" == "True" ]]; then
	COMMON_EVAL_ARGS+=(--log_length_metrics)
fi

# Detect long context length only when 'ruler' is among tasks. RULER needs a
# little engine headroom beyond the requested sequence length.
CONTEXT_LEN=""
BACKEND_MAX_LENGTH="$MAX_LENGTH"
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
        # Only add metadata and model arg if CONTEXT_LEN is numeric
        if [[ "$CONTEXT_LEN" =~ ^[0-9]+$ ]]; then
			COMMON_EVAL_ARGS+=( --metadata \'{\"max_seq_lengths\":[${CONTEXT_LEN}]}\' )

            BACKEND_MAX_LENGTH=$(( CONTEXT_LEN + 600 ))
            echo "Detected ruler context length: $CONTEXT_LEN (backend context limit=$BACKEND_MAX_LENGTH and lm-eval metadata)"
        else
            echo "Warning: Ruler detected but no numeric context length found; skipping metadata and automatic context limit."
        fi
    fi
fi

# MAX_LENGTH is a backend context limit, not a generation override. Keep each
# adapter's native argument name and omit it entirely when no value was set.
if [[ -n "$BACKEND_MAX_LENGTH" ]]; then
    case "$LM_EVAL_BACKEND" in
        vllm)
            COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,max_model_len=$BACKEND_MAX_LENGTH"
            [[ -n "$CONTEXT_LEN" ]] && export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
            ;;
        sglang)
            COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,max_model_len=$BACKEND_MAX_LENGTH"
            ;;
        megatron_lm)
            COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,seq_length=$BACKEND_MAX_LENGTH"
            ;;
        hf|openai)
            COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,max_length=$BACKEND_MAX_LENGTH"
            ;;
    esac
    echo "Using $LM_EVAL_BACKEND backend context limit: $BACKEND_MAX_LENGTH"
fi

# bfcl_v3 and Swiss AI Charter Alignment are not yet available in the Swiss-AI
# fork. Use ymetz only for jobs containing those tasks; everything else,
# including the experimental GPT suite, stays on Swiss-AI.
if echo "$TASKS" | grep -qE '(^|,)(bfcl_v3([^,]*)?|swiss_ai_charter_alignment([^,]*)?)(,|$)'; then
    if [[ "$EVAL_RESOLVED_HARNESS_REPO" != "ymetz/lm-evaluation-harness" ]]; then
        die "BFCL/Charter tasks require ymetz/lm-evaluation-harness, got $EVAL_RESOLVED_HARNESS_REPO"
    fi
fi
echo "Using lm-evaluation-harness: $EVAL_RESOLVED_HARNESS_REPO@$EVAL_RESOLVED_HARNESS_COMMIT"

LOAD_ENV_CMD="
echo Container entered: \$(date -Is)
eval_env_is_ready() {
    [[ -f '$EVAL_BASE_ENV/.complete' ]] \
        && [[ -x '$EVAL_BASE_ENV/bin/python' ]] \
        && [[ -f '$EVAL_HARNESS_OVERLAY/.complete' ]] \
        && [[ -d '$EVAL_HARNESS_OVERLAY/lm_eval' ]] \
        || return 1
    if [[ '$EVAL_VERIFY_ENV_IMPORTS' == 'true' ]]; then
        '$EVAL_BASE_ENV/bin/python' -c 'import accelerate, datasets, transformers, wandb' >/dev/null 2>&1 \
            && PYTHONPATH='$EVAL_HARNESS_OVERLAY' '$EVAL_BASE_ENV/bin/python' -c 'import lm_eval' >/dev/null 2>&1 \
            || return 1
    fi
}
if ! eval_env_is_ready; then
    echo 'Cached evaluation environment failed its in-container preflight; rebuilding it now.'
    bash '$EVAL_REPO_ROOT/scripts/build_eval_env.sh'
    source '$EVAL_ENV_MANIFEST'
fi
if ! eval_env_is_ready; then
    echo 'Evaluation environment repair did not pass its in-container preflight.' >&2
    exit 1
fi
source '$EVAL_BASE_ENV/bin/activate'
export PYTHONPATH='$EVAL_HARNESS_OVERLAY'\${PYTHONPATH:+:\$PYTHONPATH}
echo Evaluation environment activated: \$(date -Is) deep_import_check=$EVAL_VERIFY_ENV_IMPORTS
"
if [[ $LM_EVAL_BACKEND == "megatron_lm" && -d "$PWD/Megatron-LM" ]]; then
    LOAD_ENV_CMD="$LOAD_ENV_CMD
export MEGATRON_PATH='$PWD/Megatron-LM'
"
fi

# $CMD is re-parsed by the inner `bash -lc`, so --model_args is single-quoted (a token like
# think_end_token=</think> would otherwise redirect, and $(...) in a path would execute).
# A single quote in the value can't be nested safely, so refuse it.
if [[ "$COMMON_MODEL_ARGS" == *\'* ]]; then
    die "Model args must not contain a single quote: $COMMON_MODEL_ARGS"
fi

if [[ $LM_EVAL_BACKEND == "vllm" ]]; then
    CMD="python -m lm_eval --model $LM_EVAL_BACKEND --model_args='$COMMON_MODEL_ARGS' ${COMMON_EVAL_ARGS[@]}"
elif [[ $LM_EVAL_BACKEND == "sglang" ]]; then
    CMD="python -m lm_eval --model $LM_EVAL_BACKEND --model_args='$COMMON_MODEL_ARGS' ${COMMON_EVAL_ARGS[@]}"
elif [[ $LM_EVAL_BACKEND == "openai" ]]; then
    CMD="python -m lm_eval --model $API_LM_EVAL_MODEL --model_args='$COMMON_MODEL_ARGS' ${COMMON_EVAL_ARGS[@]}"
elif [[ $LM_EVAL_BACKEND == "hf" ]]; then
    CMD="accelerate launch --num_processes=4  -m lm_eval --model $LM_EVAL_BACKEND --model_args='$COMMON_MODEL_ARGS' ${COMMON_EVAL_ARGS[@]}"
elif [[ $LM_EVAL_BACKEND == "megatron_lm" ]]; then
    CMD="torchrun --nproc-per-node=4 -m lm_eval --model megatron_lm --model_args='$COMMON_MODEL_ARGS' ${COMMON_EVAL_ARGS[@]}"
fi
TIMED_CMD="echo Evaluation process starting: \$(date -Is)
$CMD"

echo "Environment manifest: $EVAL_ENV_MANIFEST"
echo "Final command: $CMD"

# Environment preparation happens once in a dependency job. This duration is
# therefore evaluation-only and remains meaningful when results are aggregated.
START_TIME=$(date +%s)
echo "Starting evaluation timing: $(date -Is)"

# Cache redirection exports to pass into the container
CACHE_EXPORTS="
export HF_HOME=$HF_HOME
export HF_DATASETS_CACHE=$HF_DATASETS_CACHE
export NLTK_DATA=$NLTK_DATA
export TRITON_CACHE_DIR=$TRITON_CACHE_DIR
export TIKTOKEN_CACHE_DIR=$TIKTOKEN_CACHE_DIR
export PYTHONUSERBASE=$PYTHONUSERBASE
export MPLCONFIGDIR=$MPLCONFIGDIR
export SPACY_DATA=$SPACY_DATA
export XDG_CACHE_HOME=$XDG_CACHE_HOME
mkdir -p $HF_DATASETS_CACHE $NLTK_DATA $TRITON_CACHE_DIR $TIKTOKEN_CACHE_DIR $PYTHONUSERBASE $MPLCONFIGDIR $SPACY_DATA
"

if [[ $LM_EVAL_BACKEND == "megatron_lm" ]]; then
  srun -n 1 --mpi=pmix -ul --environment=./containers/env_vllm.toml bash -lc "
    set -euo pipefail
    unset SSL_CERT_FILE
    export HF_DATASETS_TRUST_REMOTE_CODE=1
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    $CACHE_EXPORTS

    $LOAD_ENV_CMD
    # Make torch CUDA libs discoverable for vllm._C import
    export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:\${LD_LIBRARY_PATH:-}
    $TIMED_CMD
    "
elif [[ $LM_EVAL_BACKEND == "vllm" ]]; then
  # vLLM runs as one process which manages all GPUs.
  srun -n 1 --mpi=pmix -ul --environment=./containers/env_vllm.toml bash -lc "
    set -euo pipefail
    unset SSL_CERT_FILE
    export HF_DATASETS_TRUST_REMOTE_CODE=1
    $CACHE_EXPORTS

    $LOAD_ENV_CMD
    # Make torch CUDA libs discoverable for vllm._C import
    export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:\${LD_LIBRARY_PATH:-}
    $TIMED_CMD
    "
elif [[ $LM_EVAL_BACKEND == "sglang" ]]; then
  srun -n 1 --mpi=pmix -ul --environment=./containers/env_sglang.toml bash -lc "
    set -euo pipefail
    unset SSL_CERT_FILE
    export HF_DATASETS_TRUST_REMOTE_CODE=1
    $CACHE_EXPORTS

    $LOAD_ENV_CMD
    $TIMED_CMD
    "
elif [[ $LM_EVAL_BACKEND == "openai" ]]; then
  # No model is loaded in-job; the harness only makes HTTP calls against API_BASE_URL.
  srun -n 1 --mpi=pmix -ul --environment=./containers/env.toml --export=ALL,HF_TOKEN,HF_HOME,WANDB_API_KEY,OPENAI_API_KEY bash -lc "
    set -euo pipefail
    unset SSL_CERT_FILE
    export HF_DATASETS_TRUST_REMOTE_CODE=1
    $CACHE_EXPORTS

    $LOAD_ENV_CMD

    $TIMED_CMD
    "
else
  # hf (via accelerate)
  srun -n 1 --mpi=pmix -ul --environment=./containers/env.toml --export=ALL,HF_TOKEN,HF_HOME,WANDB_API_KEY bash -lc "
    set -euo pipefail
    unset SSL_CERT_FILE
    export HF_DATASETS_TRUST_REMOTE_CODE=1
    $CACHE_EXPORTS

    $LOAD_ENV_CMD

    # Make torch CUDA libs discoverable for vllm._C import - is just ignored for Huggingface
    export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:\${LD_LIBRARY_PATH:-}

    $TIMED_CMD
    "
fi

# End timing and calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Evaluation time: ${DURATION} seconds ($(($DURATION / 60)) minutes and $(($DURATION % 60)) seconds)"

# Goodbye.
echo "Evaluation finished"

if [[ "$EVAL_CHUNKED" == "true" ]]; then
    echo "Chunk job done. Skipping W&B upload; the controller will aggregate completed chunks."
    echo "Results in: $HARNESS_EVAL_DIR"
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
