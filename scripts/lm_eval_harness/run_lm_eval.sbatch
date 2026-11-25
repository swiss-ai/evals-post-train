#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=16
#SBATCH --job-name=evaluation
#SBATCH --mem=64000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=normal
#SBATCH --time=11:59:00

# Aux functions.
usage() {
	echo "Usage: sbatch run_lm_eval.sbatch <model> <name>"
	echo "Evaluates a model served by an OpenAI-compatible endpoint (API_BASE_URL); no model is loaded in the job. The 'model' is its name."
	echo "The 'name' shall be used as an unique identifier."
	echo "In addition of these three necessary positional arguments, you can specify the following bash environment variables. Unless explicitly stated, all of these are optional:"
	echo " TOKENIZER: A huggingface tokenizer path/name. Default: 'model', with an endpoint catalog prefix stripped."
	echo " BS: Batch size."
	echo " MAX_LENGTH: Optional context limit: the served window."
	echo " MAX_NEW_TOKENS: Generation budget of thinking/reasoning-effort runs; replaces every task's own max_gen_toks (default when thinking: 32768; with MAX_LENGTH set, MAX_LENGTH, which the harness clamps per request to the window minus the prompt)."
	echo " BOS: 'true' prepends the BOS token, 'false' leaves it to the tokenizer; default 'auto' = true for models whose tokenizer never adds the BOS token they need (Gemma 4, scripts/lm_eval_harness/bos_default.py)."
	echo " TRUST_REMOTE_CODE: 'true' loads the tokenizer with trust_remote_code=True."
	echo " LOGS_ROOT: Where the evaluation wandb&harness logs go (default: <repo>/logs/lm_eval)."
	echo " TASKS: Required. Tasks to run with lm eval harness: a task list file (one name per line, # comments) or comma-separated names."
	echo " TABLE_METRICS: task/metric pairs for the W&B summary table: a file (one pair per line, # group headers) or space-separated pairs (default: TASKS)."
	echo " WANDB_ENTITY: WandB entity name for uploading results (default: apertus)."
	echo " WANDB_PROJECT: WandB project name for uploading results (default: swissai-evals-test)."
    echo " API_BASE_URL: Endpoint to evaluate against. Required. Accepts a bare host (http://host:8000), a /v1 root, or a full /v1/(chat/)completions URL."
    echo " API_CHAT_ENDPOINT: 'true' to use /v1/chat/completions (server-side chat template; generative tasks ONLY). Default: /v1/completions, which serves mixed suites."
    echo " API_MODEL_NAME: 'model' field sent in requests (default: the <model> positional argument)."
    echo " API_NUM_CONCURRENT: Concurrent requests (default: 8)."
    echo " API_REQUESTS_PER_MINUTE: Optional endpoint-wide request limit for the benchmarked model."
    echo " API_MAX_RETRIES: Retries per request (default: 3)."
    echo " API_REQUEST_TIMEOUT: Seconds to wait for one request before retrying it (default: 3600; lm-eval's own 300 cuts off long reasoning)."
    echo " JUDGE_REQUESTS_PER_MINUTE: Optional endpoint-wide request limit, applied separately to each judge model."
    echo " JUDGE_MODEL_PREFIX: Preferred host scope for judge model IDs (default: USER)."
    echo " LM_EVAL_RATE_LIMIT_STATE_DIR: Shared coordination directory for rate-limited Slurm workers (set by launcher)."
    echo " OPENAI_API_KEY: Bearer token for the endpoint (default: CSCS_SERVING_API)."
    echo " HARNESS_LIMIT: Optional argument to pass as --limit to the lm-evaluation-harness, to limit the number of samples per task (default: no limit)."
    echo " LM_EVAL_HARNESS_BRANCH: Branch/ref resolved once by the environment-preparation job (default: the commit in requirements/lm-eval-harness.txt)."
    echo " EVAL_ENV_MANIFEST: Prepared base environment and harness overlay (required; set by launcher)."
    echo " EVAL_PREBUILT_ENV: '1' when running in the prebuilt image (containers/Dockerfile.eval): nothing is installed; the launcher writes the manifest."
    echo " EVAL_RUN_CONFIG: Normalized result provenance file used for safe resume (required; set by launcher)."
    echo " EVAL_VERIFY_ENV_IMPORTS: 'true' to repeat the preparation job's deep Python import checks inside the evaluation job (default: false)."
    echo " EVAL_CHUNKS_FILE: Resumable array chunk manifest (set automatically by launcher)."
    echo " Thinking / reasoning metrics:"
    echo " ENABLE_THINKING: 'true'/'false' chat-template argument deciding whether the model reasons (default: unset, the template's own default)."
    echo " AUTODETECT_THINK_TOKENS: 'true' to read the reasoning open/close tokens from the chat template (default: false)."
    echo " THINK_END_TOKEN: Force the reasoning close token, e.g. '</think>'. A known close token arms the strip AND the metrics."
    echo " THINK_START_TOKEN: Force the reasoning open token, e.g. '<think>'. Needed for thinking_format_has_open."
    echo " TRACK_THINKING_METRICS: 'true'/'false' to force the thinking metrics on/off (default: on iff a close token is known)."
    echo " LOG_LENGTH_METRICS: 'true' to aggregate response_length_*/thinking_length_* (thinking_format_* is always aggregated)."
    echo " REASONING_EFFORT: chat-template argument reasoning_effort (e.g. low/medium/high, read by gpt-oss's template)."
    echo " THINK_TEMPERATURE, THINK_TOP_P, THINK_TOP_K, THINK_MIN_P, THINK_REPETITION_PENALTY: sampling of thinking runs (defaults in configs/sampling_defaults.json: 0.6 / 0.95, the others unset; 'none' leaves one out)."
    echo " CHAT_FORMAT: 'apertus2' for the Apertus 2 native format (harness model apertus2-completions; needs TOKENIZER and MAX_LENGTH). Default: template."
    echo " PREFILL_MODE: apertus2 answer prefills, 'after_think' (default) or 'empty_think'."
    echo " ANSWER_FRAME: apertus2: score the 'reply' (default) or the 'claim', the frame the Apertus 2 RL math reward grades."
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
BS=${BS:-"auto:20"}
BOS=${BOS:-auto}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-false}
LOGS_ROOT=${LOGS_ROOT:-$PWD/logs/lm_eval}
TASKS=${TASKS:-}
TABLE_METRICS=${TABLE_METRICS:-$TASKS}
WANDB_ENTITY=${WANDB_ENTITY:-apertus}
WANDB_PROJECT=${WANDB_PROJECT:-swissai-evals-test}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-true}
# by-type: generative tasks with the chat template, log-likelihood ones without
# (two lm-eval passes, merged; see scripts/lm_eval_harness/task_output_types.py).
case "$APPLY_CHAT_TEMPLATE" in
    true|false|by-type) ;;
    *) echo "APPLY_CHAT_TEMPLATE must be true, false or by-type (got '$APPLY_CHAT_TEMPLATE')" >&2; exit 1 ;;
esac
HARNESS_LIMIT=${HARNESS_LIMIT:-""}
LM_EVAL_HARNESS_BRANCH=${LM_EVAL_HARNESS_BRANCH:-""}
EVAL_ENV_MANIFEST=${EVAL_ENV_MANIFEST:-""}
EVAL_CHUNKS_FILE=${EVAL_CHUNKS_FILE:-""}
EVAL_CHUNKED=${EVAL_CHUNKED:-false}
EVAL_VERIFY_ENV_IMPORTS=${EVAL_VERIFY_ENV_IMPORTS:-false}
EVAL_RUN_CONFIG=${EVAL_RUN_CONFIG:-""}

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
[[ -n "$TASKS" ]] || die "TASKS is not set: a task list file or comma-separated task names"

[[ -n "$EVAL_ENV_MANIFEST" ]] || die "EVAL_ENV_MANIFEST is required; launch through scripts/lm_eval_harness/launch_evaluations.sh"
[[ -f "$EVAL_RUN_CONFIG" ]] || die "EVAL_RUN_CONFIG is required and must be a readable file"

load_prepared_eval_env() {
    EVAL_ENV_CHECK_ERROR=""
    unset EVAL_ENV_KIND EVAL_BASE_ENV EVAL_HARNESS_OVERLAY EVAL_HARNESS_ARCHIVE \
        EVAL_RESOLVED_HARNESS_REPO EVAL_RESOLVED_HARNESS_COMMIT EVAL_RESOLVED_BASE_KEY
    if [[ ! -f "$EVAL_ENV_MANIFEST" ]]; then
        EVAL_ENV_CHECK_ERROR="manifest is missing: $EVAL_ENV_MANIFEST"
        return 1
    fi
    # The manifest is generated by scripts/lm_eval_harness/build_eval_env.sh and contains only
    # shell-escaped paths plus the resolved repository and commit.
    if ! source "$EVAL_ENV_MANIFEST"; then
        EVAL_ENV_CHECK_ERROR="manifest could not be sourced: $EVAL_ENV_MANIFEST"
        return 1
    fi
    if [[ "${EVAL_ENV_KIND:-}" == "prebuilt" ]]; then
        # Written by the launcher for EVAL_PREBUILT_ENV=1: the paths exist only
        # inside the container, which checks them itself (LOAD_ENV_CMD below).
        [[ -n "${EVAL_HARNESS_OVERLAY:-}" ]] \
            || { EVAL_ENV_CHECK_ERROR="manifest has no EVAL_HARNESS_OVERLAY"; return 1; }
        [[ -n "${EVAL_RESOLVED_HARNESS_REPO:-}" ]] \
            || { EVAL_ENV_CHECK_ERROR="manifest has no EVAL_RESOLVED_HARNESS_REPO"; return 1; }
        [[ "${EVAL_RESOLVED_HARNESS_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] \
            || { EVAL_ENV_CHECK_ERROR="manifest has no EVAL_RESOLVED_HARNESS_COMMIT"; return 1; }
        return 0
    fi
    # Do not test EVAL_BASE_ENV/bin/python with -x here. This shell runs outside
    # the container, while venv's python may be an absolute symlink to a path
    # that exists only inside it (for example /usr/local/bin/python). The
    # in-container preflight below performs the executable check.
    [[ -n "${EVAL_BASE_ENV:-}" ]] \
        || { EVAL_ENV_CHECK_ERROR="manifest has no EVAL_BASE_ENV"; return 1; }
    [[ -n "${EVAL_HARNESS_OVERLAY:-}" ]] \
        || { EVAL_ENV_CHECK_ERROR="manifest has no EVAL_HARNESS_OVERLAY"; return 1; }
    [[ -n "${EVAL_HARNESS_ARCHIVE:-}" ]] \
        || { EVAL_ENV_CHECK_ERROR="manifest has no EVAL_HARNESS_ARCHIVE"; return 1; }
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
    [[ -f "$EVAL_HARNESS_ARCHIVE" ]] \
        || { EVAL_ENV_CHECK_ERROR="harness archive is missing: $EVAL_HARNESS_ARCHIVE"; return 1; }
}

if ! load_prepared_eval_env; then
    # A prebuilt environment is never repaired by installing: that is the point of it.
    [[ "${EVAL_ENV_KIND:-}" != "prebuilt" ]] \
        || die "Prebuilt evaluation environment manifest is invalid: $EVAL_ENV_CHECK_ERROR"
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
        --export=ALL bash "$EVAL_REPO_ROOT/scripts/lm_eval_harness/build_eval_env.sh"
fi
if ! load_prepared_eval_env; then
    die "Evaluation environment repair did not produce a complete environment: $EVAL_ENV_CHECK_ERROR"
fi
# If a moving branch has advanced since preparation, a last-resort repair must
# still recreate this run's exact environment rather than silently changing it.
export EVAL_HARNESS_REF="$EVAL_RESOLVED_HARNESS_COMMIT"

# The OpenAI-compatible endpoint.
API_BASE_URL=${API_BASE_URL:-""}
API_MODEL_NAME=${API_MODEL_NAME:-""}
API_NUM_CONCURRENT=${API_NUM_CONCURRENT:-8}
API_REQUESTS_PER_MINUTE=${API_REQUESTS_PER_MINUTE:-""}
API_MAX_RETRIES=${API_MAX_RETRIES:-3}
# Requests are not streamed, so nothing comes back until the whole generation is done:
# lm-eval's default 300s abandons and retries long reasoning (32k tokens on a busy
# endpoint takes ~20-30 min). Only a ceiling -- a request returns as soon as it is done.
API_REQUEST_TIMEOUT=${API_REQUEST_TIMEOUT:-3600}
export JUDGE_REQUESTS_PER_MINUTE=${JUDGE_REQUESTS_PER_MINUTE:-""}
export JUDGE_MODEL_PREFIX=${JUDGE_MODEL_PREFIX:-${USER:-}}

# shellcheck source=scripts/common/secrets.sh
source scripts/common/secrets.sh
export HF_TOKEN="${HF_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY:-}" CSCS_SERVING_API="${CSCS_SERVING_API:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-$CSCS_SERVING_API}"

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

NUM_FEWSHOT=${NUM_FEWSHOT:-""}

if [ -f "$TASKS" ]; then
    echo "Reading task list from file: $TASKS"
    # Filter out comments and blank lines, then join with commas
    TASKS=$(grep -v '^\s*#' "$TASKS" | grep -v '^\s*$' | paste -sd, -)
fi

if [ -f "$TABLE_METRICS" ]; then
	echo "Reading table metrics from file: $TABLE_METRICS"
	TABLE_METRICS=$(grep -v '^\s*#' "$TABLE_METRICS" | grep -v '^\s*$' | paste -sd' ' -)
fi

# The job's settings are resolved and checked, and the harness model and its --model_args built, by
# scripts/lm_eval_harness/job_settings.py: it reads these settings from the environment (its INPUTS) and
# prints shell assignments. Captured first, so a refused setting stops the job: `eval "$(...)"` would hide
# the exit code.
export MODEL TOKENIZER BOS BOS_BY_CLIENT MAX_LENGTH MAX_NEW_TOKENS ENABLE_THINKING AUTODETECT_THINK_TOKENS THINK_START_TOKEN THINK_END_TOKEN TRACK_THINKING_METRICS LOG_LENGTH_METRICS REASONING_EFFORT CHAT_FORMAT PREFILL_MODE ANSWER_FRAME APPLY_CHAT_TEMPLATE API_BASE_URL API_MODEL_NAME API_CHAT_ENDPOINT API_NUM_CONCURRENT API_MAX_RETRIES API_REQUEST_TIMEOUT API_REQUESTS_PER_MINUTE TRUST_REMOTE_CODE TASKS NOTHINK_TEMPERATURE NOTHINK_TOP_P THINK_TEMPERATURE THINK_TOP_P THINK_TOP_K THINK_MIN_P THINK_REPETITION_PENALTY
SETTINGS_OUT=$(python3 -m scripts.lm_eval_harness.job_settings) || die "Could not resolve the job's settings (see above)."
eval "$SETTINGS_OUT"

# API models take an integer batch size; throughput comes from num_concurrent instead.
if [[ "$BS" == auto* ]]; then
    BS=1
    echo "batch size 'auto' unsupported, using BS=$BS (concurrency: API_NUM_CONCURRENT=$API_NUM_CONCURRENT)"
fi

# Redirect all caches to SCRATCH to avoid filling up HOME: HF_HOME and HF_DATASETS_CACHE come from
# scripts/common/cache_dirs.sh, shared with the Inspect and aggregation jobs.
# shellcheck source=scripts/common/cache_dirs.sh
source scripts/common/cache_dirs.sh
CACHE_ROOT="$EVAL_CACHE_ROOT"
export NLTK_DATA="$CACHE_ROOT/nltk_data"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TIKTOKEN_CACHE_DIR="$CACHE_ROOT/tiktoken"
export PYTHONUSERBASE="$CACHE_ROOT/python_userbase"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export SPACY_DATA="$CACHE_ROOT/spacy"
export XDG_CACHE_HOME="$CACHE_ROOT"
mkdir -p "$HF_DATASETS_CACHE" "$NLTK_DATA" "$TRITON_CACHE_DIR" "$TIKTOKEN_CACHE_DIR" "$PYTHONUSERBASE" "$MPLCONFIGDIR" "$SPACY_DATA"

# Print configuration.
echo "Configuration set:"
# Values go in printf's arguments, not the format string: a '%' would be read as a conversion.
printf '%s\n' "MODEL=$MODEL" "NAME=$NAME" "TOKENIZER=$TOKENIZER" "BS=$BS" \
	"HARNESS_LIMIT=$HARNESS_LIMIT" "LM_EVAL_HARNESS_BRANCH=$LM_EVAL_HARNESS_BRANCH" \
	"BOS=$BOS" "BOS_BY_CLIENT=${BOS_BY_CLIENT:-auto}" "TRUST_REMOTE_CODE=$TRUST_REMOTE_CODE" "TASKS=$TASKS" "MAX_LENGTH=$MAX_LENGTH" "WANDB_ENTITY=$WANDB_ENTITY" \
	"WANDB_PROJECT=$WANDB_PROJECT" "HF_HOME=$HF_HOME" \
	"API_CHAT_ENDPOINT=${API_CHAT_ENDPOINT:-false}" "CHAT_FORMAT=$CHAT_FORMAT" ""
printf '%s\n' "ENABLE_THINKING=$ENABLE_THINKING (explicit=$ENABLE_THINKING_EXPLICIT)" \
	"AUTODETECT_THINK_TOKENS=$AUTODETECT_THINK_TOKENS" "THINK_START_TOKEN=$THINK_START_TOKEN" \
	"THINK_END_TOKEN=$THINK_END_TOKEN" "TRACK_THINKING_METRICS=$TRACK_THINKING_METRICS" \
	"LOG_LENGTH_METRICS=$LOG_LENGTH_METRICS" "REASONING_EFFORT=$REASONING_EFFORT" ""

# BOS=auto is resolved for each lm-eval command (bos_pass.py, called by make_cmd): the rule depends on whether the
# chat template is applied, and by-type decides that per kind of task.
BOS_REQUESTED=$BOS
# BOS_BY_CLIENT: unset/auto = on for commands with a chat template on /v1/completions
# (--no-bos-by-client turns it off); true = --bos-by-client, refused where it cannot apply.

# Prepare logs root.
RUN_ROOT=$LOGS_ROOT/$WANDB_ENTITY/$WANDB_PROJECT/$NAME
HARNESS_DIR=$RUN_ROOT/harness
export WANDB_DIR=$RUN_ROOT
mkdir -p $HARNESS_DIR

# export HF_HUB_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
# export PERSPECTIVE_API_KEY="" # Uncomment if you want to use perspective API, needed for realtoxicityprompts task

# Finally, run the actual evaluations.
HARNESS_EVAL_DIR=$HARNESS_DIR/eval_$(date +%Y%m%d_%H%M%S)_$SLURM_JOBID
mkdir -p "$HARNESS_EVAL_DIR"
# Keep the job's output from here on with the results (stdout and stderr stay separate streams
# for Slurm). Nothing in the results pipeline reads it: the readers glob results_*.json and samples_*.jsonl.
exec > >(tee -a "$HARNESS_EVAL_DIR/run.log") 2> >(tee -a "$HARNESS_EVAL_DIR/run.log" >&2)
cp "$EVAL_RUN_CONFIG" "$HARNESS_EVAL_DIR/.eval-run-config.json"

# What job_settings.py has to say about the run (the raised budget, the tokenizer, the sampling): printed here so
# it is kept with the results. GEN_KWARGS (its --gen_kwargs) updates every task's YAML.
[[ -z "$SETTINGS_NOTES" ]] || printf '%s\n' "$SETTINGS_NOTES"

MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-""}

# --tasks, --output_path and the chat-template flags are per lm-eval pass: see make_cmd.
COMMON_EVAL_ARGS=(
	--trust_remote_code
	--batch_size $BS
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

# Aggregates response_length_*/thinking_length_* into results.json and W&B (thinking_format_*
# is aggregated whenever tracking is on, flag or not).
if [[ "$LOG_LENGTH_METRICS" == "True" ]]; then
	COMMON_EVAL_ARGS+=(--log_length_metrics)
fi

# A RULER task list names its context length (CONTEXT_LEN, from model_args.py): lm-eval gets it as metadata.
if [[ -n "$CONTEXT_LEN" ]]; then
    COMMON_EVAL_ARGS+=( --metadata \'{\"max_seq_lengths\":[${CONTEXT_LEN}]}\' )
fi

echo "Using lm-evaluation-harness: $EVAL_RESOLVED_HARNESS_REPO@$EVAL_RESOLVED_HARNESS_COMMIT"

if [[ "${EVAL_ENV_KIND:-}" == "prebuilt" ]]; then
# EVAL_PREBUILT_ENV=1: the container (containers/Dockerfile.eval) already has
# every package and the pinned harness. Check it, never install: a mismatch means the job runs
# in an image built from a different commit.
LOAD_ENV_CMD="
echo Container entered: \$(date -Is)
# lm-eval's packages are in the image's venv, first on PATH by the image's ENV. Not every runtime
# applies that (enroot/pyxis keep the base image's PATH, so python is the image's own, without them).
export PATH=\"\${LM_EVAL_VENV:-/opt/lm-eval-venv}/bin:\$PATH\"
if [[ ! -f '$EVAL_HARNESS_OVERLAY/.complete' || ! -d '$EVAL_HARNESS_OVERLAY/lm_eval' ]]; then
    echo 'EVAL_PREBUILT_ENV=1, but this container has no harness at $EVAL_HARNESS_OVERLAY. Use the image built from this commit (containers/Dockerfile.eval).' >&2
    exit 1
fi
container_harness_commit=\$(cat '$EVAL_HARNESS_OVERLAY/.commit' 2>/dev/null || true)
if [[ \"\$container_harness_commit\" != '$EVAL_RESOLVED_HARNESS_COMMIT' ]]; then
    echo \"The container's harness is $EVAL_RESOLVED_HARNESS_REPO@\$container_harness_commit, but this run's manifest pins $EVAL_RESOLVED_HARNESS_COMMIT. Use the image built from this commit.\" >&2
    exit 1
fi
export PYTHONPATH='$EVAL_HARNESS_OVERLAY'\${PYTHONPATH:+:\$PYTHONPATH}
if ! python -c 'import accelerate, datasets, transformers, wandb, lm_eval, apertus_common'; then
    echo 'EVAL_PREBUILT_ENV=1, but this container is missing evaluation packages (import check above).' >&2
    exit 1
fi
echo Evaluation environment activated: \$(date -Is) prebuilt=true harness_path='$EVAL_HARNESS_OVERLAY'
"
else
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
            && PYTHONPATH='$EVAL_HARNESS_OVERLAY' '$EVAL_BASE_ENV/bin/python' -c 'import lm_eval${EVAL_APERTUS2_REQUIREMENTS:+, apertus_common}' >/dev/null 2>&1 \
            || return 1
    fi
}
if ! eval_env_is_ready; then
    echo 'Cached evaluation environment failed its in-container preflight; rebuilding it now.'
    bash '$EVAL_REPO_ROOT/scripts/lm_eval_harness/build_eval_env.sh'
    source '$EVAL_ENV_MANIFEST'
fi
if ! eval_env_is_ready; then
    echo 'Evaluation environment repair did not pass its in-container preflight.' >&2
    exit 1
fi
source '$EVAL_BASE_ENV/bin/activate'
EVAL_RUNTIME_HARNESS_OVERLAY='$EVAL_HARNESS_OVERLAY'
runtime_root=\${SLURM_TMPDIR:-/tmp}
if [[ ! -d \"\$runtime_root\" || ! -w \"\$runtime_root\" ]]; then
    runtime_root=/tmp
fi
if [[ -r '$EVAL_HARNESS_ARCHIVE' ]]; then
    runtime_overlay=\$(mktemp -d \"\$runtime_root/lm-eval-${EVAL_RESOLVED_HARNESS_COMMIT:0:12}.XXXXXX\")
    echo Staging harness overlay locally: \$(date -Is) archive='$EVAL_HARNESS_ARCHIVE' destination=\$runtime_overlay
    if tar -xf '$EVAL_HARNESS_ARCHIVE' -C \"\$runtime_overlay\"; then
        EVAL_RUNTIME_HARNESS_OVERLAY=\$runtime_overlay
        echo Harness overlay staged locally: \$(date -Is)
    else
        echo 'WARNING: local harness staging failed; using the shared overlay.' >&2
        rm -rf \"\$runtime_overlay\"
    fi
fi
export PYTHONPATH=\$EVAL_RUNTIME_HARNESS_OVERLAY\${PYTHONPATH:+:\$PYTHONPATH}
echo Evaluation environment activated: \$(date -Is) deep_import_check=$EVAL_VERIFY_ENV_IMPORTS harness_path=\$EVAL_RUNTIME_HARNESS_OVERLAY
"
fi

# $CMD is re-parsed by the inner `bash -lc`, so --model_args is single-quoted (a token like
# think_end_token=</think> would otherwise redirect, and $(...) in a path would execute).
# A single quote in the value can't be nested safely, so refuse it.
if [[ "$COMMON_MODEL_ARGS" == *\'* ]]; then
    die "Model args must not contain a single quote: $COMMON_MODEL_ARGS"
fi

# The BOS settings of each lm-eval command (BOS=auto is resolved per command, whether the chat template applies
# decides the rule) come from scripts/lm_eval_harness/bos_pass.py: PASS_MODEL_ARGS to append to --model_args,
# PASS_PROBE (the BOS probe's command line, "" for none) and PASS_NOTES. The probe asks the server what it received
# and stops before the first task if that is not what was meant.
export BOS_REQUESTED API_LM_EVAL_MODEL BOS_CHECK

# append_bos_probe: the BOS probe of the command make_cmd just built, in TIMED_CMD.
append_bos_probe() {
    [[ -z "$PASS_PROBE" ]] || TIMED_CMD="$TIMED_CMD
$PASS_PROBE"
    return 0
}

# make_cmd <tasks> <output dir> <chat template: true|false>: one lm-eval pass, in CMD.
make_cmd() {
    local eval_args="${COMMON_EVAL_ARGS[*]} --tasks $1 --output_path $2"
    local pass_out
    pass_out=$(python3 -m scripts.lm_eval_harness.bos_pass --chat-template "$3") \
        || die "Could not resolve the BOS settings of this command (see above)."
    eval "$pass_out"
    [[ -z "$PASS_NOTES" ]] || printf '%s\n' "$PASS_NOTES"
    local COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS$PASS_MODEL_ARGS"
    local repeat_setup="" repeat_file
    if [[ -n "${EVAL_REPEAT_CONFIG:-}" ]]; then
        printf -v repeat_file '%q' "$2/repeat_config.json"
        # A subshell: scripts.* needs the repository as working directory, lm_eval does not.
        printf -v repeat_setup '(cd %q && python -m scripts.lm_eval_harness.repeat_config --policy %q --suite %q --tasks %q --output %q)\n' \
            "$PWD" "$EVAL_REPEAT_CONFIG" "${EVAL_REPEAT_TASKS:-}" "$1" "$2/repeat_config.json"
        eval_args="$eval_args --config $repeat_file"
    fi
    [[ $3 == true ]] && eval_args="$eval_args --apply_chat_template --fewshot_as_multiturn"
    CMD="python -m lm_eval --model $API_LM_EVAL_MODEL --model_args '$COMMON_MODEL_ARGS'$CHAT_TEMPLATE_ARGS $eval_args"
    CMD="$repeat_setup$CMD"
}

# EVAL_PREBUILT_CONTAINER_CONFIG (an EDF, written by the launcher for EVAL_PREBUILT_ENV=1) names the
# prebuilt image's, for clusters where the NGC image of env.toml does not start.
BASE_CONTAINER_ENV=${EVAL_PREBUILT_CONTAINER_CONFIG:-./containers/env.toml}

echo "Environment manifest: $EVAL_ENV_MANIFEST"
TIMED_CMD="echo Evaluation process starting: \$(date -Is)"
if [[ "$CHAT_FORMAT" == "apertus2" ]]; then
    # apertus2-completions only generates: stop before any request if a task is scored by
    # log-likelihood (it reads the harness's task index, so it runs in the container).
    TIMED_CMD="$TIMED_CMD
cd $PWD && python -m scripts.lm_eval_harness.task_output_types --tasks '$TASKS' --require-generative"
fi
if [[ $APPLY_CHAT_TEMPLATE != by-type ]]; then
    make_cmd "$TASKS" "$HARNESS_EVAL_DIR" "$APPLY_CHAT_TEMPLATE"
    append_bos_probe "$APPLY_CHAT_TEMPLATE"
    TIMED_CMD="$TIMED_CMD
$CMD"
    echo "Final command: $CMD"
else
    # Sort the tasks by how lm-eval scores them, in the evaluation container (it reads
    # the harness's task index), then run each kind as its own pass and merge the two
    # into $HARNESS_EVAL_DIR, so everything downstream sees one result as usual.
    KINDS_FILE="$HARNESS_EVAL_DIR/.task-kinds.sh"
    srun -n 1 --mpi=pmix -ul --environment=$BASE_CONTAINER_ENV bash -lc "
    set -euo pipefail
    unset SSL_CERT_FILE
    $LOAD_ENV_CMD
    cd $PWD && python -m scripts.lm_eval_harness.task_output_types --tasks '$TASKS' --output '$KINDS_FILE'
    " || die "Could not sort the tasks by output type (scripts/lm_eval_harness/task_output_types.py)."
    source "$KINDS_FILE"
    if [[ -z "$RAW_TASKS" || -z "$CHAT_TASKS" ]]; then
        # One kind only: a single pass straight into the eval directory.
        if [[ -n "$RAW_TASKS" ]]; then
            make_cmd "$RAW_TASKS" "$HARNESS_EVAL_DIR" false
            append_bos_probe false
        else
            make_cmd "$CHAT_TASKS" "$HARNESS_EVAL_DIR" true
            append_bos_probe true
        fi
        TIMED_CMD="$TIMED_CMD
$CMD"
        echo "Final command: $CMD"
    else
        # Outside eval_* so the result scan never takes a pass for a whole evaluation.
        PASS_ROOT="$HARNESS_DIR/.passes/$(basename "$HARNESS_EVAL_DIR")"
        make_cmd "$CHAT_TASKS" "$PASS_ROOT/chat" true
        append_bos_probe true
        TIMED_CMD="$TIMED_CMD
$CMD"
        echo "Final command (chat template): $CMD"
        make_cmd "$RAW_TASKS" "$PASS_ROOT/raw" false
        append_bos_probe false
        TIMED_CMD="$TIMED_CMD
$CMD
cd $PWD && python -m scripts.results.merge_split_results --split_dirs '$PASS_ROOT/chat' '$PASS_ROOT/raw' --output_dir '$HARNESS_EVAL_DIR'"
        echo "Final command (no chat template): $CMD"
    fi
fi

# Environment preparation happens once in a dependency job. This duration is
# therefore evaluation-only and remains meaningful when results are aggregated.
START_TIME=$(date +%s)
echo "Starting evaluation timing: $(date -Is)"

# Cache redirection exports to pass into the container. lm-eval keys its request cache by
# task, few-shot, template flag and tokenizer only, not by the generation or template
# arguments, so that one stays in the job.
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
cache_root=\${SLURM_TMPDIR:-/tmp}
[[ -d \"\$cache_root\" && -w \"\$cache_root\" ]] || cache_root=/tmp
LM_HARNESS_CACHE_PATH=\$(mktemp -d \"\$cache_root/lm-eval-request-cache.XXXXXX\")
export LM_HARNESS_CACHE_PATH
mkdir -p $HF_DATASETS_CACHE $NLTK_DATA $TRITON_CACHE_DIR $TIKTOKEN_CACHE_DIR $PYTHONUSERBASE $MPLCONFIGDIR $SPACY_DATA
"

# No model is loaded in-job; the harness only makes HTTP calls against API_BASE_URL.
srun -n 1 --mpi=pmix -ul --environment=$BASE_CONTAINER_ENV --export=ALL,HF_TOKEN,HF_HOME,WANDB_API_KEY,OPENAI_API_KEY bash -lc "
  set -euo pipefail
  unset SSL_CERT_FILE
  export HF_DATASETS_TRUST_REMOTE_CODE=1
  $CACHE_EXPORTS
  # The locked environment's torch must not load the container's torch libraries.
  export LD_LIBRARY_PATH=\$(tr ':' '\n' <<< \"\${LD_LIBRARY_PATH:-}\" | { grep -vx '.*/torch/lib' || true; } | paste -sd: -)

  $LOAD_ENV_CMD

  $TIMED_CMD
  "
# End timing and calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
printf '%s\n' "$DURATION" > "$HARNESS_EVAL_DIR/eval_duration_seconds.txt"
echo "Evaluation time: ${DURATION} seconds ($(($DURATION / 60)) minutes and $(($DURATION % 60)) seconds)"

# Goodbye.
echo "Evaluation finished"

if [[ "$EVAL_CHUNKED" == "true" ]]; then
    echo "Chunk job done. Skipping W&B upload; the controller will aggregate completed chunks."
    echo "Results in: $HARNESS_EVAL_DIR"
elif [[ "${WANDB_MODE:-}" == disabled || -z "${WANDB_API_KEY:-}" ]]; then
    echo "Skipping W&B upload ($([[ "${WANDB_MODE:-}" == disabled ]] && echo "WANDB_MODE=disabled" || echo "no WANDB_API_KEY")). Results in: $HARNESS_EVAL_DIR"
else
    echo "Uploading results to wandb"
    WANDB_CMD="cd $PWD && python -m scripts.results.update_wandb_alignment --entity $WANDB_ENTITY --project $WANDB_PROJECT --logs_root $HARNESS_EVAL_DIR --name $NAME --main_metrics $TABLE_METRICS --eval_duration $DURATION"
    echo "Running command to upload results to wandb:"
    echo "$WANDB_CMD"
    # Run in the evaluation environment, like the evaluation itself: the container's own python has no wandb.
    srun -ul --mpi=pmix --environment=$BASE_CONTAINER_ENV bash -c " \
        set -euo pipefail
        $LOAD_ENV_CMD
        $WANDB_CMD
    "
fi
echo "END TIME: $(date)"
