#!/bin/bash

# launch_evaluations.sh - Launch Apertus benchmark suite evaluations
#
# Usage:
#   bash scripts/lm_eval_harness/launch_evaluations.sh [mode] [options]
#
# Modes (required):
#   single           - Run a single task (requires --task <task_name>)
#   custom           - Run a task list (requires --task-file <path>, optional --table-metrics <path>)

#
# Model:
#   --model <name>            - The model's name (names the run and, unless --tokenizer is given,
#                               its tokenizer). Optional: --api-model-name fills its slot.
#
# Options:
#   --name <name>        - Override the eval run name (default: auto-derived from model path)
#   --task <task>         - Task name for 'single' mode (e.g. hellaswag, gsm8k_cot)
#   --wandb-project <name> - Override the W&B project name (default: e-p-t-pool)
#   --no-wandb           - Don't upload the results to W&B (exports WANDB_MODE=disabled)
#   --chat-template      - Apply the chat template to every task (the default)
#   --no-chat-template   - Apply it to none (base models)
#   --chat-template-by-type - Generative tasks with the chat template, log-likelihood ones
#                          (multiple_choice/loglikelihood output_type) without it: two lm-eval
#                          passes per job, merged into one result (scripts/lm_eval_harness/task_output_types.py)
#   --tokenizer <tok>    - Custom tokenizer (default: same as model)
#   --trust-remote-code  - Load the tokenizer with trust_remote_code=True: for models whose
#                          tokenizer is custom code in the repo (e.g. Kimi-K2's
#                          tokenization_kimi.py).
#   --bos                - Prepend BOS token (default: Gemma 4, and without a chat template
#                          every tokenizer that defines a BOS token; see bos_default.py)
#   --no-bos             - Never prepend it beyond what the tokenizer does (turns that default off)
#   --bos-by-client      - On by default for runs with a chat template on /v1/completions: the
#                          server is told to add no special tokens to the rendered template, so the model gets exactly
#                          the BOS the template wrote (one: the server's own BOS would make two after a template that
#                          starts with <s>). Ignored without a chat template. Needs a harness with bos_by_client; the
#                          BOS probe verifies it on the server. Given explicitly it is refused where it cannot apply
#                          (chat-completions endpoint, --chat-format apertus2).
#   --no-bos-by-client   - The server adds its own BOS to the template as before (the previous behaviour)
#   --repeat-config <json> - Repeat policy: repeats, repeat_scoring, task_overrides.
#   --repeats N          - Sample every repeat-capable task N times and report mean@N. Overrides the
#                          repeat config's global `repeats`; its per-task overrides stay.
#   --repeat-metric pass@K[,K..] - Also report pass@K (K <= N) on every task with a binary metric.
#                          Needs sampling: greedy repeats are identical, so --repeats N>1 is refused.
#   --num-fewshot N      - Override num_fewshot for all tasks (default: use task YAML defaults)
#                          Note: tasks with num_fewshot=0 in YAML are never overridden.
#                          OLMo3 uses 5-shot for most MC tasks; pass --num-fewshot 5 to match.
#   --max-length N       - Context limit in tokens (exports MAX_LENGTH: the served window).
#                          Default: the model's own. loglikelihood_rolling tasks (e.g.
#                          uncheatable_eval) cut documents into windows of this size.
#   --api-base-url <url> - OpenAI-compatible endpoint to evaluate (required, unless API_BASE_URL
#                          is exported). Bare host, /v1 root, or full endpoint URL.
#   --api-model-name <n> - 'model' field sent in API requests (default: the --model value)
#   --api-requests-per-minute N - Endpoint-wide request limit for the benchmarked API model
#   --api-max-retries N  - retries per failed request (default 3; a rate-limited
#                          endpoint can exhaust 3 and fail the run)
#   --api-num-concurrent N - concurrent requests (default 8)
#   --api-request-timeout S - seconds to wait for one request before retrying it (default 3600).
#                          A long reasoning generation that outlasts it is cut and restarted from scratch; once the
#                          retries are used up the run fails and its results are lost
#   --api-endpoint <e>   - 'completions' (default) or 'chat-completions'.
#                          chat-completions lets the server render the chat template (generative
#                          tasks only; required by tasks that send `tools` in the request).
#   --chat-format <f>    - 'template' (default) or 'apertus2' (Apertus 2 native format, a thinking
#                          run; needs --tokenizer, --max-length; see the docs)
#   --prefill-mode <m>   - apertus2 answer prefills: 'after_think' (default) or 'empty_think'
#   --answer-frame <f>   - apertus2: score the 'reply' (default) or the 'claim', the frame the
#                          Apertus 2 RL math reward grades (needs a harness with answer_frame)
#   --temperature V, --top-p V, --top-k V, --min-p V, --repetition-penalty V
#                        - Sampling of thinking/apertus2 runs (defaults: configs/sampling_defaults.json,
#                          0.6 / 0.95); 'none' leaves one out. On other runs, --temperature turns sampling on.
#   --sampling-profile P - A profile of the sampling config (e.g. thinking, greedy)
#   --sampling-config F  - A sampling config file instead of configs/sampling_defaults.json
#   --chunk-size N       - Tasks per resumable job chunk (default: 8)
#   --max-parallel N     - Cap concurrent chunks (default: all generated chunks)
#   --max-retries N      - Retry waves for missing tasks; chunks halve each wave (default: 1)
#   --failure-policy P   - resume (default) or fail-fast
#   --task-file <path>   - Task list for custom mode
#   --table-metrics <p>  - Main-metrics list for custom mode
#   --force-tasks <csv>  - Re-run completed tasks matching these substrings
#   --limit N            - Optional argument to pass as --limit to the lm-evaluation-harness, to limit the number of samples per task (default: no limit).
#   --harness-branch B   - Install lm-evaluation-harness from branch/ref B (default: the commit in requirements/lm-eval-harness.txt).
#                          In the prebuilt image only the harness is installed, --no-deps, over the image's packages
#   --harness-repo O/R   - Use the fork github.com/O/R of lm-evaluation-harness (default: swiss-ai/lm-evaluation-harness)
#   --reservation <name> - Submit jobs under a SLURM reservation, including an auto-launched judge
#                          (exported as SBATCH_RESERVATION; ambient SBATCH_RESERVATION is respected
#                          for evaluation jobs when the flag is absent)
#   --qos <name>         - Submit every job, including an auto-launched judge, under this SLURM QOS
#                          (exported as SBATCH_QOS), e.g. highprio
#   --partition <name>   - Submit every job, including an auto-launched judge, to this SLURM
#                          partition (exported as SBATCH_PARTITION; default: normal), e.g. highprio
#   --judge <none|auto|preset> - Judge model control:
#                          none (default): disable judge auto-launch
#                          auto: detect judge-dependent tasks and launch needed judges
#                          <preset>: launch a specific preset (qwen3.5-27b, llama-3.3-70b)
#   --judge-args <str>   - Extra arguments forwarded to scripts/lm_eval_harness/launch_judge.py
#   --judge-requests-per-minute N - Per-judge-model endpoint-wide request limit
#   --keep-judge         - Do not auto-cancel judge model after evaluation finishes
#
# Thinking / reasoning metrics (the chat template
# is rendered client-side and the reasoning stripped from the API response, and there
# --enable-thinking on its own only switches the template):
#   --thinking           - Umbrella flag: make the model reason AND record the thinking metrics.
#                          Implies --enable-thinking, --autodetect-think-tokens (unless
#                          --think-end-token is given), --track-thinking-metrics true,
#                          --log-length-metrics, and forces the chat template on.
#   --enable-thinking    - Chat-template argument: let the model reason. On its own this records
#   --no-enable-thinking   NOTHING - a reasoning close token must also be known.
#   --think-end-token <s>  - Force the reasoning close token, e.g. '</think>'. Arms the trace
#                            strip and the thinking metrics.
#   --think-start-token <s> - Force the reasoning open token, e.g. '<think>'. Needed for
#                             thinking_format_has_open; without it thinking_format_correct
#                             degrades to == thinking_format_has_close.
#   --autodetect-think-tokens - Read the open/close tokens from the model's chat template.
#   --track-thinking-metrics <true|false>  - Force the thinking metrics on/off.
#   --no-track-thinking-metrics              Default: on iff a close token is known.
#   --log-length-metrics - Aggregate response_length_* / thinking_length_* into results and W&B.
#                          thinking_format_* is aggregated regardless.
#   --reasoning-effort <level> - Chat-template argument reasoning_effort (e.g. low, medium, high)
#                          for models whose template reads it (gpt-oss). Forces the chat template on. Letters, digits, '_' and '-'.
#                          Other templates ignore it. For gpt-oss, combine with --thinking and
#                          --think-start-token '<|channel|>analysis<|message|>'
#                          --think-end-token '<|channel|>final<|message|>' (not auto-detected).
#
# Examples:
#   # Single HF model, auto-detect everything
#   bash scripts/lm_eval_harness/launch_evaluations.sh single --task gsm8k_cot --model meta-llama/Llama-3.1-8B-Instruct
#
#   # Single model in resumable chunks
#   bash scripts/lm_eval_harness/launch_evaluations.sh custom --task-file tasks.txt --table-metrics tasks_main_table.txt --model allenai/OLMo-2-1124-7B --chunk-size 8
#
#   # Base model, explicit no chat template
#   bash scripts/lm_eval_harness/launch_evaluations.sh custom --task-file tasks.txt --table-metrics tasks_main_table.txt --model Qwen/Qwen2.5-7B --no-chat-template
#
#   # Run a single task
#   bash scripts/lm_eval_harness/launch_evaluations.sh single --task hellaswag --model meta-llama/Llama-3.1-8B-Instruct
#

set -euo pipefail
mkdir -p logs

# Keys from the environment, else scripts/secrets.env. The judge launcher (launch_judge.py) reads
# CSCS_SERVING_API; the sbatch jobs read scripts/secrets.env themselves.
# shellcheck source=scripts/common/secrets.sh
source scripts/common/secrets.sh
[[ -n "${CSCS_SERVING_API:-}" ]] && export CSCS_SERVING_API

# --- Argument parsing ---
EVAL_MODE=""
if (( $# > 0 )) && [[ "$1" != --* ]]; then
    EVAL_MODE=$1
    shift
fi

EVAL_CHUNK_SIZE=8
EVAL_MAX_PARALLEL=""
EVAL_MAX_RETRIES=1
EVAL_FAILURE_POLICY="resume"
MODEL_PATH=""
MODEL_NAME=""
CHAT_TEMPLATE_OVERRIDE=""  # "", "true", "false", "by-type"
CUSTOM_TOKENIZER=""
BOS_FLAG=""
BOS_BY_CLIENT_FLAG=""
TRUST_REMOTE_CODE_FLAG=""
API_BASE_URL_FLAG=""
API_MODEL_NAME_FLAG=""
API_REQUESTS_PER_MINUTE_FLAG=""
API_MAX_RETRIES_FLAG=""
API_NUM_CONCURRENT_FLAG=""
API_REQUEST_TIMEOUT_FLAG=""
API_ENDPOINT_FLAG=""
CHAT_FORMAT_FLAG="${CHAT_FORMAT:-}"
PREFILL_MODE_FLAG=""
ANSWER_FRAME_FLAG=""
FEWSHOT_FLAG=""
# Keep an ambient HARNESS_LIMIT (the graceful launcher has no --limit flag, so callers export
# it); blanking it here would ship the cleared value into the job. --limit still overrides.
HARNESS_LIMIT="${HARNESS_LIMIT:-}"
# Same for an ambient MAX_LENGTH; --max-length overrides it.
MAX_LENGTH="${MAX_LENGTH:-}"
SINGLE_TASK=""
HARNESS_BRANCH=""
HARNESS_REPO=""
RESERVATION_FLAG=""
QOS_FLAG=""
PARTITION_FLAG=""
JUDGE_MODE="none"       # auto, none, or a preset name
JUDGE_EXTRA_ARGS=""
JUDGE_REQUESTS_PER_MINUTE_FLAG=""
KEEP_JUDGE="false"
THINKING_UMBRELLA=""
ENABLE_THINKING_OVERRIDE=""   # "", "true", "false"
THINK_END_TOKEN=""
THINK_START_TOKEN=""
AUTODETECT_THINK_TOKENS=""
TRACK_THINKING_METRICS=""     # "", "true", "false"
REASONING_EFFORT=""
LOG_LENGTH_METRICS=""
SAMPLING_TEMPERATURE_FLAG=""; SAMPLING_TOP_P_FLAG=""; SAMPLING_TOP_K_FLAG=""
SAMPLING_MIN_P_FLAG=""; SAMPLING_REPETITION_PENALTY_FLAG=""
SAMPLING_PROFILE_FLAG=""; SAMPLING_CONFIG_FLAG=""
REPEATS_FLAG=""; REPEAT_METRIC_FLAG=""
TASK_FILE_OVERRIDE=""
TABLE_METRICS_OVERRIDE=""
LOGS_ROOT_FLAG=""
WANDB_ENTITY_FLAG=""
WANDB_PROJECT_FLAG=""
SBATCH_ACCOUNT_FLAG=""
FORCE_TASKS=""
EVAL_MERGE_ONLY="false"
EVAL_DRY_RUN="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL_PATH="$2";              shift 2 ;;
        --name)         MODEL_NAME="$2";              shift 2 ;;
        --chunk-size)   EVAL_CHUNK_SIZE="$2";         shift 2 ;;
        --max-parallel) EVAL_MAX_PARALLEL="$2";       shift 2 ;;
        --max-retries)  EVAL_MAX_RETRIES="$2";        shift 2 ;;
        --failure-policy) EVAL_FAILURE_POLICY="$2";   shift 2 ;;
        --num-fewshot)  FEWSHOT_FLAG="$2";            shift 2 ;;
        --task)         SINGLE_TASK="$2";             shift 2 ;;
        --chat-template)    CHAT_TEMPLATE_OVERRIDE="true";  shift ;;
        --no-chat-template) CHAT_TEMPLATE_OVERRIDE="false"; shift ;;
        --chat-template-by-type) CHAT_TEMPLATE_OVERRIDE="by-type"; shift ;;
        --tokenizer)    CUSTOM_TOKENIZER="$2";        shift 2 ;;
        --bos)          BOS_FLAG="true";              shift ;;
        --no-bos)       BOS_FLAG="false";             shift ;;
        --bos-by-client) BOS_BY_CLIENT_FLAG="true";   shift ;;
        --no-bos-by-client) BOS_BY_CLIENT_FLAG="false"; shift ;;
        --trust-remote-code) TRUST_REMOTE_CODE_FLAG="true"; shift ;;
        --api-base-url) API_BASE_URL_FLAG="$2";       shift 2 ;;
        --api-model-name) API_MODEL_NAME_FLAG="$2";   shift 2 ;;
        --api-requests-per-minute) API_REQUESTS_PER_MINUTE_FLAG="$2"; shift 2 ;;
        --api-max-retries) API_MAX_RETRIES_FLAG="$2"; shift 2 ;;
        --api-num-concurrent) API_NUM_CONCURRENT_FLAG="$2"; shift 2 ;;
        --api-request-timeout) API_REQUEST_TIMEOUT_FLAG="$2"; shift 2 ;;
        --api-endpoint) API_ENDPOINT_FLAG="$2";       shift 2 ;;
        --chat-format)  CHAT_FORMAT_FLAG="$2";        shift 2 ;;
        --prefill-mode) PREFILL_MODE_FLAG="$2";       shift 2 ;;
        --answer-frame) ANSWER_FRAME_FLAG="$2";       shift 2 ;;
        --limit) HARNESS_LIMIT="$2";            shift 2 ;;
        --max-length) MAX_LENGTH="$2";          shift 2 ;;
        --harness-branch) HARNESS_BRANCH="$2";        shift 2 ;;
        --harness-repo)   HARNESS_REPO="$2";          shift 2 ;;
        --reservation)   RESERVATION_FLAG="$2";        shift 2 ;;
        --qos)           QOS_FLAG="$2";                shift 2 ;;
        --partition)     PARTITION_FLAG="$2";          shift 2 ;;
        --judge)         JUDGE_MODE="$2";              shift 2 ;;
        --judge-args)    JUDGE_EXTRA_ARGS="$2";        shift 2 ;;
        --judge-requests-per-minute) JUDGE_REQUESTS_PER_MINUTE_FLAG="$2"; shift 2 ;;
        --keep-judge)    KEEP_JUDGE="true";            shift ;;
        --task-file)     TASK_FILE_OVERRIDE="$2";       shift 2 ;;
        --table-metrics) TABLE_METRICS_OVERRIDE="$2";   shift 2 ;;
        --logs-root)     LOGS_ROOT_FLAG="$2";           shift 2 ;;
        --wandb-entity)  WANDB_ENTITY_FLAG="$2";        shift 2 ;;
        --wandb-project) WANDB_PROJECT_FLAG="$2";       shift 2 ;;
        --no-wandb)      export WANDB_MODE=disabled;    shift ;;
        --account)       SBATCH_ACCOUNT_FLAG="$2";      shift 2 ;;
        --repeat-config) export EVAL_REPEAT_CONFIG="$2"; shift 2 ;;
        --repeats)       REPEATS_FLAG="$2"; shift 2 ;;
        --repeat-metric) REPEAT_METRIC_FLAG="$2"; shift 2 ;;
        --force-tasks)   FORCE_TASKS="$2";              shift 2 ;;
        --merge-only)    EVAL_MERGE_ONLY="true";        shift ;;
        --debug)         EVAL_DRY_RUN="true";           shift ;;
        --thinking)                  THINKING_UMBRELLA="true";          shift ;;
        --enable-thinking)           ENABLE_THINKING_OVERRIDE="true";   shift ;;
        --no-enable-thinking)        ENABLE_THINKING_OVERRIDE="false";  shift ;;
        --think-end-token)           THINK_END_TOKEN="$2";              shift 2 ;;
        --think-start-token)         THINK_START_TOKEN="$2";            shift 2 ;;
        --autodetect-think-tokens)   AUTODETECT_THINK_TOKENS="true";    shift ;;
        --track-thinking-metrics)    TRACK_THINKING_METRICS="$2";       shift 2 ;;
        --no-track-thinking-metrics) TRACK_THINKING_METRICS="false";    shift ;;
        --reasoning-effort)          REASONING_EFFORT="$2";             shift 2 ;;
        --temperature)               SAMPLING_TEMPERATURE_FLAG="$2";        shift 2 ;;
        --top-p)                     SAMPLING_TOP_P_FLAG="$2";              shift 2 ;;
        --top-k)                     SAMPLING_TOP_K_FLAG="$2";              shift 2 ;;
        --min-p)                     SAMPLING_MIN_P_FLAG="$2";              shift 2 ;;
        --repetition-penalty)        SAMPLING_REPETITION_PENALTY_FLAG="$2"; shift 2 ;;
        --sampling-profile)          SAMPLING_PROFILE_FLAG="$2";            shift 2 ;;
        --sampling-config)           SAMPLING_CONFIG_FLAG="$2";             shift 2 ;;
        --log-length-metrics)        LOG_LENGTH_METRICS="true";         shift ;;
        *)
            echo "Error: Unknown option '$1'"
            echo "Run with no arguments for usage."
            exit 1
            ;;
    esac
done

# --- Validate mode ---
VALID_MODES=("single" "custom")
if [[ ! " ${VALID_MODES[*]} " =~ " ${EVAL_MODE} " ]]; then
    echo "Error: ${EVAL_MODE:+invalid mode '$EVAL_MODE'}${EVAL_MODE:+}$([[ -z "$EVAL_MODE" ]] && echo 'a mode is required')"
    echo "Valid modes: ${VALID_MODES[*]}"
    exit 1
fi

# --- Validate single mode ---
if [[ "$EVAL_MODE" == "single" ]]; then
    if [[ -z "$SINGLE_TASK" ]]; then
        echo "Error: 'single' mode requires --task <task_name>"
        echo "Example: bash launch_evaluations.sh single --task hellaswag --model meta-llama/Llama-3.1-8B"
        exit 1
    fi
elif [[ -n "$SINGLE_TASK" ]]; then
    echo "Error: --task can only be used with 'single' mode"
    exit 1
fi

# --- Validate run name ---
# The name is part of the run's output path, and run_lm_eval.sbatch builds its
# lm-eval command as one string run through `bash -lc`, so whitespace would
# split --output_path (lm-eval: unrecognized arguments) only once the run starts.
if [[ "$MODEL_NAME" =~ [[:space:]] ]]; then
    echo "Error: --name '$MODEL_NAME' must not contain whitespace; use - or _ instead"
    exit 1
fi

if [[ ! "$EVAL_CHUNK_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --chunk-size must be >= 1"
    exit 1
fi
if [[ -n "$EVAL_MAX_PARALLEL" && ! "$EVAL_MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --max-parallel must be >= 1"
    exit 1
fi
if [[ ! "$EVAL_MAX_RETRIES" =~ ^[0-9]+$ ]]; then
    echo "Error: --max-retries must be >= 0"
    exit 1
fi
if [[ "$EVAL_FAILURE_POLICY" != "resume" && "$EVAL_FAILURE_POLICY" != "fail-fast" ]]; then
    echo "Error: --failure-policy must be 'resume' or 'fail-fast'"
    exit 1
fi

# --- Resolve the flags together ---
# What the thinking, repeats, chat-format, effort, endpoint and sampling flags mean together is resolved and checked by
# scripts/lm_eval_harness/launch_args.py: it reads the flags (handed over for this one command, so they do not reach the
# jobs) and the ambient variables they fall back to, and prints the shell statements that set what is used below and
# export what the jobs read. Captured first, so a refused combination stops the launcher: `eval "$(...)"` would hide
# the exit code. It must run before dispatch so the forced chat template reaches the run.
LAUNCH_OUT=$(THINKING_UMBRELLA="$THINKING_UMBRELLA" ENABLE_THINKING_OVERRIDE="$ENABLE_THINKING_OVERRIDE" THINK_END_TOKEN="$THINK_END_TOKEN" THINK_START_TOKEN="$THINK_START_TOKEN" AUTODETECT_THINK_TOKENS="$AUTODETECT_THINK_TOKENS" TRACK_THINKING_METRICS="$TRACK_THINKING_METRICS" LOG_LENGTH_METRICS="$LOG_LENGTH_METRICS" REASONING_EFFORT="$REASONING_EFFORT" REPEATS_FLAG="$REPEATS_FLAG" REPEAT_METRIC_FLAG="$REPEAT_METRIC_FLAG" CHAT_FORMAT_FLAG="$CHAT_FORMAT_FLAG" PREFILL_MODE_FLAG="$PREFILL_MODE_FLAG" ANSWER_FRAME_FLAG="$ANSWER_FRAME_FLAG" CUSTOM_TOKENIZER="$CUSTOM_TOKENIZER" MAX_LENGTH="$MAX_LENGTH" BOS_FLAG="$BOS_FLAG" CHAT_TEMPLATE_OVERRIDE="$CHAT_TEMPLATE_OVERRIDE" API_ENDPOINT_FLAG="$API_ENDPOINT_FLAG" API_BASE_URL_FLAG="$API_BASE_URL_FLAG" API_MODEL_NAME_FLAG="$API_MODEL_NAME_FLAG" API_MAX_RETRIES_FLAG="$API_MAX_RETRIES_FLAG" API_NUM_CONCURRENT_FLAG="$API_NUM_CONCURRENT_FLAG" API_REQUEST_TIMEOUT_FLAG="$API_REQUEST_TIMEOUT_FLAG" API_REQUESTS_PER_MINUTE_FLAG="$API_REQUESTS_PER_MINUTE_FLAG" JUDGE_REQUESTS_PER_MINUTE_FLAG="$JUDGE_REQUESTS_PER_MINUTE_FLAG" SAMPLING_TEMPERATURE_FLAG="$SAMPLING_TEMPERATURE_FLAG" SAMPLING_TOP_P_FLAG="$SAMPLING_TOP_P_FLAG" SAMPLING_TOP_K_FLAG="$SAMPLING_TOP_K_FLAG" SAMPLING_MIN_P_FLAG="$SAMPLING_MIN_P_FLAG" SAMPLING_REPETITION_PENALTY_FLAG="$SAMPLING_REPETITION_PENALTY_FLAG" SAMPLING_PROFILE_FLAG="$SAMPLING_PROFILE_FLAG" SAMPLING_CONFIG_FLAG="$SAMPLING_CONFIG_FLAG" TRUST_REMOTE_CODE_FLAG="$TRUST_REMOTE_CODE_FLAG" MODEL_PATH="$MODEL_PATH" \
    python3 -m scripts.lm_eval_harness.launch_args) || exit 1
eval "$LAUNCH_OUT"

# --- Environment defaults ---
# sbatch reads SBATCH_RESERVATION, SBATCH_QOS and SBATCH_PARTITION natively
# (CLI > env > script directives).
[[ -n "$RESERVATION_FLAG" ]] && export SBATCH_RESERVATION="$RESERVATION_FLAG"
[[ -n "$QOS_FLAG" ]] && export SBATCH_QOS="$QOS_FLAG"
[[ -n "$PARTITION_FLAG" ]] && export SBATCH_PARTITION="$PARTITION_FLAG"
export WANDB_ENTITY=${WANDB_ENTITY:-apertus}
export WANDB_PROJECT=${WANDB_PROJECT:-e-p-t-pool}
export LOGS_ROOT=${LOGS_ROOT:-$PWD/logs/lm_eval}
[[ -n "$WANDB_ENTITY_FLAG" ]] && export WANDB_ENTITY="$WANDB_ENTITY_FLAG"
[[ -n "$WANDB_PROJECT_FLAG" ]] && export WANDB_PROJECT="$WANDB_PROJECT_FLAG"
[[ -n "$LOGS_ROOT_FLAG" ]] && export LOGS_ROOT="$LOGS_ROOT_FLAG"
[[ -n "$SBATCH_ACCOUNT_FLAG" ]] && export SBATCH_ACCOUNT="$SBATCH_ACCOUNT_FLAG"
export EVAL_CHUNK_SIZE EVAL_MAX_PARALLEL EVAL_MAX_RETRIES EVAL_FAILURE_POLICY
export EVAL_FORCE_TASKS="$FORCE_TASKS" EVAL_MERGE_ONLY EVAL_DRY_RUN KEEP_JUDGE
export JUDGE_MODE JUDGE_EXTRA_ARGS
export SBATCH_SCRIPT="${SBATCH_SCRIPT:-scripts/lm_eval_harness/run_lm_eval.sbatch}"

# --- Configure task suite ---
case "$EVAL_MODE" in
    "single")
        export TASKS="$SINGLE_TASK"
        export TABLE_METRICS="$SINGLE_TASK"
        ;;
    "custom")
        [[ -n "$TASK_FILE_OVERRIDE" ]] && export TASKS="$TASK_FILE_OVERRIDE"
        [[ -n "$TABLE_METRICS_OVERRIDE" ]] && export TABLE_METRICS="$TABLE_METRICS_OVERRIDE"
        if [[ -z "${TASKS:-}" ]]; then
            echo "Error: custom mode requires --task-file <path> (or exported TASKS)"
            exit 1
        fi
        export TABLE_METRICS=${TABLE_METRICS:-$TASKS}
        ;;
esac

# --- Auto-derive name and chat template for --model mode ---
auto_detect_chat_template() {
    local model="$1"
    # Check for common instruct/chat model name patterns
    if [[ "$model" =~ -[Ii]nstruct ]] || \
       [[ "$model" =~ -[Cc]hat ]] || \
       [[ "$model" =~ -[Ss][Ff][Tt] ]] || \
       [[ "$model" =~ -[Dd][Pp][Oo] ]] || \
       [[ "$model" =~ -[Ii]t$ ]] || \
       [[ "$model" =~ -aligned ]]; then
        echo "true"
    else
        echo "false"
    fi
}

auto_derive_name() {
    local model="$1"
    # For HF paths like "meta-llama/Llama-3.1-8B-Instruct" -> "Llama-3.1-8B-Instruct"
    # For local paths like "/capstor/.../checkpoint-8926" -> last meaningful dir component
    if [[ "$model" == */* && "$model" != /* ]]; then
        # HF-style org/model path
        echo "${model##*/}"
    elif [[ "$model" == /* ]]; then
        # Local path - use the last directory component that isn't "checkpoint-*"
        local basename
        basename=$(basename "$model")
        if [[ "$basename" =~ ^checkpoint- ]]; then
            basename=$(basename "$(dirname "$model")")
        fi
        echo "$basename"
    else
        echo "$model"
    fi
}

# --- Print configuration ---
echo "======================================"
echo "Apertus Evaluation Launcher"
echo "  Mode:   $EVAL_MODE"
[[ "$EVAL_MODE" == "single" ]] && echo "  Task:   $SINGLE_TASK"
echo "  Failure policy: $EVAL_FAILURE_POLICY"
if [[ "$EVAL_FAILURE_POLICY" == "resume" ]]; then
    echo "  Chunk size: $EVAL_CHUNK_SIZE"
    echo "  Max parallel: ${EVAL_MAX_PARALLEL:-all chunks}"
    echo "  Retry waves: $EVAL_MAX_RETRIES"
fi
echo "  Harness: ${HARNESS_REPO:-swiss-ai/lm-evaluation-harness}${HARNESS_BRANCH:+@$HARNESS_BRANCH}"
if [[ "$THINKING_TOUCHED" == "true" ]]; then
    echo "  Thinking: enable=${ENABLE_THINKING_OVERRIDE:-<unset>} autodetect=${AUTODETECT_THINK_TOKENS:-false} track=${TRACK_THINKING_METRICS:-<derive>} lengths=${LOG_LENGTH_METRICS:-false}"
    [[ -n "$THINK_START_TOKEN" || -n "$THINK_END_TOKEN" ]] && echo "  Think tokens: start='${THINK_START_TOKEN:-<none>}' end='${THINK_END_TOKEN:-<none>}'"
fi
[[ -n "$REASONING_EFFORT" ]] && echo "  Reasoning effort: $REASONING_EFFORT"
echo "  API:    ${API_BASE_URL} (model=${API_MODEL_NAME:-<from --model>})"
echo "  API RPM: ${API_REQUESTS_PER_MINUTE:-unlimited}"
echo "  API max retries: ${API_MAX_RETRIES:-3 (default)}, concurrent requests: ${API_NUM_CONCURRENT:-8 (default)}"
echo "  API request timeout: ${API_REQUEST_TIMEOUT:-3600 (default)} s"
if [[ "${API_CHAT_ENDPOINT:-false}" == "true" ]]; then
    echo "  API endpoint: chat-completions"
else
    echo "  API endpoint: completions"
fi
[[ "${CHAT_FORMAT:-}" == "apertus2" ]] && echo "  Chat format: apertus2 (prefill=$PREFILL_MODE)"
[[ "${ANSWER_FRAME:-reply}" == "claim" ]] && echo "  Answer frame: claim (scored as by the Apertus 2 RL math reward)"
echo "  Slurm:  partition=${SBATCH_PARTITION:-normal} qos=${SBATCH_QOS:-<default>}${SBATCH_RESERVATION:+ reservation=$SBATCH_RESERVATION}"
echo "  Judge RPM: ${JUDGE_REQUESTS_PER_MINUTE:-unlimited}"
echo "  Judge model prefix: ${JUDGE_MODEL_PREFIX:-<none>}"

# --- Few-shot override ---
[[ -n "$FEWSHOT_FLAG" ]] && export NUM_FEWSHOT="$FEWSHOT_FLAG"

# --- Harness limit override ---
[[ -n "$HARNESS_LIMIT" ]] && export HARNESS_LIMIT="$HARNESS_LIMIT"

# --- Backend context limit ---
if [[ -n "$MAX_LENGTH" ]]; then
    [[ "$MAX_LENGTH" =~ ^[1-9][0-9]*$ ]] || { echo "Error: --max-length must be a positive integer (got '$MAX_LENGTH')." >&2; exit 1; }
    export MAX_LENGTH
fi
[[ -n "$HARNESS_BRANCH" ]] && export LM_EVAL_HARNESS_BRANCH="$HARNESS_BRANCH"
if [[ -n "$HARNESS_REPO" ]]; then
    [[ "$HARNESS_REPO" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] \
        || { echo "Error: --harness-repo must be <owner>/<repo> on GitHub (got '$HARNESS_REPO')." >&2; exit 1; }
    export LM_EVAL_HARNESS_REPO="$HARNESS_REPO"
fi

# Judge allocation is deferred to submit_evaluation(), after its result scan.
# If none is set, rely on an already hosted judge or a manual launch.
export JUDGE_JOB_IDS=""
JUDGE_TASKS_PATTERN="alpaca_eval|multijail|aya_redteaming|arena_hard_v01|arena_hard_v2|harmbench|hallulens|realtoxicitypromptsllama"

# warn if tasks are detected but judge is explicitly disabled (mode=none)
if [[ "$JUDGE_MODE" == "none" ]]; then
    if [[ -f "$TASKS" ]]; then
        if grep -qE "$JUDGE_TASKS_PATTERN" "$TASKS"; then
            echo "WARNING: Detected judge-dependent tasks but judge model launching is disabled (--judge none)"
            echo "Make sure a judge model is available via the CSCS serving platform or launch it manually"
        fi
    fi
fi

# A W&B upload asked for with --wandb-entity/--wandb-project needs the key, as in the other runners (the
# sbatch job only warns and skips the upload), so a missing one stops here.
# (secrets.sh only sets shell variables, so the key is passed on.)
WANDB_ENTITY=$WANDB_ENTITY_FLAG WANDB_PROJECT=$WANDB_PROJECT_FLAG WANDB_API_KEY=${WANDB_API_KEY:-} \
    bash -c 'source scripts/common/wandb.sh && wandb_setup > /dev/null' || exit 1
# The datasets of these tasks are gated on Hugging Face (the terms are accepted once, then a token reads them).
# Without HF_TOKEN (environment or scripts/secrets.env) or a logged-in token file the run fails when it loads
# the dataset, after the model has loaded: stop here (unless the run is offline, from the cache). Add a task when
# its dataset is gated.
GATED_TASKS_PATTERN='(^|[[:space:],])(gpqa|flores_plus|bouquet|harmbench)'
if [[ -f "${TASKS:-}" ]]; then GATED_CHECK_TASKS=$(<"$TASKS"); else GATED_CHECK_TASKS=${TASKS:-}; fi
if [[ -z "${HF_TOKEN:-}" && ! -s "${HF_HOME:-$HOME/.cache/huggingface}/token" && "${HF_HUB_OFFLINE:-0}" != 1 \
      && "${HF_DATASETS_OFFLINE:-0}" != 1 && "$GATED_CHECK_TASKS" =~ $GATED_TASKS_PATTERN ]]; then
    echo "Error: ${BASH_REMATCH[2]}* loads a gated Hugging Face dataset: set HF_TOKEN (environment or scripts/secrets.env)," \
         "or log in with huggingface-cli, and accept the dataset's terms." >&2
    exit 1
fi
# shellcheck source=scripts/common/preflight.sh
source scripts/common/preflight.sh
preflight_done

# --- Submit ---
EVAL_JOB_IDS=()

if [[ -z "$MODEL_NAME" ]]; then
    MODEL_NAME=$(auto_derive_name "$MODEL_PATH")
    if [[ "$THINKING_UMBRELLA" == "true" || "$ENABLE_THINKING_OVERRIDE" == "true" ]]; then
        MODEL_NAME="${MODEL_NAME}-think"
    fi
    [[ "${CHAT_FORMAT:-}" == "apertus2" ]] && MODEL_NAME="${MODEL_NAME}-apertus2-${PREFILL_MODE}"
    [[ "${ANSWER_FRAME:-reply}" == "claim" ]] && MODEL_NAME="${MODEL_NAME}-claim"
    [[ -n "$REASONING_EFFORT" ]] && MODEL_NAME="${MODEL_NAME}-effort-${REASONING_EFFORT}"
fi

export APPLY_CHAT_TEMPLATE="${CHAT_TEMPLATE_OVERRIDE:-true}"

[[ -n "$CUSTOM_TOKENIZER" ]] && export TOKENIZER="$CUSTOM_TOKENIZER"
[[ -n "$BOS_FLAG" ]] && export BOS="$BOS_FLAG"
[[ -n "$BOS_BY_CLIENT_FLAG" ]] && export BOS_BY_CLIENT="$BOS_BY_CLIENT_FLAG"

echo "  Model:  $MODEL_PATH"
echo "  Name:   $MODEL_NAME"
echo "  Chat:   $APPLY_CHAT_TEMPLATE"
[[ -n "$CUSTOM_TOKENIZER" ]] && echo "  Tok:    $CUSTOM_TOKENIZER"
[[ -n "$BOS_FLAG" ]] && echo "  BOS:    $BOS_FLAG"
[[ -n "$FEWSHOT_FLAG" ]] && echo "  Fewshot: $FEWSHOT_FLAG"
[[ -n "$MAX_LENGTH" ]] && echo "  Max length: $MAX_LENGTH"
[[ -n "$HARNESS_BRANCH" ]] && echo "  Harness branch: $HARNESS_BRANCH"
echo "  W&B:    $WANDB_ENTITY/$WANDB_PROJECT"
echo "======================================"

source scripts/lm_eval_harness/evaluation_orchestrator.sh
submit_evaluation "$MODEL_PATH" "$MODEL_NAME"
EVAL_JOB_IDS+=("$ORCHESTRATION_FINAL_JOB_ID")

# --- Judge cleanup job ---
# After all eval jobs are submitted, schedule a cleanup job that cancels judge
# SLURM jobs once all evaluations finish.
if [[ "$EVAL_FAILURE_POLICY" == "fail-fast" && -n "$JUDGE_JOB_IDS" \
      && "$KEEP_JUDGE" != "true" && ${#EVAL_JOB_IDS[@]} -gt 0 ]]; then
    DEP_STRING=$(IFS=':'; echo "${EVAL_JOB_IDS[*]}")
    SCANCEL_CMD="scancel $JUDGE_JOB_IDS"
    CLEANUP_JOB=$(sbatch --parsable \
        --account=infra01 \
        --partition="${SBATCH_PARTITION:-normal}" \
        --job-name "judge-cleanup" \
        --dependency="afterany:${DEP_STRING}" \
        --time=00:05:00 \
        --wrap="$SCANCEL_CMD")
    echo "Judge cleanup job $CLEANUP_JOB will cancel judge(s) [$JUDGE_JOB_IDS] after evals finish"
fi
