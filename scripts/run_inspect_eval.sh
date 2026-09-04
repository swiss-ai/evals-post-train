#!/bin/bash
#
# run_inspect_eval.sh - Run any benchmark from Inspect AI (https://inspect.aisi.org.uk/) /
# inspect_evals (https://github.com/UKGovernmentBEIS/inspect_evals) against a model, as an
# alternative to the lm-evaluation-harness pipeline driven by launch_evaluations.sh /
# evaluate.sbatch. Inspect's execution and logging model differs enough from lm-eval-harness's
# (task registry, model roles, .eval log files) that it gets its own script rather than another
# launch_evaluations.sh backend.
#
# The model under test can be reached two ways:
#   - Any Inspect-native model string passed straight through, e.g. --model anthropic/claude-...
#     (you manage that provider's own env vars, e.g. ANTHROPIC_API_KEY).
#   - A model served behind an OpenAI-compatible HTTP endpoint (CSCS serving, `vllm serve`, ...),
#     the same way this repo's lm-eval-harness pipeline already evaluates against one with
#     `--backend openai`: pass --api-base-url and --model becomes the bare served model name.
#     This wraps --model through Inspect's generic `openai-api` provider.
#
# Some benchmarks need extra model roles beyond the model under test (e.g. tau2-bench's
# "user"-role simulator, or an LLM-as-judge "grader" role) or extra task parameters (e.g. tau2's
# banking `retrieval_config`, or a `message_limit`) -- pass those with repeatable
# --model-role/--task-arg flags, or forward anything else straight to `inspect eval` after `--`.
#
# Usage:
#   scripts/run_inspect_eval.sh --task <task[,task...]> --model <model> [options] [-- <extra args>]
#
# Required:
#   --task <name[,name...]>  One or more inspect_evals task names, e.g. tau2_retail, gsm8k,
#                             gaia. A bare name (no "/") is expanded to inspect_evals/<name>;
#                             pass a full path/module (e.g. my_tasks/custom.py) to use something
#                             outside inspect_evals. Comma-separated names run as separate
#                             `inspect eval` invocations (use --eval-set to run them as one
#                             `inspect eval-set` instead).
#   --model <model>          Inspect model string (e.g. openai/gpt-4o, anthropic/claude-...), or
#                             just the served model name when --api-base-url is given.
#
# Options:
#   --api-base-url <url>     OpenAI-compatible endpoint serving --model (bare host, /v1 root, or
#                             full /v1/chat/completions URL). Also settable via API_BASE_URL.
#                             When given, --model is wrapped as openai-api/<provider-name>/<model>.
#   --provider-name <name>   Inspect provider name used with --api-base-url (default: swissai).
#                             Determines the env vars Inspect reads for it
#                             (<NAME>_BASE_URL, <NAME>_API_KEY).
#   --model-role <role=model> Assign an Inspect model role, e.g. --model-role user=openai/gpt-4.1
#                             (tau2's user-simulator) or --model-role grader=openai/gpt-4o.
#                             Repeatable. The role's own provider env vars (e.g. OPENAI_API_KEY)
#                             are your responsibility to set.
#   --task-arg <key=value>   Extra `-T key=value` task parameter, e.g. --task-arg message_limit=10
#                             or --task-arg retrieval_config=full_kb. Repeatable.
#   --limit <n>               Restrict to n samples per task (useful for smoke-testing).
#   --eval-set                Run all --task names as one `inspect eval-set` (parallel, with
#                              automatic retry of failed samples) instead of one `inspect eval`
#                              call per task.
#                              Note: when stdout isn't a terminal (e.g. inside an sbatch job),
#                              Inspect's default "full" display renders a Rich TUI that emits raw
#                              ANSI escapes -- unreadable once redirected to a .out/.err file. This
#                              script defaults to `--display plain` in that case so the SLURM
#                              .out/.err logs read like lm-eval-harness's; pass `-- --display
#                              <mode>` to override (e.g. --display none).
#   --name <name>              Run name, used for the log directory (default: derived from --model).
#   --logs-dir <path>          Where Inspect writes .eval log files (default: logs/inspect/<name>).
#   --wandb-entity <entity>    WandB entity to upload results to. Also settable via WANDB_ENTITY.
#                              Uploading only happens when both this and --wandb-project are set
#                              (unlike evaluate.sbatch's lm-eval-harness pipeline, upload here is
#                              opt-in -- this script is also used for one-off/smoke-test runs that
#                              you may not want landing in W&B).
#   --wandb-project <project>  WandB project to upload results to. Also settable via WANDB_PROJECT.
#   -- <extra args>            Everything after a literal -- is forwarded verbatim to
#                              `inspect eval`/`inspect eval-set` (e.g. -- --temperature 0.5
#                              --max-connections 10).
#
# Environment variables (used if the matching flag is not given):
#   API_BASE_URL, API_MODEL_NAME (defaults to --model) -- endpoint for the model under test.
#   TARGET_API_KEY -- API key for the --api-base-url endpoint (default: CSCS_SERVING_API from
#   scripts/cscs_serving_api_key.txt, same fallback as evaluate.sbatch's openai backend).
#   WANDB_ENTITY, WANDB_PROJECT -- see --wandb-entity/--wandb-project above.
#   WANDB_API_KEY -- required if uploading to W&B (default: scripts/wandb_api_key.txt, same
#   fallback as evaluate.sbatch).
#   SKIP_INSTALL=1 to skip the runtime pip install of inspect-ai/inspect-evals.
#
# Results are NOT viewed at https://inspect.aisi.org.uk/ -- that site is Inspect's
# documentation. Logs land in --logs-dir as .eval files; view them with:
#   inspect view --log-dir <logs-dir>
#
# Example (tau2-bench retail domain, smoke test):
#   scripts/run_inspect_eval.sh --task tau2_retail --model Qwen/Qwen3-8B \
#     --api-base-url http://nid001234:8000 --model-role user=openai/gpt-4.1-2025-04-14 \
#     --task-arg message_limit=10 --limit 5

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

die() { echo "$*" >&2; exit 1; }

usage() {
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}

TASKS=""
MODEL=""
NAME=""
PROVIDER_NAME="swissai"
LIMIT=""
LOGS_DIR=""
EVAL_SET=0
API_BASE_URL=${API_BASE_URL:-""}
API_MODEL_NAME=${API_MODEL_NAME:-""}
WANDB_ENTITY=${WANDB_ENTITY:-""}
WANDB_PROJECT=${WANDB_PROJECT:-""}
MODEL_ROLES=()
TASK_ARGS=()
EXTRA_ARGS=()

while (( $# > 0 )); do
    case "$1" in
        --task) TASKS=$2; shift 2 ;;
        --model) MODEL=$2; shift 2 ;;
        --name) NAME=$2; shift 2 ;;
        --provider-name) PROVIDER_NAME=$2; shift 2 ;;
        --model-role) MODEL_ROLES+=("$2"); shift 2 ;;
        --task-arg) TASK_ARGS+=("$2"); shift 2 ;;
        --limit) LIMIT=$2; shift 2 ;;
        --eval-set) EVAL_SET=1; shift ;;
        --logs-dir) LOGS_DIR=$2; shift 2 ;;
        --api-base-url) API_BASE_URL=$2; shift 2 ;;
        --wandb-entity) WANDB_ENTITY=$2; shift 2 ;;
        --wandb-project) WANDB_PROJECT=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *) die "Unknown argument: $1 (see --help)" ;;
    esac
done

[[ -n "$TASKS" ]] || die "Missing --task. See --help."
[[ -n "$MODEL" ]] || die "Missing --model. See --help."
if [[ -n "$WANDB_ENTITY" && -z "$WANDB_PROJECT" ]] || [[ -z "$WANDB_ENTITY" && -n "$WANDB_PROJECT" ]]; then
    die "--wandb-entity and --wandb-project must be given together (or neither, to skip W&B upload)."
fi

API_MODEL_NAME=${API_MODEL_NAME:-$MODEL}
NAME=${NAME:-$(echo "$MODEL" | tr '/: ' '---')}
LOGS_DIR=${LOGS_DIR:-"logs/inspect/$NAME"}

if [[ -n "$API_BASE_URL" ]]; then
    PROVIDER_ENV_NAME=$(echo "$PROVIDER_NAME" | tr '[:lower:]-' '[:upper:]_')

    # Normalize to a /v1 root, same as evaluate.sbatch's openai backend (minus the
    # completions-suffix logic: Inspect's openai-api provider wants the /v1 root itself).
    if [[ "$API_BASE_URL" != */v1 && "$API_BASE_URL" != */v1/ ]]; then
        API_BASE_URL="${API_BASE_URL%/}/v1"
    fi

    CSCS_SERVING_API="${CSCS_SERVING_API:-}"
    if [[ -z "$CSCS_SERVING_API" && -f ./scripts/cscs_serving_api_key.txt ]]; then
        CSCS_SERVING_API="$(tr -d '\r\n' < ./scripts/cscs_serving_api_key.txt)"
    fi
    TARGET_API_KEY="${TARGET_API_KEY:-$CSCS_SERVING_API}"
    key_var="${PROVIDER_ENV_NAME}_API_KEY"
    export "${key_var}"="${!key_var:-$TARGET_API_KEY}"
    export "${PROVIDER_ENV_NAME}_BASE_URL"="$API_BASE_URL"

    TASK_MODEL="openai-api/${PROVIDER_NAME}/${API_MODEL_NAME}"
else
    TASK_MODEL="$MODEL"
fi

if (( ${#MODEL_ROLES[@]} > 0 )); then
    echo "NOTE: --model-role roles (${MODEL_ROLES[*]}) need their own provider credentials" \
        "(e.g. OPENAI_API_KEY for an openai/... role) -- that's on you to set." >&2
fi

mkdir -p "$LOGS_DIR"

# Snapshot existing .eval logs so the W&B upload below (if requested) only picks up logs this
# run produces, not ones already sitting in $LOGS_DIR from a previous run against the same name.
PRE_RUN_LOGS=$(find "$LOGS_DIR" -name '*.eval' 2>/dev/null | sort)

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
    # The `openai` package is required by Inspect's openai-api provider (used whenever
    # --api-base-url is given) even for non-OpenAI-hosted endpoints -- confirmed by a bare
    # `inspect-ai` install failing with "OpenAI Compatible API requires optional dependencies"
    # against a served model. It's not an inspect-ai extra, just a separate package.
    #
    # gdown is likewise required by inspect_evals/scicode's dataset loader (a Google Drive
    # download) but not declared as an inspect-evals dependency -- confirmed by a real run
    # (--task scicode, plain and aaii/-wrapped alike) failing with "Google Drive download
    # requires optional dependencies. Install with: pip install gdown" otherwise.
    pip install --no-cache-dir --upgrade "inspect-ai>=0.3.258" "inspect-evals" openai gdown \
        || die "pip install of inspect-ai/inspect-evals failed. Set SKIP_INSTALL=1 if the environment already has them."
else
    echo "SKIP_INSTALL=1: using the preinstalled environment (no pip install)"
fi

IFS=',' read -ra TASK_ARRAY <<< "$TASKS"
RESOLVED_TASKS=()
for task in "${TASK_ARRAY[@]}"; do
    [[ "$task" == */* ]] && RESOLVED_TASKS+=("$task") || RESOLVED_TASKS+=("inspect_evals/$task")
done

COMMON_ARGS=(--model "$TASK_MODEL" --log-dir "$LOGS_DIR")
for role in "${MODEL_ROLES[@]}"; do COMMON_ARGS+=(--model-role "$role"); done
for arg in "${TASK_ARGS[@]}"; do COMMON_ARGS+=(-T "$arg"); done
[[ -n "$LIMIT" ]] && COMMON_ARGS+=(--limit "$LIMIT")

# Default to a plain-text display when not attached to a terminal (see --eval-set note above) --
# an explicit --display in EXTRA_ARGS is appended after this and wins (Inspect keeps the last
# occurrence of a repeated option).
[[ -t 1 ]] || COMMON_ARGS+=(--display plain)

COMMON_ARGS+=("${EXTRA_ARGS[@]}")

echo "Configuration set:"
printf '%s\n' "TASKS=${RESOLVED_TASKS[*]}" "MODEL=$MODEL" "TASK_MODEL=$TASK_MODEL" "NAME=$NAME" \
    "LOGS_DIR=$LOGS_DIR" "LIMIT=${LIMIT:-<none>}" "MODEL_ROLES=${MODEL_ROLES[*]:-<none>}" \
    "TASK_ARGS=${TASK_ARGS[*]:-<none>}" ""

if (( EVAL_SET )); then
    CMD=(inspect eval-set "${RESOLVED_TASKS[@]}" "${COMMON_ARGS[@]}")
    echo "Running: ${CMD[*]}"
    "${CMD[@]}"
else
    for task in "${RESOLVED_TASKS[@]}"; do
        CMD=(inspect eval "$task" "${COMMON_ARGS[@]}")
        echo "Running: ${CMD[*]}"
        "${CMD[@]}"
    done
fi

echo ""
echo "Done. View results with: inspect view --log-dir $LOGS_DIR"

if [[ -n "$WANDB_ENTITY" ]]; then
    NEW_LOGS=$(comm -13 <(echo "$PRE_RUN_LOGS") <(find "$LOGS_DIR" -name '*.eval' 2>/dev/null | sort))
    if [[ -z "$NEW_LOGS" ]]; then
        echo "No new .eval logs found under $LOGS_DIR -- skipping W&B upload." >&2
    else
        WANDB_API_KEY="${WANDB_API_KEY:-}"
        if [[ -z "$WANDB_API_KEY" && -f ./scripts/wandb_api_key.txt ]]; then
            WANDB_API_KEY="$(tr -d '\r\n' < ./scripts/wandb_api_key.txt)"
        fi
        [[ -n "$WANDB_API_KEY" ]] || die "W&B upload requested but WANDB_API_KEY is not set and scripts/wandb_api_key.txt is missing."
        export WANDB_API_KEY

        UPLOAD_CMD=(python -m scripts.alignment.update_wandb_inspect --entity "$WANDB_ENTITY" \
            --project "$WANDB_PROJECT" --name "$NAME")
        while IFS= read -r log; do UPLOAD_CMD+=(--eval-log "$log"); done <<< "$NEW_LOGS"

        echo ""
        echo "Uploading results to wandb"
        echo "Running: ${UPLOAD_CMD[*]}"
        "${UPLOAD_CMD[@]}"
    fi
fi
