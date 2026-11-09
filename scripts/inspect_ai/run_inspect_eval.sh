#!/bin/bash
#
# run_inspect_eval.sh - Run any benchmark from Inspect AI (https://inspect.aisi.org.uk/) /
# inspect_evals (https://github.com/UKGovernmentBEIS/inspect_evals) against a model, as an
# alternative to the lm-evaluation-harness pipeline driven by launch_evaluations.sh /
# run_lm_eval.sbatch. Inspect's execution and logging model differs enough from lm-eval-harness's
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
#   scripts/inspect_ai/run_inspect_eval.sh --task <task[,task...]> --model <model> [options] [-- <extra args>]
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
#   --sandbox <type>           Override the tasks' sandbox (Inspect's --sandbox): docker, local or
#                              opensandbox. opensandbox runs each sample's sandbox as a Kata
#                              microVM through an OpenSandbox server
#                              (scripts/inspect_ai/opensandbox_sandbox.py): no Docker daemon or
#                              Kubernetes access, needs OPENSANDBOX_API_KEY. The task's own
#                              Dockerfile or one-service compose file is kept (a Dockerfile is
#                              replayed in the sandbox, as for Harbor). Leftover sandboxes of
#                              the run are deleted at the end.
#   --wandb-entity <entity>    W&B entity and project to upload the results to (also WANDB_ENTITY,
#   --wandb-project <project>  WANDB_PROJECT; default apertus/apertus-1.5-post-training-v0.1).
#                              Uploaded whenever a W&B key is available, like lm-eval-harness runs.
#   --no-wandb                 Don't upload (also WANDB_MODE=disabled), e.g. for a smoke test.
#   --temperature V, --top-p V, --top-k V, --min-p V, --repetition-penalty V
#                              Sampling of the model under test (defaults: configs/sampling_defaults.json,
#                              0.6 / 0.95; 'none' leaves one out). Passed as Inspect's own --temperature,
#                              --top-p and --top-k; min-p and repetition-penalty go in --extra-body, which
#                              an OpenAI-compatible server reads. A literal `-- --temperature ...` still wins.
#   --sampling-profile <name>  A profile of the sampling config (e.g. greedy); --sampling-config <file>
#                              reads another file instead of configs/sampling_defaults.json.
#   --enable-thinking, --no-enable-thinking
#                              Switch the model's chat-template thinking on or off (chat_template_kwargs
#                              enable_thinking, in --extra-body). Unset: the model's server-side default.
#   --reasoning-effort <level> Reasoning effort (none|minimal|low|medium|high|xhigh|max): Inspect's own
#                              --reasoning-effort, plus the template kwarg of the same name. Same flags
#                              as the other launchers (scripts/common/thinking.py); the reasoning is not
#                              stripped from the answer here, a server-side reasoning parser does that.
#   -- <extra args>            Everything after a literal -- is forwarded verbatim to
#                              `inspect eval`/`inspect eval-set` (e.g. -- --temperature 0.5
#                              --max-connections 10).
#
# Environment variables (used if the matching flag is not given; keys also from
# scripts/secrets.env, see scripts/common/secrets.sh):
#   API_BASE_URL, API_MODEL_NAME (defaults to --model) -- endpoint for the model under test.
#   TARGET_API_KEY -- API key for the --api-base-url endpoint (default: CSCS_SERVING_API, same
#   fallback as run_lm_eval.sbatch's openai backend).
#   WANDB_ENTITY, WANDB_PROJECT -- see --wandb-entity/--wandb-project above.
#   WANDB_API_KEY -- for the W&B upload (without it, no upload).
#   CRITPT_API_KEY -- grading-server credential for the CritPt task (scripts/aaii/aa_critpt.py).
#   Optional -- unset is the expected state until Artificial Analysis
#   grants access (email critpt@artificialanalysis.ai); the run still completes and writes its
#   submission batch to a local file instead of posting it.
#   SKIP_INSTALL=1 to run with the current environment as is, instead of the cached venv of
#   requirements/inspect-runtime.lock (scripts/common/locked_venv.sh).
#   EVAL_INSPECT_VENV -- a virtualenv holding requirements/inspect-runtime.lock, used instead of
#   the cached one; its bin/ goes first on PATH so `inspect`, `python` and `python3` below all
#   run from it. The prebuilt image sets it: Inspect lives apart from lm-eval-harness there, whose datasets pin
#   would otherwise hold inspect-evals back.
#
# Results are NOT viewed at https://inspect.aisi.org.uk/ -- that site is Inspect's
# documentation. Logs land in --logs-dir as .eval files; view them with:
#   inspect view --log-dir <logs-dir>
#
# Example (tau2-bench retail domain, smoke test):
#   scripts/inspect_ai/run_inspect_eval.sh --task tau2_retail --model Qwen/Qwen3-8B \
#     --api-base-url http://nid001234:8000 --model-role user=openai/gpt-4.1-2025-04-14 \
#     --task-arg message_limit=10 --limit 5

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

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
NO_WANDB=0
MODEL_ROLES=()
TASK_ARGS=()
EXTRA_ARGS=()
SANDBOX=""
SAMPLING_TEMPERATURE_FLAG=""; SAMPLING_TOP_P_FLAG=""; SAMPLING_TOP_K_FLAG=""
SAMPLING_MIN_P_FLAG=""; SAMPLING_REPETITION_PENALTY_FLAG=""
SAMPLING_PROFILE_FLAG=""; SAMPLING_CONFIG_FLAG=""
THINKING_ENABLE_FLAG=""; THINKING_EFFORT_FLAG=""

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
        --sandbox) SANDBOX=$2; shift 2 ;;
        --api-base-url) API_BASE_URL=$2; shift 2 ;;
        --wandb-entity) WANDB_ENTITY=$2; shift 2 ;;
        --wandb-project) WANDB_PROJECT=$2; shift 2 ;;
        --no-wandb) NO_WANDB=1; shift ;;
        --temperature) SAMPLING_TEMPERATURE_FLAG=$2; shift 2 ;;
        --top-p) SAMPLING_TOP_P_FLAG=$2; shift 2 ;;
        --top-k) SAMPLING_TOP_K_FLAG=$2; shift 2 ;;
        --min-p) SAMPLING_MIN_P_FLAG=$2; shift 2 ;;
        --repetition-penalty) SAMPLING_REPETITION_PENALTY_FLAG=$2; shift 2 ;;
        --sampling-profile) SAMPLING_PROFILE_FLAG=$2; shift 2 ;;
        --sampling-config) SAMPLING_CONFIG_FLAG=$2; shift 2 ;;
        --enable-thinking) THINKING_ENABLE_FLAG=--enable-thinking; shift ;;
        --no-enable-thinking) THINKING_ENABLE_FLAG=--no-enable-thinking; shift ;;
        --reasoning-effort) THINKING_EFFORT_FLAG=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *) die "Unknown argument: $1 (see --help)" ;;
    esac
done

[[ -n "$TASKS" ]] || die "Missing --task. See --help."
[[ -n "$MODEL" ]] || die "Missing --model. See --help."
[[ -z "$SANDBOX" || "$SANDBOX" == docker || "$SANDBOX" == local || "$SANDBOX" == opensandbox ]] ||
    die "--sandbox must be docker, local or opensandbox, got: $SANDBOX"
# shellcheck source=scripts/common/secrets.sh
source scripts/common/secrets.sh
if [[ "$SANDBOX" == opensandbox ]]; then
    [[ -n "${OPENSANDBOX_API_KEY:-}" ]] || die "--sandbox opensandbox needs OPENSANDBOX_API_KEY (environment or scripts/secrets.env)"
    export OPENSANDBOX_API_KEY
    [[ -n "${OPENSANDBOX_DOMAIN:-}" ]] && export OPENSANDBOX_DOMAIN
    # Every sandbox of this run carries the label; whatever is left of them is deleted on exit
    # (a sample's own are deleted as it ends, the rest by Inspect's task cleanup, this is the net).
    export OPENSANDBOX_RUN_LABEL=${OPENSANDBOX_RUN_LABEL:-inspect-$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')}
fi
# shellcheck source=scripts/common/wandb.sh
source scripts/common/wandb.sh
if (( NO_WANDB )); then
    wandb_setup "--no-wandb"
else
    wandb_setup || exit 1
fi

API_MODEL_NAME=${API_MODEL_NAME:-$MODEL}
NAME=${NAME:-$(echo "$MODEL" | tr '/: ' '---')}
LOGS_DIR=${LOGS_DIR:-"logs/inspect/$NAME"}

if [[ -n "$API_BASE_URL" ]]; then
    PROVIDER_ENV_NAME=$(echo "$PROVIDER_NAME" | tr '[:lower:]-' '[:upper:]_')

    # Normalize to a /v1 root, same as run_lm_eval.sbatch's openai backend (minus the
    # completions-suffix logic: Inspect's openai-api provider wants the /v1 root itself).
    if [[ "$API_BASE_URL" != */v1 && "$API_BASE_URL" != */v1/ ]]; then
        API_BASE_URL="${API_BASE_URL%/}/v1"
    fi

    TARGET_API_KEY="${TARGET_API_KEY:-${CSCS_SERVING_API:-}}"
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

# CritPt's grading-server credential (scripts/aaii/aa_critpt.py's own on_task_end Hook reads this
# directly from the environment) -- unconditional, like CSCS_SERVING_API above, since it's
# harmless to export for tasks that never look at it. Optional: unset just means that Hook
# writes its submission batch to a local file instead of posting it.
[[ -n "${CRITPT_API_KEY:-}" ]] && export CRITPT_API_KEY

mkdir -p "$LOGS_DIR"

# Snapshot existing .eval logs so the W&B upload below (if requested) only picks up logs this
# run produces, not ones already sitting in $LOGS_DIR from a previous run against the same name.
PRE_RUN_LOGS=$(find "$LOGS_DIR" -name '*.eval' 2>/dev/null | sort)

# Auto-imported by Python at interpreter startup for any dir on PYTHONPATH (the standard
# sitecustomize mechanism) -- patches Inspect's `local` sandbox provider (`--sandbox local`, for
# hosts without a Docker daemon) to avoid "OSError: [Errno 7] Argument list too long" on
# SciCode's larger multi-subproblem compositions. Unconditional (not gated behind SKIP_INSTALL):
# a preinstalled environment needs this safety net just as much as a freshly pip-installed one.
# See the patch's own module docstring for the full failure/fix reasoning.
export PYTHONPATH="$(pwd)/scripts/inspect_ai/sandbox_patches${PYTHONPATH:+:$PYTHONPATH}"

# EVAL_PREBUILT_ENV=1 (the prebuilt image, containers/Dockerfile.eval) has the same lock baked in.
if [[ "${SKIP_INSTALL:-0}" == "1" || "${EVAL_PREBUILT_ENV:-0}" == "1" ]]; then
    echo "SKIP_INSTALL=1 or EVAL_PREBUILT_ENV=1: using the preinstalled environment (no install)"
elif [[ -z "${EVAL_INSPECT_VENV:-}" ]]; then
    # shellcheck source=scripts/common/locked_venv.sh
    source "$(dirname "${BASH_SOURCE[0]}")/../common/locked_venv.sh"
    locked_venv "$(dirname "${BASH_SOURCE[0]}")/../../requirements/inspect-runtime.lock" 3.11 ||
        die "Installing requirements/inspect-runtime.lock failed. Set SKIP_INSTALL=1 if the environment already has it."
    EVAL_INSPECT_VENV=$LOCKED_VENV
fi

if [[ -n "${EVAL_INSPECT_VENV:-}" ]]; then
    [[ -x "$EVAL_INSPECT_VENV/bin/inspect" ]] || die "EVAL_INSPECT_VENV=$EVAL_INSPECT_VENV has no bin/inspect"
    export PATH="$EVAL_INSPECT_VENV/bin:$PATH"
fi

# A --sandbox local run has no Docker image to COPY these into, unlike inspect_evals/
# scicode's own default -- its scorer imports them as bare top-level modules
# ("from test_util import ...", "from process_data import ..."), so they need to already
# be importable from wherever `inspect eval` itself runs (this directory, since nothing
# below `cd`s elsewhere). Copied unconditionally -- also when nothing was installed above
# (SKIP_INSTALL, prebuilt image): two small files, harmless for any other task.
python3 -c "
import pathlib
import shutil

import inspect_evals.scicode

src = pathlib.Path(inspect_evals.scicode.__file__).parent
for name in ('test_util.py', 'process_data.py'):
    shutil.copy(src / name, name)
" || die "failed to copy inspect_evals/scicode's helper modules (test_util.py, process_data.py) into the working directory"

IFS=',' read -ra TASK_ARRAY <<< "$TASKS"
RESOLVED_TASKS=()
for task in "${TASK_ARRAY[@]}"; do
    [[ "$task" == */* ]] && RESOLVED_TASKS+=("$task") || RESOLVED_TASKS+=("inspect_evals/$task")
done

COMMON_ARGS=(--model "$TASK_MODEL" --log-dir "$LOGS_DIR")
for role in ${MODEL_ROLES[@]+"${MODEL_ROLES[@]}"}; do COMMON_ARGS+=(--model-role "$role"); done
for arg in ${TASK_ARGS[@]+"${TASK_ARGS[@]}"}; do COMMON_ARGS+=(-T "$arg"); done
[[ -n "$LIMIT" ]] && COMMON_ARGS+=(--limit "$LIMIT")
[[ -n "$SANDBOX" ]] && COMMON_ARGS+=(--sandbox "$SANDBOX")

# Default to a plain-text display when not attached to a terminal (see --eval-set note above) --
# an explicit --display in EXTRA_ARGS is appended after this and wins (Inspect keeps the last
# occurrence of a repeated option).
[[ -t 1 ]] || COMMON_ARGS+=(--display plain)

# Sampling of the model under test (scripts/common/sampling.py; defaults in configs/sampling_defaults.json).
# It goes before EXTRA_ARGS, so a literal `-- --temperature ...` still wins (Inspect keeps the last
# occurrence). Inspect has no --min-p or --repetition-penalty: those travel in --extra-body.
SAMPLING_ARGS=(--framework inspect --format shell)
[[ -n "$SAMPLING_CONFIG_FLAG"             ]] && SAMPLING_ARGS+=(--config "$SAMPLING_CONFIG_FLAG")
[[ -n "$SAMPLING_PROFILE_FLAG"            ]] && SAMPLING_ARGS+=(--profile "$SAMPLING_PROFILE_FLAG")
[[ -n "$SAMPLING_TEMPERATURE_FLAG"        ]] && SAMPLING_ARGS+=(--temperature "$SAMPLING_TEMPERATURE_FLAG")
[[ -n "$SAMPLING_TOP_P_FLAG"              ]] && SAMPLING_ARGS+=(--top-p "$SAMPLING_TOP_P_FLAG")
[[ -n "$SAMPLING_TOP_K_FLAG"              ]] && SAMPLING_ARGS+=(--top-k "$SAMPLING_TOP_K_FLAG")
[[ -n "$SAMPLING_MIN_P_FLAG"              ]] && SAMPLING_ARGS+=(--min-p "$SAMPLING_MIN_P_FLAG")
[[ -n "$SAMPLING_REPETITION_PENALTY_FLAG" ]] && SAMPLING_ARGS+=(--repetition-penalty "$SAMPLING_REPETITION_PENALTY_FLAG")
[[ -n "$SAMPLING_TEMPERATURE_FLAG$SAMPLING_TOP_P_FLAG$SAMPLING_TOP_K_FLAG$SAMPLING_MIN_P_FLAG$SAMPLING_REPETITION_PENALTY_FLAG$SAMPLING_PROFILE_FLAG$SAMPLING_CONFIG_FLAG" ]] &&
    SAMPLING_ARGS+=(--warn-greedy)
SAMPLING_OUT=$(python3 -m scripts.common.sampling "${SAMPLING_ARGS[@]}") || exit 1
eval "$SAMPLING_OUT"
[[ -n "$SAMPLING_TEMPERATURE" ]] && COMMON_ARGS+=(--temperature "$SAMPLING_TEMPERATURE")
[[ -n "$SAMPLING_TOP_P"       ]] && COMMON_ARGS+=(--top-p "$SAMPLING_TOP_P")
[[ -n "$SAMPLING_TOP_K"       ]] && COMMON_ARGS+=(--top-k "$SAMPLING_TOP_K")
# Thinking and reasoning effort (scripts/common/thinking.py): nothing is sent unless asked for. The
# effort is Inspect's own --reasoning-effort; both settings also go in --extra-body as chat-template
# kwargs, next to the sampling keys below.
THINKING_ARGS=(--format shell)
[[ -n "$THINKING_ENABLE_FLAG" ]] && THINKING_ARGS+=("$THINKING_ENABLE_FLAG")
[[ -n "$THINKING_EFFORT_FLAG" ]] && THINKING_ARGS+=(--reasoning-effort "$THINKING_EFFORT_FLAG")
THINKING_OUT=$(python3 -m scripts.common.thinking "${THINKING_ARGS[@]}") || exit 1
eval "$THINKING_OUT"
[[ -n "$THINKING_EFFORT" ]] && COMMON_ARGS+=(--reasoning-effort "$THINKING_EFFORT")
SAMPLING_EXTRA_BODY=""
[[ -n "$SAMPLING_MIN_P" ]] && SAMPLING_EXTRA_BODY="\"min_p\": $SAMPLING_MIN_P"
[[ -n "$SAMPLING_REPETITION_PENALTY" ]] &&
    SAMPLING_EXTRA_BODY="${SAMPLING_EXTRA_BODY:+$SAMPLING_EXTRA_BODY, }\"repetition_penalty\": $SAMPLING_REPETITION_PENALTY"
[[ -n "$THINKING_EXTRA_BODY" ]] &&
    SAMPLING_EXTRA_BODY="${SAMPLING_EXTRA_BODY:+$SAMPLING_EXTRA_BODY, }$THINKING_EXTRA_BODY"
if [[ -n "$SAMPLING_EXTRA_BODY" ]]; then
    COMMON_ARGS+=(--extra-body "{$SAMPLING_EXTRA_BODY}")
    for arg in ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; do
        [[ "$arg" == --extra-body* ]] && echo "WARNING: your --extra-body replaces the min-p/repetition-penalty/thinking settings." >&2
    done
fi

COMMON_ARGS+=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})

echo "Configuration set:"
printf '%s\n' "TASKS=${RESOLVED_TASKS[*]}" "MODEL=$MODEL" "TASK_MODEL=$TASK_MODEL" "NAME=$NAME" \
    "LOGS_DIR=$LOGS_DIR" "LIMIT=${LIMIT:-<none>}" "MODEL_ROLES=${MODEL_ROLES[*]:-<none>}" \
    "TASK_ARGS=${TASK_ARGS[*]:-<none>}" "SANDBOX=${SANDBOX:-<task default>}" ""

if [[ "$SANDBOX" == opensandbox ]]; then
    cleanup_sandboxes() {
        python3 -c "
import asyncio, os
from scripts.common.opensandbox_sandbox import delete_run_sandboxes
n = asyncio.run(delete_run_sandboxes(os.environ['OPENSANDBOX_RUN_LABEL']))
if n:
    print(f'deleted {n} leftover OpenSandbox sandbox(es) of run {os.environ[\"OPENSANDBOX_RUN_LABEL\"]}')
" || echo "WARNING: could not delete leftover sandboxes of run $OPENSANDBOX_RUN_LABEL" >&2
    }
    trap cleanup_sandboxes EXIT
fi

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

if (( WANDB_UPLOAD )); then
    NEW_LOGS=$(comm -13 <(echo "$PRE_RUN_LOGS") <(find "$LOGS_DIR" -name '*.eval' 2>/dev/null | sort))
    if [[ -z "$NEW_LOGS" ]]; then
        echo "No new .eval logs found under $LOGS_DIR -- skipping W&B upload." >&2
    else
        UPLOAD_CMD=(python -m scripts.results.update_wandb_inspect --entity "$WANDB_ENTITY" \
            --project "$WANDB_PROJECT" --name "$NAME")
        while IFS= read -r log; do UPLOAD_CMD+=(--eval-log "$log"); done <<< "$NEW_LOGS"

        echo ""
        echo "Uploading results to wandb"
        echo "Running: ${UPLOAD_CMD[*]}"
        "${UPLOAD_CMD[@]}"
    fi
fi
