#!/bin/bash
#
# run_gaia2_eval.sh - Run Omnilingual GAIA2 (facebook/omnilingual-gaia2: Meta's GAIA2 agent
# benchmark translated into 10 languages, 4 subsets of 160 scenarios each) through Meta's
# gaia2-runner, against a model behind an OpenAI-compatible endpoint. Each scenario is a
# simulated phone of apps (email, calendar, chats, cabs, shopping, ...); the agent (OpenClaw)
# works through the apps' command-line tools while events arrive, and an LLM judge compares its
# actions and the apps' final state with the reference.
#
# Every scenario runs in its own Kata microVM on an OpenSandbox server
# (scripts/gaia2/opensandbox_launcher.py): no Docker, no Kubernetes access, only
# OPENSANDBOX_API_KEY. The agent and the judge run inside the sandbox and call their models from
# there, so --api-base-url and --judge-base-url must be reachable from the sandbox: a public
# endpoint such as the SwissAI gateway, not a server inside a Clariden job. The sandbox reaches
# only those hosts (--sandbox-network allowlist).
#
# gaia2-runner, gaia2-cli and gaia2-core are pinned to one commit of
# facebookresearch/meta-agents-research-environments (requirements/gaia2-runtime.lock, Python
# >= 3.12), as are the agent and oracle images (ghcr.io/swiss-ai/gaia2-oc and gaia2-oracle,
# built by .github/workflows/gaia2-images.yml). See docs/gaia2.md.
#
# Usage:
#   scripts/gaia2/run_gaia2_eval.sh --model <served model> --api-base-url <url> [options]
#   scripts/gaia2/run_gaia2_eval.sh --gold [options]
#
# Options:
#   --model <name>             Served model name at --api-base-url. OpenClaw drives the apps through
#                               native tool calls: the endpoint must serve them for this model.
#   --api-base-url <url>       OpenAI-compatible endpoint (bare host or /v1 root).
#   --gold                      Meta's oracle image (each scenario's reference actions replayed, no
#                               model): a check of the sandboxes, apps and judge; all should pass.
#   --language <l[,l]|all>      cmn_Hans, deu_Latn, fra_Latn, hin_Deva, ind_Latn, ita_Latn, jpn_Jpan,
#                               por_Latn, spa_Latn, tur_Latn (default: all).
#   --subset <s[,s]|all>        execution, search, ambiguity, adaptability (default: all).
#   --limit <n>                 The first n scenarios of each language and subset.
#   --pass-at <k>               Attempts per scenario (default: 1; Meta's headline metric is pass@3).
#   --concurrency <n>           Sandboxes alive at once (default: 4). Mind the endpoints' rate limits:
#                               every scenario makes many model calls, and the judge several.
#   --thinking off|low|medium|high  OpenClaw's thinking level (default: low).
#   --timeout <s>               Seconds per scenario (default: 1200; at least its duration + 60).
#   --idle-timeout <s>          Seconds without agent activity before the daemon gives up (default: 600).
#   --judge-model <name>        The judge (default: CSCS-Inference/zai-org/GLM-5.3). Meta's numbers use
#                               gpt-oss-120b (reasoning effort low): with another judge, scores are not
#                               directly comparable to theirs.
#   --judge-base-url <url>      The judge's endpoint (default: https://api.swissai.svc.cscs.ch/v1).
#   --judge-extra-body <json>   Extra fields for the judge's requests, e.g. '{"reasoning_effort": "low"}'.
#   --agent-image <ref>, --oracle-image <ref>  Other images (default: the pinned ghcr ones).
#   --sandbox-network allowlist|public  allowlist (default): the sandbox reaches only the model's and the
#                               judge's hosts; public: the open internet.
#   --name <name>              Run name (default: derived from --model).
#   --work-dir <path>          Output dir (default: logs/gaia2/<name>): runs/<language>/<subset>/
#                               (gaia2-runner's config and results) and summary.json.
#   --wandb-entity <entity>     W&B entity and project to upload the results to (also WANDB_ENTITY,
#   --wandb-project <project>   WANDB_PROJECT; default apertus/e-p-t-pool): one run named --name, with
#                               per language and subset the success rate (and pass@k), n_scenarios,
#                               n_errors and a few sample scenarios. Not for --gold runs.
#   --no-wandb                  Don't upload (also WANDB_MODE=disabled).
#
# Environment (keys also from scripts/secrets.env, see scripts/common/secrets.sh):
#   TARGET_API_KEY       key for --api-base-url (default: CSCS_SERVING_API).
#   JUDGE_API_KEY        key for --judge-base-url (default: the model's key).
#   OPENSANDBOX_API_KEY  key for the OpenSandbox server (required).
#   OPENSANDBOX_DOMAIN   the server (default: Breithorn's Kata endpoint).
#   OPENSANDBOX_RUN_LABEL  sandboxes are labelled sandbox-run=<label> (default: random); the run
#                        deletes what is left of its label at the end.
#   OPENSANDBOX_LOG_DIR  if set, each sandbox's diagnostics are saved there before it is killed.
#   GAIA2_SANDBOX_CPU, GAIA2_SANDBOX_MEMORY  per sandbox (default: 2, 4Gi).
#   GAIA2_HF_CACHE       where gaia2-runner caches the dataset (default: ~/.cache/gaia2/hf_datasets).
#   HF_TOKEN             if the dataset needs one.
#   GAIA2_VENV           venv to run from (default: the cached venv of the lock, see
#                        scripts/common/locked_venv.sh). With EVAL_PREBUILT_ENV=1 (the prebuilt image,
#                        containers/Dockerfile.eval) it must already hold gaia2-runner.
#   WANDB_API_KEY        for the W&B upload (without it, no upload).
#
# The run exits non-zero when any attempt got no verdict (errors in summary.json): those scores
# count it as not passed, as gaia2-runner does, and are incomplete.
#
# Examples:
#   scripts/gaia2/run_gaia2_eval.sh --gold --language deu_Latn --subset execution --limit 2 --no-wandb
#   scripts/gaia2/run_gaia2_eval.sh --model SwissAI-Research/swiss-ai/chonk-lc-16k-v0.2-sft \
#     --api-base-url https://api.swissai.svc.cscs.ch/v1 --language deu_Latn --limit 5 --no-wandb
#   On Clariden: sbatch scripts/gaia2/run_gaia2_eval.sbatch --model ... (see that script).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

die() { echo "$*" >&2; exit 1; }

usage() {
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}

MODEL=""
NAME=""
WORK_DIR=""
GOLD=0
API_BASE_URL=${API_BASE_URL:-""}
NO_WANDB=0
RUN_ARGS=()

while (( $# > 0 )); do
    case "$1" in
        --model) MODEL=$2; shift 2 ;;
        --name) NAME=$2; shift 2 ;;
        --work-dir) WORK_DIR=$2; shift 2 ;;
        --api-base-url) API_BASE_URL=$2; shift 2 ;;
        --gold) GOLD=1; shift ;;
        --wandb-entity) WANDB_ENTITY=$2; shift 2 ;;
        --wandb-project) WANDB_PROJECT=$2; shift 2 ;;
        --no-wandb) NO_WANDB=1; shift ;;
        --language|--subset|--limit|--pass-at|--concurrency|--thinking|--timeout|--idle-timeout|\
        --judge-model|--judge-base-url|--judge-extra-body|--agent-image|--oracle-image|--sandbox-network)
            RUN_ARGS+=("$1" "$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown argument: $1 (see --help)" ;;
    esac
done

if [[ "$GOLD" == "1" ]]; then
    [[ -z "$MODEL" ]] || die "--gold replays the reference actions; drop --model."
    MODEL_ARGS=(--gold)
else
    [[ -n "$MODEL" ]] || die "Missing --model (or --gold). See --help."
    [[ -n "$API_BASE_URL" ]] || die "Missing --api-base-url. See --help."
    [[ "$API_BASE_URL" == */v1 || "$API_BASE_URL" == */v1/ ]] || API_BASE_URL="${API_BASE_URL%/}/v1"
    MODEL_ARGS=(--model "$MODEL" --api-base-url "${API_BASE_URL%/}")
fi

# shellcheck source=scripts/common/secrets.sh
source scripts/common/secrets.sh
# shellcheck source=scripts/common/cache_dirs.sh
source scripts/common/cache_dirs.sh
# shellcheck source=scripts/common/wandb.sh
source scripts/common/wandb.sh
if (( NO_WANDB )); then
    wandb_setup "--no-wandb"
elif (( GOLD )); then
    wandb_setup "--gold: reference actions, not a model"
else
    wandb_setup || exit 1
fi

[[ -n "${OPENSANDBOX_API_KEY:-}" ]] || die "OPENSANDBOX_API_KEY is not set: every scenario runs on OpenSandbox."
export OPENSANDBOX_API_KEY

NAME=${NAME:-$(echo "${MODEL:-gold}" | tr '/: ' '---')}
WORK_DIR=${WORK_DIR:-"logs/gaia2/$NAME"}

# Read by scripts/gaia2/run.py and handed to gaia2-runner by name, never written to its config.
export EVAL_MODEL_API_KEY="${TARGET_API_KEY:-${CSCS_SERVING_API:-}}"
export EVAL_JUDGE_API_KEY="${JUDGE_API_KEY:-$EVAL_MODEL_API_KEY}"
[[ -n "$EVAL_JUDGE_API_KEY" ]] || die "No key for the judge: set JUDGE_API_KEY (or TARGET_API_KEY / CSCS_SERVING_API)."

# gaia2-runner needs Python >= 3.12, which uv fetches where the system has none (Clariden's 3.6).
export PATH="$HOME/.local/bin:$PATH"
if [[ "${EVAL_PREBUILT_ENV:-0}" == "1" ]]; then
    PY="${GAIA2_VENV:-}/bin/python"
    [[ -n "${GAIA2_VENV:-}" && -x "$PY" ]] ||
        die "EVAL_PREBUILT_ENV=1 but GAIA2_VENV (${GAIA2_VENV:-unset}) has no bin/python:" \
            "this image has no gaia2 environment, and a prebuilt image installs nothing at run time."
    echo "EVAL_PREBUILT_ENV=1: using $GAIA2_VENV (no install)"
else
    if [[ -z "${GAIA2_VENV:-}" ]]; then
        command -v uv >/dev/null 2>&1 || die "uv is required (https://docs.astral.sh/uv/)."
        # shellcheck source=scripts/common/locked_venv.sh
        source scripts/common/locked_venv.sh
        locked_venv requirements/gaia2-runtime.lock 3.12 ||
            die "Installing requirements/gaia2-runtime.lock failed."
        GAIA2_VENV=$LOCKED_VENV
    fi
    PY="$GAIA2_VENV/bin/python"
    [[ -x "$PY" ]] || die "GAIA2_VENV=$GAIA2_VENV has no bin/python"
fi

mkdir -p "$WORK_DIR"
CMD=("$PY" -m scripts.gaia2.run "${MODEL_ARGS[@]}" --work-dir "$WORK_DIR" ${RUN_ARGS[@]+"${RUN_ARGS[@]}"})
echo "Running: ${CMD[*]}"
# In the background with SIGTERM passed on, so that a stopped run (e.g. the sbatch's time-limit
# signal) still writes summary.json.
"${CMD[@]}" &
RUN_PID=$!
trap 'kill -TERM "$RUN_PID" 2>/dev/null || true' TERM INT
STATUS=0
wait "$RUN_PID" || STATUS=$?
while kill -0 "$RUN_PID" 2>/dev/null; do wait "$RUN_PID" || STATUS=$?; done
trap - TERM INT
echo ""
echo "Done. Scores and run metadata: $WORK_DIR/summary.json"

if (( WANDB_UPLOAD )) && [[ -f "$WORK_DIR/summary.json" ]]; then
    UPLOAD_CMD=("$PY" -m scripts.results.update_wandb_gaia2 --entity "$WANDB_ENTITY"
                --project "$WANDB_PROJECT" --name "$NAME" --work-dir "$WORK_DIR")
    echo ""
    echo "Uploading results to wandb"
    echo "Running: ${UPLOAD_CMD[*]}"
    "${UPLOAD_CMD[@]}"
fi
exit $STATUS
