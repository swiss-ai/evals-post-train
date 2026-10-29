#!/bin/bash
#
# run_evalscope_eval.sh - Run SWE-bench (Verified) and SWE-bench Pro through EvalScope
# (https://github.com/modelscope/evalscope) against a model served behind an OpenAI-compatible
# endpoint -- a third harness next to lm-evaluation-harness (launch_evaluations.sh) and Inspect
# (run_inspect_eval.sh), for these two benchmarks only. Both run the same protocol: a multi-turn
# bash agent loop inside the instance's own container, then the instance's tests on the agent's
# patch (resolved = every FAIL_TO_PASS and PASS_TO_PASS test passes).
#
#   swe_bench      SWE-bench Verified, 500 tasks (swebench/sweb.eval.x86_64.* images).
#   swe_bench_pro  SWE-bench Pro v1, 731 tasks (jefzda/sweap-images:<tag>) -- see "v1 vs V2" below.
#
# EvalScope lives in its own venv (requirements/evalscope-runtime.txt, pinned exactly): its
# math-verify pin conflicts with the lm-eval/Inspect environment's antlr4 pin.
#
# Every sample gets its own sandbox container, so this needs a container runtime:
#   --sandbox docker  EvalScope's own path, a local Docker daemon (a dev box / docker launcher).
#   --sandbox k8s     One unprivileged pod per sandbox in the current namespace, through
#                     scripts/common/k8s_sandbox.py (no Docker daemon needed; same
#                     approach as evals-svc's Terminal-Bench trial pods). Needs a ServiceAccount
#                     that can create/get/delete pods and pods/exec. The images are linux/amd64,
#                     so the pods are pinned to amd64 nodes (not Clariden's GH200s).
#   Default: k8s when running in a pod, else docker.
#
# Usage:
#   scripts/evalscope/run_evalscope_eval.sh --benchmark <swe_bench|swe_bench_pro|swe_bench,swe_bench_pro> \
#     --model <served model> --api-base-url <url> [options]
#
# Required:
#   --benchmark <b[,b]>        swe_bench, swe_bench_pro, or both (comma-separated).
#   --model <name>             Served model name at --api-base-url. The model must serve tool
#                               calls (vLLM --enable-auto-tool-choice + parser, sglang
#                               --tool-call-parser) for the default toolcall strategy; else
#                               use --strategy swe_bench_backticks.
#   --api-base-url <url>       OpenAI-compatible endpoint (bare host or /v1 root). Also settable
#                               via API_BASE_URL.
#
# Options:
#   --name <name>              Run name for the work dir (default: derived from --model).
#   --work-dir <path>          Output dir (default: logs/evalscope/<name>). Holds EvalScope's own
#                               outputs (predictions/, reviews/, reports/, logs/) plus summary.json.
#   --sandbox docker|k8s       See above.
#   --sandbox-config <json>    Merged into every sandbox's config, e.g.
#                               '{"memory_limit": "16g", "cpu_limit": 4}' (default 2 CPUs per sandbox;
#                               memory 4g with docker, and with k8s an 8g limit over a 4g request,
#                               since 4g OOM-kills some SWE-bench Pro tests -- see run.py).
#   --limit <n>                 First n samples per benchmark (smoke tests); a fraction < 1 is a
#                               share of the dataset.
#   --shuffle                   Shuffle each dataset (EvalScope's seed 42) before --limit, so a small
#                               --limit covers several repos (SWE-bench Verified is sorted by repo).
#   --max-concurrency <n>       Samples in flight at once = sandboxes alive at once (default: 4).
#   --max-steps <n>             Agent loop step cap (default: the benchmark's, 250 for both).
#   --strategy <name>           swe_bench_toolcall (the default, function calling) or
#                               swe_bench_backticks (text-based actions, no tool-call parser needed).
#   --command-timeout <s>       Default timeout for each agent bash command.
#   --generation-config <json> Merged over the default {"temperature": 0}, e.g.
#                               '{"max_tokens": 8192, "retries": 100, "retry_interval": 60}' (more
#                               patient retries for a rate-limited endpoint; EvalScope's default is 5).
#   --dataset-revision <b>=<sha> Hugging Face dataset revision for benchmark <b> (repeatable). Both
#                               are pinned by default (scripts/evalscope/run.py: BENCHMARKS);
#                               swe_bench_pro's pin is the last v1 commit, see "v1 vs V2" below.
#   --extra-param <key=value>   An EvalScope extra_param for the selected benchmark(s) (repeatable),
#                               e.g. swe_bench_pro's eval_timeout=7200 or swe_bench_pro_repo_path=/path.
#   --api-timeout <s>           Per-request timeout towards the model endpoint.
#   --gold                      Score each instance's reference patch instead of running a model (no
#                               --model/--api-base-url): a check of the scoring pipeline, where every
#                               instance should come out resolved.
#
# Environment:
#   TARGET_API_KEY   key for --api-base-url (default: scripts/cscs_serving_api_key.txt, the same
#                    fallback run_inspect_eval.sh uses).
#   EVALSCOPE_VENV   venv to install into / run from (default: a cache dir keyed by the
#                    requirements file's hash). SKIP_INSTALL=1 runs with the current python3 as is.
#                    With EVAL_PREBUILT_ENV=1 (the prebuilt image, containers/Dockerfile.eval) it
#                    must already hold EvalScope: nothing is installed at run time.
#   EVALSCOPE_SWE_BENCH_PRO_REPO  a SWE-bench_Pro-os checkout at EvalScope's pinned commit, used
#                    instead of cloning one into ~/.cache/evalscope (the prebuilt image has one).
#   SANDBOX_K8S_*  --sandbox k8s pod settings (namespace, tolerations, owner pod, pull secret,
#                    ready timeout): see scripts/common/k8s_sandbox.py.
#   HF_TOKEN         if the dataset download needs one.
#
# SWE-bench_Pro also clones scaleapi/SWE-bench_Pro-os (pinned by EvalScope to ca10a60) for each
# instance's run script and parser, into ~/.cache/evalscope; it needs git and network access.
#
# SWE-bench Pro v1 vs V2: EvalScope implements v1 (731 tasks, jefzda/sweap-images, the v1 run
# scripts). Scale released V2 on 2026-09-22 (642 validated tasks shipped as Harbor tasks, images
# ghcr.io/scaleapi/swe-bench_pro-v2) and made it the dataset's default; v1 scoring against V2 rows
# fails even the reference patches. So this pins the dataset to its last v1 commit -- the scores
# are SWE-bench Pro v1 numbers, not comparable with V2 ones.
#
# Example (smoke test, 2 instances of each):
#   scripts/evalscope/run_evalscope_eval.sh --benchmark swe_bench,swe_bench_pro --model swiss-ai/Apertus-70B-Instruct-2509 \
#     --api-base-url https://api.swissai.cscs.ch/v1 --limit 2 --max-concurrency 2

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

die() { echo "$*" >&2; exit 1; }

usage() {
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}

BENCHMARK=""
MODEL=""
NAME=""
WORK_DIR=""
SANDBOX=""
GOLD=0
API_BASE_URL=${API_BASE_URL:-""}
RUN_ARGS=()

while (( $# > 0 )); do
    case "$1" in
        --benchmark) BENCHMARK=$2; shift 2 ;;
        --model) MODEL=$2; shift 2 ;;
        --name) NAME=$2; shift 2 ;;
        --work-dir) WORK_DIR=$2; shift 2 ;;
        --sandbox) SANDBOX=$2; shift 2 ;;
        --api-base-url) API_BASE_URL=$2; shift 2 ;;
        --gold) GOLD=1; shift ;;
        --shuffle) RUN_ARGS+=("$1"); shift ;;
        --sandbox-config|--limit|--max-concurrency|--max-steps|--strategy|--command-timeout|\
        --generation-config|--dataset-revision|--extra-param|--api-timeout)
            RUN_ARGS+=("$1" "$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown argument: $1 (see --help)" ;;
    esac
done

[[ -n "$BENCHMARK" ]] || die "Missing --benchmark (swe_bench, swe_bench_pro, or both). See --help."
IFS=',' read -ra BENCHMARK_LIST <<< "$BENCHMARK"
for b in "${BENCHMARK_LIST[@]}"; do
    [[ "$b" == swe_bench || "$b" == swe_bench_pro ]] || die "Unknown --benchmark '$b' (swe_bench or swe_bench_pro)."
done
if [[ "$GOLD" == "1" ]]; then
    [[ -z "$MODEL" ]] || die "--gold scores the reference patches; drop --model."
    MODEL_ARGS=(--gold)
else
    [[ -n "$MODEL" ]] || die "Missing --model. See --help."
    [[ -n "$API_BASE_URL" ]] || die "Missing --api-base-url. See --help."
fi

if [[ -z "$SANDBOX" ]]; then
    [[ -f /var/run/secrets/kubernetes.io/serviceaccount/token ]] && SANDBOX=k8s || SANDBOX=docker
fi
[[ "$SANDBOX" == docker || "$SANDBOX" == k8s ]] || die "--sandbox must be docker or k8s, got: $SANDBOX"

NAME=${NAME:-$(echo "${MODEL:-gold}" | tr '/: ' '---')}
WORK_DIR=${WORK_DIR:-"logs/evalscope/$NAME"}

if [[ "$GOLD" != "1" ]]; then
    # Same /v1 normalization as run_inspect_eval.sh.
    if [[ "$API_BASE_URL" != */v1 && "$API_BASE_URL" != */v1/ ]]; then
        API_BASE_URL="${API_BASE_URL%/}/v1"
    fi
    MODEL_ARGS=(--model "$MODEL" --api-base-url "${API_BASE_URL%/}")
fi

CSCS_SERVING_API="${CSCS_SERVING_API:-}"
if [[ -z "$CSCS_SERVING_API" && -f ./scripts/cscs_serving_api_key.txt ]]; then
    CSCS_SERVING_API="$(tr -d '\r\n' < ./scripts/cscs_serving_api_key.txt)"
fi
export EVALSCOPE_API_KEY="${TARGET_API_KEY:-$CSCS_SERVING_API}"

REQUIREMENTS=requirements/evalscope-runtime.txt
if [[ "${SKIP_INSTALL:-0}" == "1" ]]; then
    PY=python3
    echo "SKIP_INSTALL=1: running with $(command -v python3) as is (no venv, no install)"
elif [[ "${EVAL_PREBUILT_ENV:-0}" == "1" ]]; then
    PY="${EVALSCOPE_VENV:-}/bin/python"
    [[ -n "${EVALSCOPE_VENV:-}" && -x "$PY" ]] ||
        die "EVAL_PREBUILT_ENV=1 but EVALSCOPE_VENV (${EVALSCOPE_VENV:-unset}) has no bin/python:" \
            "this image has no EvalScope environment, and a prebuilt image installs nothing at run time."
    echo "EVAL_PREBUILT_ENV=1: using $EVALSCOPE_VENV (no install)"
else
    req_hash=$( (sha256sum "$REQUIREMENTS" 2>/dev/null || shasum -a 256 "$REQUIREMENTS") | cut -c1-12)
    EVALSCOPE_VENV=${EVALSCOPE_VENV:-"${XDG_CACHE_HOME:-$HOME/.cache}/evals-post-train/evalscope-$req_hash"}
    PY="$EVALSCOPE_VENV/bin/python"
    if [[ ! -f "$EVALSCOPE_VENV/.installed-$req_hash" ]]; then
        echo "Installing $REQUIREMENTS into $EVALSCOPE_VENV"
        if command -v uv >/dev/null 2>&1; then
            { uv venv --quiet --allow-existing --python ">=3.11" "$EVALSCOPE_VENV" &&
                uv pip install --quiet --python "$PY" -r "$REQUIREMENTS"; } ||
                die "Installing $REQUIREMENTS into $EVALSCOPE_VENV failed."
        else
            { python3 -m venv "$EVALSCOPE_VENV" &&
                "$PY" -m pip install --quiet --no-cache-dir -r "$REQUIREMENTS"; } ||
                die "Installing $REQUIREMENTS into $EVALSCOPE_VENV failed."
        fi
        touch "$EVALSCOPE_VENV/.installed-$req_hash"
    fi
fi

mkdir -p "$WORK_DIR"
CMD=("$PY" -m scripts.evalscope.run --benchmark "$BENCHMARK" "${MODEL_ARGS[@]}"
     --work-dir "$WORK_DIR" --sandbox "$SANDBOX"
     ${RUN_ARGS[@]+"${RUN_ARGS[@]}"})
echo "Running: ${CMD[*]}"
"${CMD[@]}"
echo ""
echo "Done. Scores and run metadata: $WORK_DIR/summary.json"
