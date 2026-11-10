#!/bin/bash
set -euo pipefail
: "${EVAL_REPO_ROOT:?Set the absolute cluster evals-post-train checkout path}"
cd "$EVAL_REPO_ROOT"
export PYTHONNOUSERSITE=1 WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

if [[ -n "${EVAL_ENV_MANIFEST:-}" ]]; then
    [[ -r "$EVAL_ENV_MANIFEST" ]] || { echo "Missing environment manifest" >&2; exit 2; }
    source "$EVAL_ENV_MANIFEST"
    if [[ "${EVAL_ENV_ISOLATED:-false}" == true ]]; then
        source scripts/helpfulness/runtime_isolation.sh
    fi
    : "${EVAL_BASE_ENV:?Prepared manifest has no base environment}"
    : "${EVAL_HARNESS_OVERLAY:?Prepared manifest has no harness overlay}"
    [[ -f "$EVAL_BASE_ENV/.complete" && -f "$EVAL_HARNESS_OVERLAY/.complete" ]] \
        || { echo "Prepared runtime/overlay is incomplete or not mounted" >&2; exit 2; }
    EVAL_PYTHON="$EVAL_BASE_ENV/bin/python"
    export PYTHONPATH="$EVAL_REPO_ROOT:$EVAL_HARNESS_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
else
    : "${EVAL_PYTHON:?Set an absolute prepared container Python path or EVAL_ENV_MANIFEST}"
    export PYTHONPATH="$EVAL_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi
[[ "$EVAL_PYTHON" == /* && -x "$EVAL_PYTHON" ]] \
    || { echo "EVAL_PYTHON must be an executable absolute container path" >&2; exit 2; }
"$EVAL_PYTHON" -c 'import sys; assert sys.version_info >= (3, 11), "Python >=3.11 required"'

mode=${1:-}
case "$mode" in
    command)
        shift
        exec "$EVAL_PYTHON" -m scripts.helpfulness.acceptance "$@"
        ;;
    replay|regenerate|calibrate) ;;
    *)
        echo "Usage: $0 replay CONFIG OUTPUT | {regenerate|calibrate} CONFIG OUTPUT ENDPOINT" >&2
        echo "       $0 command <acceptance CLI arguments>" >&2
        exit 2
        ;;
esac
expected=3
[[ "$mode" == replay ]] || expected=4
[[ $# -eq "$expected" ]] || { echo "Wrong argument count for $mode" >&2; exit 2; }
config=$2
output=$3
[[ -r "$config" ]] || { echo "Unreadable acceptance configuration: $config" >&2; exit 2; }
[[ "$output" == /* ]] || { echo "Use an absolute permanent output path" >&2; exit 2; }
if [[ "$mode" != replay && ! -r "$4" ]]; then
    echo "Unreadable endpoint configuration: $4" >&2
    exit 2
fi
# Atomic reservation prevents two queued jobs from writing the same experiment.
mkdir -p "$(dirname "$output")"
mkdir "$output"
plan="$output/plan.json"
"$EVAL_PYTHON" -m scripts.helpfulness.acceptance plan --config "$config" --output "$plan"
"$EVAL_PYTHON" -c '
import json, sys
expected = {"replay": "scorer_replay", "regenerate": "reference_regeneration",
            "calibrate": "judge_calibration"}[sys.argv[1]]
with open(sys.argv[2]) as stream:
    actual = json.load(stream)["kind"]
if actual != expected:
    raise SystemExit(f"Requested {expected}, configuration declares {actual}")
' "$mode" "$plan"
candidate=()
if [[ "$mode" != replay ]]; then
    producer=regenerate
    [[ "$mode" != calibrate ]] || producer=judge
    "$EVAL_PYTHON" -m scripts.helpfulness.acceptance "$producer" \
        --plan "$plan" --endpoint "$4" --output "$output/candidate.json"
    candidate=(--candidate "$output/candidate.json")
fi
"$EVAL_PYTHON" -m scripts.helpfulness.acceptance run \
    --plan "$plan" "${candidate[@]}" --output "$output/report"
if [[ "$mode" == calibrate ]]; then
    "$EVAL_PYTHON" -m scripts.helpfulness.acceptance freeze \
        --report "$output/report/report.json" --output "$output/judge-lock.json"
fi
