#!/bin/bash
set -euo pipefail
umask 077
: "${EVAL_REGRESSION_ROOT:?Set a fresh permanent run directory}"
: "${EVAL_REPO_ROOT:?Set the staged checkout}"
: "${EVAL_CONTAINER_CONFIG:?Set the verified EDF}"
cd "$EVAL_REPO_ROOT"
source scripts/helpfulness/runtime_isolation.sh
export PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 WANDB_MODE=disabled
export HF_HOME="$EVAL_REGRESSION_ROOT/assets/huggingface"
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0
phase=runtime
record_status() {
    python3 - "$EVAL_REGRESSION_ROOT" "$phase" "$1" "$2" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
root, phase, status, code = sys.argv[1:]
Path(root, "workflow-status.json").write_text(json.dumps({
    "phase": phase, "status": status, "exit_code": int(code),
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "IFEval full released-response scorer replay; not checkpoint inference",
}, indent=2) + "\n")
PY
}
on_exit() {
    local code=$?
    if [[ "$code" -ne 0 ]]; then record_status failed "$code"; fi
}
trap on_exit EXIT
record_status running 0
export EVAL_HARNESS_REPO=swiss-ai/lm-evaluation-harness
EVAL_HARNESS_REF=$(python3 - <<'PY'
from scripts.helpfulness.native import harness_pin
revision = harness_pin()
if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
    raise SystemExit("Expected one immutable 40-hex Swiss AI harness revision")
print(revision)
PY
)
export EVAL_HARNESS_REF
python3 - "$EVAL_REGRESSION_ROOT/submission-preflight.json" <<'PY'
import os
from pathlib import Path
from scripts.helpfulness.cli import load_suite
from scripts.helpfulness.core import write_json
import sys
suite_path = Path(os.environ["EVAL_SUITE_FILE"])
suite = load_suite(suite_path)
names = [item["name"] for item in suite["benchmarks"]]
if set(names) != {"ifeval", "ifbench", "alpaca_eval_2_lc", "mt_bench", "sycophancy", "summeval"}:
    raise SystemExit("Expected the requested six-benchmark suite without TruthfulQA or SimpleQA")
write_json(Path(sys.argv[1]), {
    "harness_revision": os.environ["EVAL_HARNESS_REF"],
    "suite": str(suite_path), "benchmarks": names,
    "actual_job_scope": "ifeval_scorer_replay",
})
PY
python3 - "$EVAL_CONTAINER_CONFIG" > "$EVAL_REGRESSION_ROOT/image.sha256" <<'PY'
import hashlib, sys, tomllib
with open(sys.argv[1], "rb") as stream:
    image = tomllib.load(stream)["image"]
with open(image, "rb") as stream:
    print(hashlib.file_digest(stream, "sha256").hexdigest())
PY
export EVAL_ENV_CACHE_ROOT="$EVAL_REGRESSION_ROOT/runtime-cache"
export EVAL_ENV_MANIFEST="$EVAL_REGRESSION_ROOT/environment.sh"
export EVAL_HARNESS_REF LM_EVAL_BACKEND=openai
export EVAL_ENV_ISOLATED=true
export EVAL_RUNTIME_REQUIREMENTS="$EVAL_REPO_ROOT/requirements/helpfulness-runtime.txt"
bash scripts/lm_eval_harness/build_eval_env.sh
source "$EVAL_ENV_MANIFEST"
export PYTHONPATH="$EVAL_REPO_ROOT:$EVAL_HARNESS_OVERLAY"
python="$EVAL_BASE_ENV/bin/python"
"$python" -m scripts.helpfulness.check_runtime_isolation \
    > "$EVAL_REGRESSION_ROOT/runtime-isolation.json"
"$python" -m pip check > "$EVAL_REGRESSION_ROOT/pip-check.txt"
"$python" -m pip freeze --all > "$EVAL_REGRESSION_ROOT/requirements.frozen.txt"
"$python" - "$EVAL_REGRESSION_ROOT/runtime-fingerprint.json" <<'PY'
import sys
from pathlib import Path
from scripts.helpfulness import acceptance, native
from scripts.helpfulness.core import write_json
native.require_harness()
write_json(Path(sys.argv[1]), acceptance.fingerprint())
PY
phase=assets
record_status running 0
export HELPFULNESS_TEST_NLTK_DATA="$EVAL_REGRESSION_ROOT/assets/nltk"
export HELPFULNESS_TEST_ALPACA_DATA="$EVAL_REGRESSION_ROOT/assets/alpaca"
export NLTK_DATA="$HELPFULNESS_TEST_NLTK_DATA"
"$python" - <<'PY'
import os
from pathlib import Path
from scripts.helpfulness import native
native.prepare_assets(Path(os.environ["HELPFULNESS_TEST_NLTK_DATA"]))
native.prepare_alpaca_assets(Path(os.environ["HELPFULNESS_TEST_ALPACA_DATA"]))
PY
phase=implementation-tests
record_status running 0
"$python" - "$EVAL_REGRESSION_ROOT/implementation-tests.json" <<'PY'
import json, sys, unittest
from pathlib import Path
suite = unittest.defaultTestLoader.discover("tests", pattern="test_helpfulness*.py")
result = unittest.TextTestRunner(verbosity=1).run(suite)
Path(sys.argv[1]).write_text(json.dumps({
    "tests_run": result.testsRun, "failures": len(result.failures),
    "errors": len(result.errors), "skipped": result.skipped,
    "scientific_acceptance_performed": False,
}, indent=2) + "\n")
raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)
PY
phase=independent-original-fixtures
record_status running 0
"$python" -m scripts.helpfulness.prepare_ifeval_replay prepare \
    --directory "$EVAL_REGRESSION_ROOT/ifeval" --nltk-directory "$NLTK_DATA"
phase=scorer-replay
record_status running 0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$python" -m scripts.helpfulness.prepare_ifeval_replay run \
    --directory "$EVAL_REGRESSION_ROOT/ifeval"
record_status passed 0
