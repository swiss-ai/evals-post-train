#!/bin/bash
# One-time KISTI HANGANG login-node setup for the medical expert evals in this folder (medbench.sbatch,
# healthbench.sbatch, hallubench.sbatch). Safe to re-run: each step checks for its result first. No job, no GPU:
# downloads and venvs only (UV_CONCURRENT_* and thread caps keep it under the login node's process limits).
#   bash scripts/medical_kisti/setup.sh            # everything
#   bash scripts/medical_kisti/setup.sh --check    # list what is missing
# Everything goes to $E (default /scratch/$USER/medical_evals/env; the group's /appsdata quota was full in Nov 2026):
#   uv/ python/       uv and its CPython builds (3.12, 3.13), usable on the host and inside the pyxis containers
#   venv313/          a2_gateway.py: apertus-common 494c2be [hf] (private swiss-ai repo: needs read access)
#   tokenizer/        Apertus_2_instruct as trained, copied from an SFT export (TOKENIZER_FROM)
#   src/              pinned checkouts: lm-evaluation-harness (public fork with the medbench *_g tasks),
#                     healthbench and hallubench (EPFLiGHT, private: ask for read access, or set HEALTHBENCH_SRC /
#                     HALLUBENCH_SRC to an existing checkout at the pinned commit)
#   lmeval/           lm-eval [api,ifeval] + CPU torch
#   hallubench/       Inspect 0.3.251 + hallubench
#   pydeps-vllm/      openai 2.44.0 for the image's vLLM server (the image's 2.7.2 breaks it)
#   models/gemma-4-31B-it   the judge/grader of HealthBench and hallubench (JUDGE_FROM: an existing copy to link)
#   data/healthbench  HealthBench full + Professional (healthbench/fetch_data.sh, OpenAI's public bucket)
#   data/hallubench   the four benchmark repos (hallubench/scripts/fetch_data.sh)
#   hf/ cache/        HF_HOME and XDG_CACHE_HOME for the jobs (datasets for lm-eval and Inspect, nltk data)
# Compute nodes have no internet: every dataset and model is fetched here.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
E=${MEDICAL_EVAL_ENV:-/scratch/$USER/medical_evals/env}
TOKENIZER_FROM=${TOKENIZER_FROM:-/appsdata/hanint05/hf-exports/bridge/chonk-lc-16k-v0.2-iter_0018315}
COMMON_COMMIT=494c2be79080aa8fbcd46409b96560f72707cd69
LMEVAL_URL=https://github.com/Xkrilandar/lm-evaluation-harness.git
LMEVAL_COMMIT=506c979fa8ae65f9cc6846e4c3ec13d385161964
HEALTHBENCH_URL=git@github.com:EPFLiGHT/healthbench.git
HEALTHBENCH_COMMIT=505c894d92cb4a6789f339c84f6a409799e4574f
HALLUBENCH_URL=git@github.com:EPFLiGHT/hallubench.git
HALLUBENCH_COMMIT=a8f08547dd522b97153de9a1633652bbc686a5d9
JUDGE_REPO=google/gemma-4-31B-it
export UV_CACHE_DIR=$E/uv-cache UV_PYTHON_INSTALL_DIR=$E/python
export UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export HF_HOME=$E/hf XDG_CACHE_HOME=$E/cache NLTK_DATA=$E/cache/nltk_data
[ -n "${HF_TOKEN:-}" ] || HF_TOKEN=$(tr -d '\n' < "$HOME/.huggingface_token" 2>/dev/null || true)
export HF_TOKEN
CHECK=false; [ "${1:-}" = --check ] && CHECK=true

# the lm-eval tasks medbench.sbatch runs; the HF cache is warmed by loading them
LMEVAL_TASKS=pubmedqa_g,medmcqa_g,medqa_g,medxpertqa_g,mmlu_pro_zs,ifeval

at_commit() { [ "$(git -C "$1" rev-parse HEAD 2>/dev/null)" = "$2" ]; }
todo=()
[ -x "$E/uv/bin/uv" ] || todo+=(uv)
[ -x "$E/venv313/bin/python" ] && "$E/venv313/bin/python" -I -c 'import apertus_common' 2>/dev/null || todo+=(gateway)
[ -f "$E/tokenizer/tokenizer.json" ] || todo+=(tokenizer)
at_commit "$E/src/lm-evaluation-harness" $LMEVAL_COMMIT && at_commit "$E/src/healthbench" $HEALTHBENCH_COMMIT \
    && at_commit "$E/src/hallubench" $HALLUBENCH_COMMIT || todo+=(sources)
"$E/lmeval/bin/python" -c 'import lm_eval, langdetect' 2>/dev/null || todo+=(lmeval)
"$E/hallubench/bin/python" -c 'import inspect_ai, hallubench' 2>/dev/null || todo+=(hallubench)
[ -d "$E/pydeps-vllm/openai" ] || todo+=(pydeps)
[ -f "$E/models/gemma-4-31B-it/config.json" ] || todo+=(judge)
[ -s "$E/data/healthbench/professional.jsonl" ] && [ -s "$E/data/healthbench/oss_eval_full.jsonl" ] || todo+=(healthbench-data)
[ -f "$E/data/hallubench/.ready-$HALLUBENCH_COMMIT" ] || todo+=(hallubench-data)
[ -f "$E/.lmeval-datasets-ready" ] || todo+=(lmeval-datasets)
ls -d "$XDG_CACHE_HOME"/inspect_ai/hf_datasets/bench-llm_or-bench-* >/dev/null 2>&1 || todo+=(inspect-datasets)

[ ${#todo[@]} -eq 0 ] && { echo "medical evals ready ($E)"; exit 0; }
$CHECK && { echo "medical evals not ready, missing: ${todo[*]} (run: bash $HERE/setup.sh)"; exit 1; }
echo "medical eval setup: ${todo[*]}"
has() { [[ " ${todo[*]} " == *" $1 "* ]]; }
mkdir -p "$E/src" "$E/models" "$HF_HOME" "$XDG_CACHE_HOME"

[ -x "$E/uv/bin/uv" ] || /usr/bin/python3.12 -m pip install -q --target "$E/uv" uv==0.13.6
uv() { PYTHONPATH="$E/uv" "$E/uv/bin/uv" "$@"; }
uv python install -q 3.12 3.13

checkout() {  # checkout <dir> <url> <commit> [existing checkout to copy from]
    local d=$1 url=$2 commit=$3 from=${4:-}
    at_commit "$d" "$commit" && return 0
    if [ ! -d "$d/.git" ]; then
        if [ -n "$from" ]; then git clone -q "$from" "$d"; else git clone -q "$url" "$d"; fi
    fi
    git -C "$d" fetch -q origin 2>/dev/null || true
    git -C "$d" checkout -q "$commit"
}
if has sources; then
    checkout "$E/src/lm-evaluation-harness" $LMEVAL_URL $LMEVAL_COMMIT
    checkout "$E/src/healthbench" $HEALTHBENCH_URL $HEALTHBENCH_COMMIT "${HEALTHBENCH_SRC:-}"
    checkout "$E/src/hallubench" $HALLUBENCH_URL $HALLUBENCH_COMMIT "${HALLUBENCH_SRC:-}"
fi
if has gateway; then
    [ -x "$E/venv313/bin/python" ] || uv venv -q --python 3.13 "$E/venv313"
    uv pip install -q --python "$E/venv313/bin/python" numpy \
        "apertus-common[hf] @ git+https://github.com/swiss-ai/apertus-common@$COMMON_COMMIT"
fi
if has tokenizer; then
    # byte-identical to apertus-omni-tokenizer Apertus_2_instruct @ b801ff1 in the Chonk SFT exports
    mkdir -p "$E/tokenizer"
    cp "$TOKENIZER_FROM"/{tokenizer.json,tokenizer_config.json,special_tokens_map.json,generation_config.json} "$E/tokenizer/"
fi
if has lmeval; then
    [ -x "$E/lmeval/bin/python" ] || uv venv -q --python 3.12 "$E/lmeval"
    uv pip install -q --python "$E/lmeval/bin/python" --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match torch
    uv pip install -q --python "$E/lmeval/bin/python" -e "$E/src/lm-evaluation-harness[api,ifeval]" "datasets>=2.19,<4" \
        transformers sentencepiece huggingface_hub
fi
# datasets < 4 (PubMedQA's bigbio loader is a dataset script, which 4.x refuses), but MedXpertQA's Hub metadata now
# uses 4.x's List feature type, unknown to 3.x: register it as Sequence (the 3.x name of the same list feature).
SITECUSTOM=$("$E/lmeval/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/sitecustomize.py
if ! grep -q '"List"' "$SITECUSTOM" 2>/dev/null; then
    cat > "$SITECUSTOM" <<'PY'
# medical_kisti/setup.sh: datasets 3.x reads Hub metadata written by datasets 4 (feature type "List" = 3.x Sequence)
try:
    import datasets.features.features as _f
    _f._FEATURE_TYPES.setdefault("List", _f.Sequence)
except Exception:
    pass
PY
fi
if has hallubench; then
    [ -x "$E/hallubench/bin/python" ] || uv venv -q --python 3.12 "$E/hallubench"
    uv pip install -q --python "$E/hallubench/bin/python" inspect_ai==0.3.251 agent-client-protocol==0.12.1 \
        "openai>=2.45,<3" datasets -e "$E/src/hallubench"
fi
if has pydeps; then
    /usr/bin/python3.12 -m pip install -q --no-deps --no-compile --target "$E/pydeps-vllm" openai==2.44.0
fi
if has judge; then
    if [ -n "${JUDGE_FROM:-}" ]; then
        ln -sfn "$JUDGE_FROM" "$E/models/gemma-4-31B-it"
    else
        "$E/lmeval/bin/python" -c 'import sys; from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2], max_workers=4)' $JUDGE_REPO "$E/models/gemma-4-31B-it"
    fi
fi
if has healthbench-data; then
    bash "$E/src/healthbench/fetch_data.sh" "$E/data/healthbench"
fi
if has hallubench-data; then
    bash "$E/src/hallubench/scripts/fetch_data.sh" "$E/data/hallubench"
    rm -f "$E/data/hallubench"/.ready-*; touch "$E/data/hallubench/.ready-$HALLUBENCH_COMMIT"
fi
if has inspect-datasets; then
    HALLUBENCH_DATA_DIR=$E/data/hallubench "$E/hallubench/bin/python" -c '
from hallubench.tasks.or_bench import or_bench_hard_1k, or_bench_toxic
from hallubench.tasks.aa_omniscience import aa_omniscience
or_bench_hard_1k(); or_bench_toxic(); aa_omniscience(domain="Health")'
fi
if has lmeval-datasets; then
    HF_DATASETS_CACHE=$E/hf/datasets-lmeval HF_DATASETS_TRUST_REMOTE_CODE=1 "$E/lmeval/bin/python" - "$LMEVAL_TASKS" "$HERE/lm_eval_tasks" <<'PY'
import sys, nltk
from lm_eval.tasks import TaskManager, get_task_dict
tasks, include = sys.argv[1].split(","), sys.argv[2]
for pkg in ("punkt", "punkt_tab"):
    nltk.download(pkg, quiet=True)
d = get_task_dict(tasks, TaskManager(include_path=include))
print("lm-eval datasets cached:", len(d), "tasks/groups")
PY
    touch "$E/.lmeval-datasets-ready"
fi
echo "medical evals ready ($E)"
