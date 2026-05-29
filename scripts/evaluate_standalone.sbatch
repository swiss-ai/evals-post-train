#!/bin/bash
#SBATCH --account=a127
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --job-name=standalone-evaluation
#SBATCH --mem=64000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=normal
#SBATCH --time=02:00:00

# evaluate_standalone.sbatch - Run normalized standalone benchmarks under CSCS CE.
#
# Usage: sbatch evaluate_standalone.sbatch <model> <name>
#
# The batch script itself intentionally does not use #SBATCH --environment.
# Per CSCS CE guidance, each payload step enters the EDF through srun
# --environment to avoid nested or non-host execution surprises.

set -euo pipefail

usage() {
    echo "Usage: sbatch evaluate_standalone.sbatch <model> <name>"
    echo "Required env: STANDALONE_TASKS"
}

if (( $# != 2 )); then
    usage
    exit 1
fi

MODEL=$1
NAME=$2

LOGS_ROOT=${LOGS_ROOT:-/capstor/store/cscs/swissai/infra01/eval-logs}
WANDB_ENTITY=${WANDB_ENTITY:-apertus}
WANDB_PROJECT=${WANDB_PROJECT:-swissai-evals-standalone}
TABLE_METRICS=${TABLE_METRICS:-configs/standalone/smoke_main_table.txt}
STANDALONE_TASKS=${STANDALONE_TASKS:-}
STANDALONE_EDF=${STANDALONE_EDF:-./containers/env.toml}
STANDALONE_LIMIT=${STANDALONE_LIMIT:-}
SANDBOX_BACKEND=${SANDBOX_BACKEND:-none}
CONTAINER_CACHE_BACKEND=${CONTAINER_CACHE_BACKEND:-none}
LOCAL_REGISTRY_HOME=${LOCAL_REGISTRY_HOME:-}
LOCAL_REGISTRY_DIR=${LOCAL_REGISTRY_DIR:-${RUN_ROOT:-/tmp}/local_registry_${SLURM_JOBID:-local}}
PODMAN_SERVICE_PORT=${PODMAN_SERVICE_PORT:-}
PODMAN_SERVICE_HOST=${PODMAN_SERVICE_HOST:-}
PODMAN_SERVICE_USE_EXTERNAL=${PODMAN_SERVICE_USE_EXTERNAL:-false}
PODMAN_SERVICE_STORAGE_OPTS=${PODMAN_SERVICE_STORAGE_OPTS:-ignore_chown_errors=true}
SWE_BENCH_FAST_AUTO_BUILD=${SWE_BENCH_FAST_AUTO_BUILD:-true}
export SWE_USE_PODMAN_BUILD=${SWE_USE_PODMAN_BUILD:-true}
export SWE_PODMAN_BUILD_STORAGE_OPTS=${SWE_PODMAN_BUILD_STORAGE_OPTS:-ignore_chown_errors=true}
detect_swe_arch() {
    case "$(uname -m)" in
        aarch64|arm64) echo arm64 ;;
        x86_64|amd64) echo x86_64 ;;
        *) uname -m ;;
    esac
}
export SWE_ARCH=${SWE_ARCH:-$(detect_swe_arch)}
export SWE_RELAX_CONDA_BUILDS=${SWE_RELAX_CONDA_BUILDS:-auto}
export SWE_RELAX_CONDA_PACKAGE_PINS=${SWE_RELAX_CONDA_PACKAGE_PINS:-setuptools pip python}
LM_EVAL_BACKEND=${LM_EVAL_BACKEND:-vllm}
BS=${BS:-auto:20}
BOS=${BOS:-false}
MAX_LENGTH=${MAX_LENGTH:-4096}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-true}
REVISION=${REVISION:-}
VLLM_MEMORY_FRACTION=${VLLM_MEMORY_FRACTION:-0.75}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
SWE_WORKER_CPUS=${SLURM_CPUS_PER_TASK:-4}
SWE_DEFAULT_MAX_WORKERS=$(( (SWE_WORKER_CPUS * 3 + 3) / 4 ))
if (( SWE_DEFAULT_MAX_WORKERS < 1 )); then
    SWE_DEFAULT_MAX_WORKERS=1
fi
export SWE_MAX_WORKERS=${SWE_MAX_WORKERS:-$SWE_DEFAULT_MAX_WORKERS}

export WANDB_API_KEY="${WANDB_API_KEY:-$(tr -d '\r\n' < ./scripts/wandb_api_key.txt 2>/dev/null || true)}"
export HF_TOKEN="${HF_TOKEN:-$(tr -d '\r\n' < ./scripts/hf_token.txt 2>/dev/null || true)}"

if [[ -z "$STANDALONE_TASKS" ]]; then
    echo "ERROR: Set STANDALONE_TASKS."
    exit 1
fi

if [[ -z "${TOKENIZER:-}" ]]; then
    TOKENIZER=$MODEL
fi

IS_SWEBENCH=false
if [[ ",$STANDALONE_TASKS," == *",swebench_verified,"* ]]; then
    IS_SWEBENCH=true
fi

find_swe_bench_fast_source_dir() {
    local candidate
    for candidate in \
        "${SWE_BENCH_FAST_SOURCE_DIR:-}" \
        "swe-bench-fast-main" \
        "swe-bench-fast" \
        "external/swe-bench-fast"; do
        if [[ -n "$candidate" && -f "$candidate/go.mod" && -f "$candidate/cmd/swe-bench-fast/main.go" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

absolute_path() {
    if [[ "$1" == /* ]]; then
        echo "$1"
    else
        echo "$PWD/$1"
    fi
}

find_swe_bench_fast_bin() {
    local source_dir candidate
    for candidate in \
        "${SWE_BENCH_FAST_BIN:-}" \
        "swe-bench-fast-main/dist/swe-bench-fast" \
        "swe-bench-fast-main/bin/swe-bench-fast" \
        "swe-bench-fast/dist/swe-bench-fast" \
        "swe-bench-fast/bin/swe-bench-fast" \
        "external/swe-bench-fast/dist/swe-bench-fast" \
        "external/swe-bench-fast/bin/swe-bench-fast"; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    if source_dir=$(find_swe_bench_fast_source_dir); then
        echo "$source_dir/dist/swe-bench-fast"
        return 0
    fi
    if command -v swe-bench-fast >/dev/null 2>&1; then
        command -v swe-bench-fast
        return 0
    fi
    return 1
}

configure_swe_bench_fast_defaults() {
    local source_dir fast_bin
    if fast_bin=$(find_swe_bench_fast_bin); then
        export SWE_BENCH_FAST_BIN="$(absolute_path "${SWE_BENCH_FAST_BIN:-$fast_bin}")"
    fi

    if [[ -z "${SWE_EVALUATOR:-}" ]]; then
        if [[ -n "${SWE_BENCH_FAST_BIN:-}" || -n "${SWE_BENCH_FAST_SOURCE_DIR:-}" ]] || find_swe_bench_fast_source_dir >/dev/null 2>&1; then
            export SWE_EVALUATOR=fast
        else
            export SWE_EVALUATOR=official
        fi
    fi

    if [[ "$SWE_EVALUATOR" == "fast" && ! -x "${SWE_BENCH_FAST_BIN:-}" ]]; then
        if [[ "$SWE_BENCH_FAST_AUTO_BUILD" != "true" ]]; then
            echo "ERROR: SWE_EVALUATOR=fast but SWE_BENCH_FAST_BIN is not executable: ${SWE_BENCH_FAST_BIN:-<unset>}" >&2
            exit 1
        fi
        source_dir=$(find_swe_bench_fast_source_dir) || {
            echo "ERROR: SWE_EVALUATOR=fast requires swe-bench-fast source or binary." >&2
            echo "Set SWE_BENCH_FAST_BIN, or place the checkout at swe-bench-fast-main, swe-bench-fast, or external/swe-bench-fast." >&2
            exit 1
        }
        echo "Building swe-bench-fast from $source_dir"
        make -C "$source_dir" build
        export SWE_BENCH_FAST_BIN="$(absolute_path "$source_dir")/dist/swe-bench-fast"
    fi
}

if [[ "$IS_SWEBENCH" == "true" ]]; then
    configure_swe_bench_fast_defaults
else
    export SWE_EVALUATOR=${SWE_EVALUATOR:-official}
fi

COMMON_MODEL_ARGS="dtype=bfloat16"
if [[ $BOS = true ]]; then
    COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,add_bos_token=True"
fi
if [[ -n "$REVISION" ]]; then
    MAYBE_REVISION=",revision=$REVISION"
else
    MAYBE_REVISION=""
fi

if [[ $LM_EVAL_BACKEND == "vllm" ]]; then
    COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,pretrained=$MODEL,tokenizer=$TOKENIZER$MAYBE_REVISION,tensor_parallel_size=${GPUS_PER_NODE:-4},gpu_memory_utilization=$VLLM_MEMORY_FRACTION,enable_thinking=False"
elif [[ $LM_EVAL_BACKEND == "hf" ]]; then
    COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,pretrained=$MODEL,tokenizer=$TOKENIZER$MAYBE_REVISION"
else
    echo "ERROR: Unsupported standalone LM_EVAL_BACKEND: $LM_EVAL_BACKEND" >&2
    exit 1
fi

export LM_EVAL_MODEL_ARGS=${LM_EVAL_MODEL_ARGS:-$COMMON_MODEL_ARGS}
export MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-32}

LM_EVAL_HARNESS_PIP_SPEC="git+https://github.com/ymetz/lm-evaluation-harness.git"
if [[ -n "${LM_EVAL_HARNESS_BRANCH:-}" ]]; then
    LM_EVAL_HARNESS_PIP_SPEC="${LM_EVAL_HARNESS_PIP_SPEC}@${LM_EVAL_HARNESS_BRANCH}"
fi

INSTALL_CMD="pip install --no-cache-dir --upgrade \"$LM_EVAL_HARNESS_PIP_SPEC\" && \
pip install --no-cache-dir --upgrade \
    \"huggingface-hub>=1.3.0,<2.0\" \
    \"transformers==5.6.2\" \
    \"accelerate>=1.1.0\" \
    \"tokenizers==0.22.2\" \
    datasets \
    docker \
    unidiff \
    python-dotenv \
    tqdm \
    modal \
    tenacity \
    beautifulsoup4 \
    chardet \
    ghapi \
    GitPython \
    rich"

if [ -f "$TABLE_METRICS" ]; then
    echo "Reading table metrics from file: $TABLE_METRICS"
    TABLE_METRICS=$(grep -v '^\s*#' "$TABLE_METRICS" | grep -v '^\s*$' | paste -sd' ' -)
fi

RUN_ROOT=$LOGS_ROOT/$WANDB_ENTITY/$WANDB_PROJECT/$NAME
STANDALONE_DIR=$RUN_ROOT/standalone
EVAL_DIR=$STANDALONE_DIR/eval_$(date +%Y%m%d_%H%M%S)_${SLURM_JOBID:-local}
mkdir -p "$EVAL_DIR"
export WANDB_DIR=$RUN_ROOT
if [[ "${LOCAL_REGISTRY_DIR}" == "/tmp/local_registry_"* || "${LOCAL_REGISTRY_DIR}" == "/tmp/local_registry_local" ]]; then
    LOCAL_REGISTRY_DIR="$RUN_ROOT/local_registry_${SLURM_JOBID:-local}"
fi

LOCAL_REGISTRY_STARTED=0
PODMAN_SERVICE_PID=""
cleanup_container_services() {
    if (( LOCAL_REGISTRY_STARTED == 1 )); then
        echo "Stopping local container registry"
        registry down || true
    fi
    if [[ -n "$PODMAN_SERVICE_PID" ]]; then
        echo "Stopping Podman API service: $PODMAN_SERVICE_PID"
        kill "$PODMAN_SERVICE_PID" >/dev/null 2>&1 || true
        wait "$PODMAN_SERVICE_PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup_container_services EXIT

choose_podman_service_port() {
    local base offset port
    base=$((24000 + (${SLURM_JOBID:-0} % 20000)))
    for offset in $(seq 0 199); do
        port=$((base + offset))
        if ! (: >/dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1; then
            echo "$port"
            return 0
        fi
    done
    return 1
}

choose_podman_service_host() {
    if [[ -n "$PODMAN_SERVICE_HOST" ]]; then
        echo "$PODMAN_SERVICE_HOST"
    elif command -v hostname >/dev/null 2>&1 && [[ -n "$(hostname -I 2>/dev/null | awk '{print $1}')" ]]; then
        hostname -I | awk '{print $1}'
    else
        hostname -f 2>/dev/null || hostname
    fi
}

start_podman_docker_api() {
    if [[ "$PODMAN_SERVICE_USE_EXTERNAL" == "true" && ( "${DOCKER_HOST:-}" == tcp://* || "${DOCKER_HOST:-}" == http://* || "${DOCKER_HOST:-}" == https://* ) ]]; then
        echo "Using existing DOCKER_HOST=$DOCKER_HOST"
        return 0
    fi
    if [[ -n "${DOCKER_HOST:-}" ]]; then
        echo "Ignoring preexisting DOCKER_HOST=$DOCKER_HOST; starting per-job Podman API service."
        unset DOCKER_HOST
        unset CONTAINER_HOST
    fi
    if ! command -v podman >/dev/null 2>&1; then
        echo "ERROR: SWE-bench requires a Docker-compatible API, but neither DOCKER_HOST nor podman is available." >&2
        exit 1
    fi

    if [[ -z "$PODMAN_SERVICE_PORT" ]]; then
        PODMAN_SERVICE_PORT=$(choose_podman_service_port) || {
            echo "ERROR: Could not find an available localhost port for podman system service." >&2
            exit 1
        }
    fi

    PODMAN_SERVICE_HOST=$(choose_podman_service_host)
    PODMAN_SERVICE_BIND_URI="tcp://0.0.0.0:${PODMAN_SERVICE_PORT}"
    PODMAN_SERVICE_URI="tcp://${PODMAN_SERVICE_HOST}:${PODMAN_SERVICE_PORT}"
    PODMAN_SERVICE_LOG="$EVAL_DIR/podman_service.log"
    echo "Starting Podman Docker-compatible API at $PODMAN_SERVICE_URI"
    PODMAN_SERVICE_CMD=(podman)
    if [[ -n "$PODMAN_SERVICE_STORAGE_OPTS" ]]; then
        read -r -a PODMAN_SERVICE_STORAGE_OPT_LIST <<< "$PODMAN_SERVICE_STORAGE_OPTS"
        for storage_opt in "${PODMAN_SERVICE_STORAGE_OPT_LIST[@]}"; do
            PODMAN_SERVICE_CMD+=(--storage-opt "$storage_opt")
        done
    fi
    PODMAN_SERVICE_CMD+=(system service --time=0 "$PODMAN_SERVICE_BIND_URI")
    "${PODMAN_SERVICE_CMD[@]}" >"$PODMAN_SERVICE_LOG" 2>&1 &
    PODMAN_SERVICE_PID=$!

    for _ in $(seq 1 60); do
        if curl --silent --fail "http://127.0.0.1:${PODMAN_SERVICE_PORT}/_ping" >/dev/null 2>&1; then
            export DOCKER_HOST="$PODMAN_SERVICE_URI"
            export CONTAINER_HOST="$PODMAN_SERVICE_URI"
            echo "Podman API service is ready: DOCKER_HOST=$DOCKER_HOST"
            return 0
        fi
        if ! kill -0 "$PODMAN_SERVICE_PID" >/dev/null 2>&1; then
            echo "ERROR: podman system service exited early. Log:" >&2
            cat "$PODMAN_SERVICE_LOG" >&2 || true
            exit 1
        fi
        sleep 1
    done

    echo "ERROR: Timed out waiting for podman system service. Log:" >&2
    cat "$PODMAN_SERVICE_LOG" >&2 || true
    exit 1
}

start_local_registry() {
    if [[ "$CONTAINER_CACHE_BACKEND" != "local_registry" ]]; then
        return 0
    fi
    if [[ -n "$LOCAL_REGISTRY_HOME" ]]; then
        if [[ ! -f "$LOCAL_REGISTRY_HOME/env-registry" ]]; then
            echo "ERROR: LOCAL_REGISTRY_HOME does not contain env-registry: $LOCAL_REGISTRY_HOME" >&2
            exit 1
        fi
        # shellcheck disable=SC1090
        . "$LOCAL_REGISTRY_HOME/env-registry"
    elif ! command -v registry >/dev/null 2>&1; then
        echo "ERROR: CONTAINER_CACHE_BACKEND=local_registry requires either --local-registry-home or registry on PATH." >&2
        exit 1
    fi
    mkdir -p "$LOCAL_REGISTRY_DIR"
    LOCAL_REGISTRY="$(registry up "$LOCAL_REGISTRY_DIR")"
    export LOCAL_REGISTRY
    export SWE_USE_PODMAN_CACHED=${SWE_USE_PODMAN_CACHED:-true}
    LOCAL_REGISTRY_STARTED=1
    echo "Local container registry: $LOCAL_REGISTRY"
    registry status
}

RUN_ARGS=(
    --model "$MODEL"
    --name "$NAME"
    --output-dir "$EVAL_DIR"
    --sandbox-backend "$SANDBOX_BACKEND"
)

if [[ -n "$STANDALONE_TASKS" ]]; then
    RUN_ARGS+=(--tasks "$STANDALONE_TASKS")
fi
if [[ -n "$STANDALONE_LIMIT" ]]; then
    RUN_ARGS+=(--limit "$STANDALONE_LIMIT")
fi
RUN_CMD=(python -m scripts.standalone.run "${RUN_ARGS[@]}")
GEN_RUN_CMD=(python -m scripts.standalone.run "${RUN_ARGS[@]}" --phase generate)
printf -v RUN_CMD_STR '%q ' "${RUN_CMD[@]}"
printf -v GEN_RUN_CMD_STR '%q ' "${GEN_RUN_CMD[@]}"
printf -v WORKDIR_Q '%q' "$PWD"

HOST_PYTHON=${HOST_PYTHON:-python}
HOST_VENV=$EVAL_DIR/host_venv
HOST_SITE=$EVAL_DIR/host_python
HOST_EXTRA_PYTHONPATH=""
HOST_DEPS=(
    docker
    datasets
    unidiff
    python-dotenv
    tqdm
    modal
    tenacity
    beautifulsoup4
    chardet
    ghapi
    GitPython
    rich
)

install_host_python_deps() {
    if [[ ! -x "$HOST_VENV/bin/python" ]]; then
        if ! "$HOST_PYTHON" -m venv "$HOST_VENV"; then
            if command -v uv >/dev/null 2>&1; then
                mkdir -p "$HOST_SITE"
                uv pip install --python "$HOST_PYTHON" --upgrade --target "$HOST_SITE" "${HOST_DEPS[@]}"
                HOST_EXTRA_PYTHONPATH="$HOST_SITE"
                return 0
            fi
            echo "ERROR: Could not create host virtualenv at $HOST_VENV and uv is not on PATH." >&2
            echo "Set HOST_PYTHON to a Python with venv support, or provide uv on PATH." >&2
            exit 1
        fi
    fi
    HOST_PYTHON="$HOST_VENV/bin/python"

    if "$HOST_PYTHON" -m pip --version >/dev/null 2>&1; then
        "$HOST_PYTHON" -m pip install --upgrade pip
        "$HOST_PYTHON" -m pip install --upgrade "${HOST_DEPS[@]}"
    elif command -v uv >/dev/null 2>&1; then
        uv pip install --python "$HOST_PYTHON" --upgrade "${HOST_DEPS[@]}"
    else
        echo "ERROR: Could not install host Python dependencies: no pip in $HOST_VENV and uv is not on PATH." >&2
        echo "Set HOST_PYTHON to a Python with venv/pip support, or pre-create a virtualenv at $HOST_VENV." >&2
        exit 1
    fi
}

echo "START TIME: $(date)"
echo "Standalone configuration:"
echo "  MODEL=$MODEL"
echo "  NAME=$NAME"
echo "  STANDALONE_TASKS=$STANDALONE_TASKS"
echo "  STANDALONE_EDF=$STANDALONE_EDF"
echo "  SANDBOX_BACKEND=$SANDBOX_BACKEND"
echo "  CONTAINER_CACHE_BACKEND=$CONTAINER_CACHE_BACKEND"
echo "  LOCAL_REGISTRY_HOME=$LOCAL_REGISTRY_HOME"
echo "  LOCAL_REGISTRY_DIR=$LOCAL_REGISTRY_DIR"
echo "  PODMAN_SERVICE_STORAGE_OPTS=$PODMAN_SERVICE_STORAGE_OPTS"
echo "  DOCKER_HOST=${DOCKER_HOST:-}"
echo "  LM_EVAL_BACKEND=$LM_EVAL_BACKEND"
echo "  LM_EVAL_MODEL_ARGS=$LM_EVAL_MODEL_ARGS"
echo "  APPLY_CHAT_TEMPLATE=$APPLY_CHAT_TEMPLATE"
echo "  MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
if [[ "$IS_SWEBENCH" == "true" ]]; then
    echo "  SWE_EVALUATOR=$SWE_EVALUATOR"
    echo "  SWE_MAX_WORKERS=$SWE_MAX_WORKERS"
    echo "  SWE_CACHE_LEVEL=${SWE_CACHE_LEVEL:-env}"
    echo "  SWE_TIMEOUT=${SWE_TIMEOUT:-1800}"
    echo "  SWE_ARCH=$SWE_ARCH"
    echo "  SWE_RELAX_CONDA_BUILDS=$SWE_RELAX_CONDA_BUILDS"
    echo "  SWE_RELAX_CONDA_PACKAGE_PINS=$SWE_RELAX_CONDA_PACKAGE_PINS"
    echo "  SWE_USE_PODMAN_BUILD=$SWE_USE_PODMAN_BUILD"
    echo "  SWE_PODMAN_BUILD_STORAGE_OPTS=$SWE_PODMAN_BUILD_STORAGE_OPTS"
fi
echo "  EVAL_DIR=$EVAL_DIR"
echo "  W&B=$WANDB_ENTITY/$WANDB_PROJECT"

START_TIME=$(date +%s)

if [[ "$IS_SWEBENCH" == "true" ]]; then
    echo "SWE-bench phase 1/2: generating predictions inside $STANDALONE_EDF"
    srun -n 1 --mpi=pmix -ul --environment="$STANDALONE_EDF" --export=ALL,HF_TOKEN,WANDB_API_KEY,MODEL_API_BASE,MODEL_API_KEY,MODEL_API_MODEL,SANDBOX_BACKEND bash -lc "
        set -euo pipefail
        cd $WORKDIR_Q
        unset SSL_CERT_FILE
        unset DOCKER_HOST
        unset CONTAINER_HOST
        $INSTALL_CMD
        export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:\${LD_LIBRARY_PATH:-}
        $GEN_RUN_CMD_STR
    "

    echo "SWE-bench phase 2/2: running $SWE_EVALUATOR evaluator on host with Podman/Docker"
    start_podman_docker_api
    if [[ "$SWE_EVALUATOR" != "fast" ]]; then
        start_local_registry
    fi
    install_host_python_deps
    if [[ "$SWE_EVALUATOR" == "official" ]]; then
        export PYTHONPATH="$PWD:$PWD/swe-bench-reference${HOST_EXTRA_PYTHONPATH:+:$HOST_EXTRA_PYTHONPATH}:${PYTHONPATH:-}"
    else
        export PYTHONPATH="$PWD${HOST_EXTRA_PYTHONPATH:+:$HOST_EXTRA_PYTHONPATH}:${PYTHONPATH:-}"
    fi
    "$HOST_PYTHON" -c 'import docker, os; print("docker module", getattr(docker, "__file__", None), getattr(docker, "__path__", None)); assert hasattr(docker, "from_env"), "Imported docker module is not the Docker SDK"; client = docker.from_env(); print("Docker API reachable", client.version().get("ApiVersion"), "via", os.environ.get("DOCKER_HOST"))'
    "$HOST_PYTHON" -m scripts.standalone.run "${RUN_ARGS[@]}" --phase evaluate
else
    srun -n 1 --mpi=pmix -ul --environment="$STANDALONE_EDF" --export=ALL,HF_TOKEN,WANDB_API_KEY,MODEL_API_BASE,MODEL_API_KEY,MODEL_API_MODEL,SANDBOX_BACKEND bash -lc "
        set -euo pipefail
        cd $WORKDIR_Q
        unset SSL_CERT_FILE
        $INSTALL_CMD
        export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:\${LD_LIBRARY_PATH:-}
        $RUN_CMD_STR
    "
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Standalone evaluation finished in ${DURATION} seconds"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "Uploading standalone results to wandb"
    WANDB_ARGS=(
        python -m scripts.alignment.update_wandb_alignment
        --entity "$WANDB_ENTITY"
        --project "$WANDB_PROJECT"
        --logs_root "$EVAL_DIR"
        --name "$NAME"
        --main_metrics $TABLE_METRICS
        --eval_duration "$DURATION"
    )
    printf -v WANDB_CMD_STR '%q ' "${WANDB_ARGS[@]}"
    WANDB_CMD="cd $WORKDIR_Q && $WANDB_CMD_STR"
    echo "Running command to upload results to wandb:"
    echo "$WANDB_CMD"
    srun -n 1 --mpi=pmix -ul --environment="$STANDALONE_EDF" --export=ALL,WANDB_API_KEY bash -lc "$WANDB_CMD"
else
    echo "WANDB_API_KEY is empty; skipping W&B upload."
fi

echo "Results in: $EVAL_DIR"
echo "END TIME: $(date)"
