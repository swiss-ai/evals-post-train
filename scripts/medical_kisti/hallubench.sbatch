#!/bin/bash
#SBATCH --job-name=med-hallubench
#SBATCH --account=in0138
#SBATCH --partition=intTest
#SBATCH --comment=field=nlp;appl=vllm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --propagate=NONE
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#
# hallubench (Inspect AI) on an Apertus 2 (Chonk) HF export, KISTI HANGANG, two nodes: over-refusal (OR-Bench hard
# 1k + toxic), Health-ORSC, MedSafetyBench, AA-Omniscience Health and a Pokemon hallucination probe, judged by
# Gemma-4-31B-it without thinking. Task settings are hallubench's own (greedy except Pokemon's T 0.7).
#   node A  vLLM serves the export (TP1 DP4 expert parallel, as medbench.sbatch) behind a2_gateway.py
#           (Apertus 2 format, thinking THINKING_LEVEL)
#   node B  the judge on GPUs 0-1 (vLLM TP2, 32k context); Inspect runs on the host of node A
# THINKING_LEVEL (default medium, as our earlier Chonk runs; v0.2 then writes an empty think frame).
#
# Run setup.sh once on the login node, then submit from a /scratch directory:
#   cd /scratch/$USER && sbatch --export=ALL,NAME=<results tag>,MODEL=<HF export dir> ~/evals-post-train/scripts/medical_kisti/hallubench.sbatch
# Results: $ROOT/eval_results/hallubench/<NAME>/ (Inspect .eval logs + summary.csv); server logs and the gateway log in
# $ROOT/eval_results/hallubench/servers/.
set -uo pipefail
ulimit -c 0
: "${NAME:?set NAME, the results tag}" "${MODEL:?set MODEL to an Apertus 2 HF export}"
HERE=${MEDICAL_KISTI_DIR:-$(cd "$(dirname "$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=//p')")" && pwd)}
ROOT=${MEDICAL_EVAL_ROOT:-/scratch/$USER/medical_evals}
E=${MEDICAL_EVAL_ENV:-$ROOT/env}
IMAGE=${IMAGE:-/scratch/hanint02/images/nemo-rl-apertus2-cfcfede4-ucclfix.sqsh}
PYDEPS=$E/pydeps-vllm
JUDGE_DIR=$E/models/gemma-4-31B-it
JUDGE=google/gemma-4-31B-it
OUT=$ROOT/eval_results/hallubench
mkdir -p "$OUT/servers" "$ROOT/cache/vllm"
MEDICAL_EVAL_ENV=$E bash "$HERE/setup.sh" --check || exit 1
[ -f "$OUT/$NAME/summary.csv" ] && { echo "##### $NAME already done"; exit 0; }
HOSTS=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
[ "${#HOSTS[@]}" -ge 2 ] || { echo "!!! needs two nodes"; exit 1; }
A=${HOSTS[0]}; B=${HOSTS[1]}
echo "##### $(date) $NAME: $MODEL on ${HOSTS[*]}, thinking ${THINKING_LEVEL:-medium}"

# --- node B: judge (container, GPUs 0-1) ---
srun --ntasks=1 --nodes=1 --nodelist="$B" --overlap --network=disable_rdzv_get \
    --container-image="$IMAGE" --container-writable \
    --container-mounts=/scratch:/scratch,/appsdata:/appsdata --container-workdir="$ROOT" bash -s \
    > "$OUT/servers/judge.$SLURM_JOB_ID.log" 2>&1 <<EOF &
set -uo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=$PYDEPS:\${PYTHONPATH:-}
export NCCL_NET=Socket NCCL_PROTO='^LL128' VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false HF_HOME=$ROOT/cache/hf VLLM_CACHE_ROOT=$ROOT/cache/vllm-8001 XDG_CACHE_HOME=$ROOT/cache
export FLASHINFER_WORKSPACE_BASE=$ROOT/cache/flashinfer-8001 TRITON_CACHE_DIR=/tmp/triton-8001 PYTHONNOUSERSITE=1
exec vllm serve $JUDGE_DIR --served-model-name $JUDGE --port 8001 --tensor-parallel-size 2 --max-model-len 32768 \
    --gpu-memory-utilization 0.90 --seed 0 --default-chat-template-kwargs '{"enable_thinking": false}'
EOF
judge_pid=$!

# --- node A: the export on all 4 GPUs (container) + gateway (host) ---
srun --ntasks=1 --nodes=1 --nodelist="$A" --overlap --network=disable_rdzv_get \
    --container-image="$IMAGE" --container-writable \
    --container-mounts=/scratch:/scratch,/appsdata:/appsdata --container-workdir="$ROOT" bash -s \
    > "$OUT/servers/target_$NAME.$SLURM_JOB_ID.log" 2>&1 <<EOF &
set -uo pipefail
ulimit -c 0
export PYTHONPATH=$PYDEPS:\${PYTHONPATH:-}
export NCCL_NET=Socket NCCL_PROTO='^LL128' VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1 HF_DEACTIVATE_ASYNC_LOAD=1
export TOKENIZERS_PARALLELISM=false HF_HOME=$ROOT/cache/hf VLLM_CACHE_ROOT=$ROOT/cache/vllm-8100 XDG_CACHE_HOME=$ROOT/cache
export FLASHINFER_WORKSPACE_BASE=$ROOT/cache/flashinfer-8100 TRITON_CACHE_DIR=/tmp/triton-8100 PYTHONNOUSERSITE=1
exec vllm serve $MODEL --served-model-name chonk --port 8100 --seed 0 \
    --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
    --max-model-len 16384 --gpu-memory-utilization 0.88 --max-num-seqs 256 --max-num-batched-tokens 16384 \
    --trust-remote-code
EOF
target_pid=$!
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=4 \
    "$E/venv313/bin/python" -I "$HERE/a2_gateway.py" --upstream http://127.0.0.1:8100 \
    --tokenizer "$E/tokenizer" --port 8000 --model-path "$MODEL" --names "chonk,$NAME" --thinking "${THINKING_LEVEL:-medium}" \
    --log "$OUT/servers/gateway_$NAME.$SLURM_JOB_ID.jsonl" > "$OUT/servers/gateway_$NAME.$SLURM_JOB_ID.log" 2>&1 &
gw_pid=$!

wait_up() {  # wait_up <url> <pid> <log>
    local t0; t0=$(date +%s)
    until curl -sf "$1" > /dev/null; do
        kill -0 "$2" 2>/dev/null || { echo "!!! server for $1 died, see $3"; tail -30 "$3"; return 1; }
        [ $(( $(date +%s) - t0 )) -lt 3600 ] || { echo "!!! $1 not up after 60 min"; return 1; }
        sleep 15
    done
}
wait_up "http://$B:8001/v1/models" $judge_pid "$OUT/servers/judge.$SLURM_JOB_ID.log" || exit 1
wait_up http://127.0.0.1:8100/v1/models $target_pid "$OUT/servers/target_$NAME.$SLURM_JOB_ID.log" || exit 1
wait_up http://127.0.0.1:8000/health $gw_pid "$OUT/servers/gateway_$NAME.$SLURM_JOB_ID.log" || exit 1
echo "##### $(date) servers up"
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d \
  "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the first-line treatment of anaphylaxis?\"}],\"max_tokens\":256,\"temperature\":0}" \
  | head -c 1200; echo

export PATH=$E/hallubench/bin:$PATH PYTHONNOUSERSITE=1 HALLUBENCH_DATA_DIR=$E/data/hallubench
export HF_HOME=$E/hf XDG_CACHE_HOME=$E/cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export XDG_DATA_HOME=/tmp/inspect_data-$SLURM_JOB_ID
export JUDGE_BASE_URL=http://$B:8001/v1 JUDGE_API_KEY=none TARGET_BASE_URL=http://localhost:8000/v1 TARGET_API_KEY=none
mkdir -p "$OUT/$NAME"
status=0
if python "$E/src/hallubench/scripts/run_all.py" --model "openai-api/target/$NAME" --grader "openai-api/judge/$JUDGE" \
        --log-dir "$OUT/$NAME" --skip-noharm --connections "${HALLU_MAX_CONNECTIONS:-512}" \
        --max-samples "${HALLU_MAX_SAMPLES:-128}"; then
    (cd "$E/src/hallubench" && python scripts/summarize.py "$OUT/$NAME" --csv "$OUT/$NAME/summary.csv")
else
    echo "!!! $NAME: some tasks failed"; status=1
fi
echo "##### $(date) done, status $status"
kill $gw_pid $target_pid $judge_pid 2>/dev/null; wait 2>/dev/null
exit $status
