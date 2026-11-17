#!/bin/bash
#SBATCH --job-name=med-healthbench
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
# HealthBench Professional + HealthBench (full, 5,000 conversations) on an Apertus 2 (Chonk) HF export, KISTI HANGANG,
# two nodes, graded by Gemma-4-31B-it (not the paper's GPT-4.1; scores are on our grader's scale, compare within it).
#   node A  vLLM serves the export (TP1 DP4 expert parallel, as medbench.sbatch) behind a2_gateway.py
#           (Apertus 2 format, thinking THINKING_LEVEL, budget max(client max_tokens, 8,192))
#   node B  in the NeMo-RL Apertus 2 image: healthbench/run_subsets.py, answers from the endpoint at T 0.7, the
#           Gemma-4 grader in process (TP4, 32k context)
# THINKING_LEVEL (default low): the medical RL expert trained its open-ended (rubric) rows at low; Chonk SFT v0.2
# knows only low and high.
#
# Run setup.sh once on the login node, then submit from a /scratch directory:
#   cd /scratch/$USER && sbatch --export=ALL,TAG=<results tag>,MODEL=<HF export dir> ~/evals-post-train/scripts/medical_kisti/healthbench.sbatch
# SMOKE=1: Professional only, 20 examples (with --time=00:45:00). WORKERS: requests in flight (384).
# Results: $E/src/healthbench/results/<subset>_gemma4_31b/results_<TAG>_<subset>.json (healthbench writes next to
# itself), gateway log and job logs in $ROOT/runs/<job>/.
set -uo pipefail
ulimit -c 0
: "${MODEL:?set MODEL to an Apertus 2 HF export}" "${TAG:?set TAG, the results tag}"
HERE=${MEDICAL_KISTI_DIR:-$(cd "$(dirname "$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=//p')")" && pwd)}
ROOT=${MEDICAL_EVAL_ROOT:-/scratch/$USER/medical_evals}
E=${MEDICAL_EVAL_ENV:-$ROOT/env}
RUN=$ROOT/runs/$SLURM_JOB_ID
IMAGE=${IMAGE:-/scratch/hanint02/images/nemo-rl-apertus2-cfcfede4-ucclfix.sqsh}
PYDEPS=$E/pydeps-vllm
JUDGE=$E/models/gemma-4-31B-it
HB=$E/src/healthbench
WORKERS=${WORKERS:-384}
SMOKE=${SMOKE:-0}
mkdir -p "$RUN" "$ROOT/cache/vllm"
MEDICAL_EVAL_ENV=$E bash "$HERE/setup.sh" --check || exit 1
HOSTS=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
[ "${#HOSTS[@]}" -ge 2 ] || { echo "!!! needs two nodes"; exit 1; }
A=${HOSTS[0]}; B=${HOSTS[1]}
NAME=$(basename "$MODEL")
ENDPOINT=http://$A:8000
echo "##### $(date) $MODEL as $TAG, thinking ${THINKING_LEVEL:-low}: endpoint on $A, evals on $B; run dir $RUN"

# --- node A: vLLM (container) + gateway (host) ---
srun --ntasks=1 --nodes=1 --nodelist="$A" --overlap --network=disable_rdzv_get --container-image="$IMAGE" \
    --container-writable --container-mounts=/scratch:/scratch,/appsdata:/appsdata --container-workdir="$ROOT" \
    bash -s > "$RUN/vllm.log" 2>&1 <<EOF &
set -uo pipefail
ulimit -c 0
export PYTHONPATH=$PYDEPS:\${PYTHONPATH:-}
export NCCL_NET=Socket NCCL_PROTO='^LL128' VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1 HF_DEACTIVATE_ASYNC_LOAD=1
export TOKENIZERS_PARALLELISM=false HF_HOME=$ROOT/cache/hf VLLM_CACHE_ROOT=$ROOT/cache/vllm XDG_CACHE_HOME=$ROOT/cache
export FLASHINFER_WORKSPACE_BASE=$ROOT/cache/flashinfer TRITON_CACHE_DIR=/tmp/triton PYTHONNOUSERSITE=1
exec vllm serve $MODEL --served-model-name chonk --port 8100 --seed 0 \
    --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
    --max-model-len 16384 --gpu-memory-utilization 0.88 --max-num-seqs 256 --max-num-batched-tokens 16384 \
    --trust-remote-code
EOF
VLLM=$!
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=4 \
    "$E/venv313/bin/python" -I "$HERE/a2_gateway.py" --upstream http://127.0.0.1:8100 \
    --tokenizer "$E/tokenizer" --port 8000 --model-path "$MODEL" --names "chonk,$NAME" --thinking "${THINKING_LEVEL:-low}" \
    --log "$RUN/gateway.jsonl" > "$RUN/gateway.log" 2>&1 &
GW=$!
t0=$(date +%s)
until curl -sf "$ENDPOINT/health" > /dev/null; do
    kill -0 $VLLM 2>/dev/null || { echo "!!! vLLM died, see $RUN/vllm.log"; tail -40 "$RUN/vllm.log"; exit 1; }
    kill -0 $GW 2>/dev/null || { echo "!!! gateway died"; tail -20 "$RUN/gateway.log"; exit 1; }
    [ $(( $(date +%s) - t0 )) -lt $(( ${STARTUP_MIN:-60} * 60 )) ] || { echo "!!! not up after ${STARTUP_MIN:-60} min"; exit 1; }
    sleep 15
done
echo "##### $(date) endpoint up after $(( $(date +%s) - t0 )) s: $ENDPOINT"
curl -s "$ENDPOINT/v1/chat/completions" -H 'Content-Type: application/json' -d \
  '{"model":"chonk","messages":[{"role":"user","content":"What is the first-line treatment of anaphylaxis?"}],"max_tokens":512,"temperature":0.7}' \
  | head -c 1500; echo

# --- node B: HealthBench in the container ---
subs=professional,full; hbx=()
[ "$SMOKE" = 1 ] && subs=professional hbx=(--examples 20)
echo "##### $(date) HealthBench $subs"
srun --ntasks=1 --nodes=1 --nodelist="$B" --overlap --network=disable_rdzv_get --container-image="$IMAGE" \
    --container-writable --container-mounts=/scratch:/scratch,/appsdata:/appsdata --container-workdir="$HB" \
    env PYTHONPATH="$PYDEPS" NCCL_NET=Socket NCCL_PROTO='^LL128' NCCL_IB_DISABLE=1 FI_PROVIDER=cxi \
        VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
        HF_HOME=$ROOT/cache/hf VLLM_CACHE_ROOT=$ROOT/cache/vllm-b XDG_CACHE_HOME=$ROOT/cache \
        FLASHINFER_WORKSPACE_BASE=$ROOT/cache/flashinfer-b TRITON_CACHE_DIR=/tmp/triton-b PYTHONNOUSERSITE=1 \
        HEALTHBENCH_DATA_DIR=$E/data/healthbench OPENAI_API_KEY=none \
    /usr/bin/python3.12 run_subsets.py --model "$MODEL" --tag "$TAG" --subsets "$subs" -- \
        --grader "$JUDGE" --grader-max-model-len 32768 --tp 4 \
        --model-endpoint "$ENDPOINT" --endpoint-workers $WORKERS ${hbx[@]+"${hbx[@]}"} \
    > "$RUN/healthbench.log" 2>&1
status=$?
echo "##### $(date) HealthBench exit $status; results in $HB/results/"
kill $GW $VLLM 2>/dev/null; wait 2>/dev/null
exit $status
