#!/bin/bash
#SBATCH --job-name=med-medbench
#SBATCH --account=in0138
#SBATCH --partition=intTest
#SBATCH --comment=field=nlp;appl=vllm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --propagate=NONE
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#
# Medical multiple choice + IFEval on an Apertus 2 (Chonk) HF export, KISTI HANGANG, one node.
# vLLM serves the export (NeMo-RL Apertus 2 image, TP1 DP4 expert parallel) behind a2_gateway.py, which encodes chats
# in the Apertus 2 format (system frame {"thinking": THINKING_LEVEL, "tools": []}) and returns the reply frame as the
# message content. lm-eval (local-chat-completions) runs pubmedqa_g, medmcqa_g, medqa_g, medxpertqa_g (the medbench
# step-by-step prompt ending in "Final Answer: X", scored on the reply), mmlu_pro_zs_health (instruction in the user
# turn, no few-shot; lm_eval_tasks/mmlu_pro_zs; the health subject only: all 14 subjects add 11k questions, about 2 h
# with thinking) and ifeval, at T 0.6 / top-p 0.95 (the Apertus thinking-mode convention),
# 8,192 new tokens with thinking (the gateway's budget is max(client max_tokens, 8,192)). Then claim_score.py scores
# the same samples on the claim frame (the verifiable-answer channel), from the gateway log.
#
# THINKING_LEVEL (default high): Chonk SFT v0.2 was trained on low and high only and thinks only at high; at medium
# it writes an empty think frame. The medical RL expert trained its MCQs at high.
#
# Run setup.sh once on the login node, then submit from a /scratch directory:
#   cd /scratch/$USER && sbatch --export=ALL,TAG=<results tag>,MODEL=<HF export dir> ~/evals-post-train/scripts/medical_kisti/medbench.sbatch
# TASKS overrides the task list (pass it in the environment, not inside --export: sbatch splits --export on commas).
# SMOKE=1: --limit 20 per task. CONCURRENT: requests in flight (384; the server holds 256 per GPU).
# Results: $ROOT/eval_results/medbench/<TAG>_<job>/ (lm-eval layout) and claim_score.json; server logs and the gateway
# log (every request with its think, claim and reply) in $ROOT/runs/<job>/.
set -uo pipefail
ulimit -c 0
: "${MODEL:?set MODEL to an Apertus 2 HF export}" "${TAG:?set TAG, the results tag}"
HERE=${MEDICAL_KISTI_DIR:-$(cd "$(dirname "$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=//p')")" && pwd)}
ROOT=${MEDICAL_EVAL_ROOT:-/scratch/$USER/medical_evals}
E=${MEDICAL_EVAL_ENV:-$ROOT/env}
RUN=$ROOT/runs/$SLURM_JOB_ID
OUT=$ROOT/eval_results/medbench/${TAG}_$SLURM_JOB_ID
IMAGE=${IMAGE:-/scratch/hanint02/images/nemo-rl-apertus2-cfcfede4-ucclfix.sqsh}
PYDEPS=$E/pydeps-vllm
SMOKE=${SMOKE:-0}
mkdir -p "$RUN" "$OUT" "$ROOT/cache/vllm"
MEDICAL_EVAL_ENV=$E bash "$HERE/setup.sh" --check || exit 1
NAME=$(basename "$MODEL")
echo "##### $(date) $MODEL as $TAG on $(hostname), thinking ${THINKING_LEVEL:-high}; run dir $RUN"

# --- vLLM on all 4 GPUs (container) ---
srun --ntasks=1 --overlap --network=disable_rdzv_get --container-image="$IMAGE" --container-writable \
    --container-mounts=/scratch:/scratch,/appsdata:/appsdata --container-workdir="$ROOT" bash -s > "$RUN/vllm.log" 2>&1 <<EOF &
set -uo pipefail
ulimit -c 0
export PYTHONPATH=$PYDEPS:\${PYTHONPATH:-}
export NCCL_NET=Socket NCCL_PROTO='^LL128' VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1 HF_DEACTIVATE_ASYNC_LOAD=1
export TOKENIZERS_PARALLELISM=false HF_HOME=$ROOT/cache/hf VLLM_CACHE_ROOT=$ROOT/cache/vllm XDG_CACHE_HOME=$ROOT/cache
export FLASHINFER_WORKSPACE_BASE=$ROOT/cache/flashinfer TRITON_CACHE_DIR=/tmp/triton PYTHONNOUSERSITE=1
exec vllm serve $MODEL --served-model-name chonk --port 8100 --seed 0 \
    --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel --max-model-len 16384 \
    --gpu-memory-utilization 0.88 --max-num-seqs 256 --max-num-batched-tokens 16384 --trust-remote-code
EOF
VLLM=$!
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=4 \
    "$E/venv313/bin/python" -I "$HERE/a2_gateway.py" --upstream http://127.0.0.1:8100 \
    --tokenizer "$E/tokenizer" --port 8000 --model-path "$MODEL" --names "chonk,$NAME" --thinking "${THINKING_LEVEL:-high}" \
    --log "$RUN/gateway.jsonl" --log-raw > "$RUN/gateway.log" 2>&1 &
GW=$!
t0=$(date +%s)
until curl -sf http://127.0.0.1:8000/health > /dev/null; do
    kill -0 $VLLM 2>/dev/null || { echo "!!! vLLM died, see $RUN/vllm.log"; tail -40 "$RUN/vllm.log"; exit 1; }
    kill -0 $GW 2>/dev/null || { echo "!!! gateway died"; tail -20 "$RUN/gateway.log"; exit 1; }
    [ $(( $(date +%s) - t0 )) -lt $(( ${STARTUP_MIN:-60} * 60 )) ] || { echo "!!! not up after ${STARTUP_MIN:-60} min"; exit 1; }
    sleep 15
done
echo "##### $(date) endpoint up after $(( $(date +%s) - t0 )) s"
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d \
  '{"model":"chonk","messages":[{"role":"user","content":"What is the first-line treatment of anaphylaxis?"}],"max_tokens":512,"temperature":0.6,"top_p":0.95}' \
  | head -c 1500; echo

export HF_HOME=$E/hf XDG_CACHE_HOME=$E/cache NLTK_DATA=$E/cache/nltk_data HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_DATASETS_CACHE=$E/hf/datasets-lmeval
export OPENAI_API_KEY=none OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
LIMIT=(); [ "$SMOKE" = 1 ] && LIMIT=(--limit 20)
TASKS=${TASKS:-pubmedqa_g,medmcqa_g,medqa_g,medxpertqa_g,mmlu_pro_zs_health,ifeval}
echo "##### $(date) lm-eval: $TASKS"
"$E/lmeval/bin/python" -m lm_eval --model local-chat-completions --apply_chat_template \
    --model_args "model=$NAME,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=${CONCURRENT:-384},max_retries=3,timeout=7200,max_gen_toks=4096,tokenized_requests=False" \
    --gen_kwargs temperature=0.6,top_p=0.95,do_sample=True \
    --tasks "$TASKS" --batch_size 1 --log_samples --output_path "$OUT" --include_path "$HERE/lm_eval_tasks" --seed 42 \
    ${LIMIT[@]+"${LIMIT[@]}"}
status=$?
echo "##### $(date) lm-eval exit $status"
"$E/lmeval/bin/python" "$HERE/claim_score.py" "$OUT" "$RUN/gateway.jsonl" --out "$OUT/claim_score.json"
kill $GW $VLLM 2>/dev/null; wait 2>/dev/null
exit $status
