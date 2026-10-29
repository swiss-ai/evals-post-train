# SwissAI Evaluation Pipeline

Evaluation infrastructure for benchmarking Large Language Models on SLURM clusters (CSCS Alps), with W&B integration for results tracking. It runs four evaluation frameworks, each with its own entry point and pinned environment:

| Framework | For | Entry point | Docs |
|---|---|---|---|
| [lm-evaluation-harness](https://github.com/swiss-ai/lm-evaluation-harness) | Standard benchmarks (MMLU, GSM8K, ...) on a checkpoint or an endpoint; serves the model in the job, resumable chunks | `scripts/lm_eval_harness/launch_evaluations.sh` | [docs/lm-eval-harness.md](docs/lm-eval-harness.md) |
| [Inspect AI](https://inspect.aisi.org.uk/) | Benchmarks with their own scorer, a judge or a simulated user (tau2-bench, HLE, AA-LCR, ...), against an endpoint | `scripts/inspect_ai/run_inspect_eval.sh` | [docs/inspect-ai.md](docs/inspect-ai.md) |
| [Harbor](https://github.com/harbor-framework/harbor) | Agentic coding and terminal benchmarks (SWE-bench, Terminal-Bench, ...): an agent in a task container, local Docker or Kubernetes pods | `scripts/harbor/run_harbor_eval.sh` | [docs/agentic.md](docs/agentic.md) |
| [EvalScope](https://github.com/modelscope/evalscope) | SWE-bench Verified / Pro, the alternative to Harbor | `scripts/evalscope/run_evalscope_eval.sh` | [docs/agentic.md](docs/agentic.md#evalscope-swe-bench-swe-bench-pro-the-alternative-to-harbor) |

Artificial Analysis's protocol versions of some benchmarks live in `scripts/aaii/` ([docs/inspect-ai.md](docs/inspect-ai.md)). Results go to W&B, Every Eval Ever / Hugging Face exports and tables: [docs/results.md](docs/results.md).

---

## Quick Start

This pipeline **launches evaluations** on the cluster (see [The Launch Script](docs/lm-eval-harness.md#the-launch-script)) and turns their W&B results into **tables** (see [Reporting: Building Result Tables](docs/results.md#reporting-building-result-tables)). For reasoning models, the `--thinking` option additionally makes the model reason and records the reasoning metrics — see [Thinking / Reasoning Metrics](docs/lm-eval-harness.md#thinking--reasoning-metrics).

```bash
# Evaluate a single model on the benchmark suite (with custom name)
bash scripts/lm_eval_harness/launch_evaluations.sh default --model meta-llama/Llama-3.1-8B-Instruct --name Llama-Baseline

# Resumable chunks of 8 tasks, with at most 4 chunks running concurrently
bash scripts/lm_eval_harness/launch_evaluations.sh default --model meta-llama/Llama-3.1-8B-Instruct \
  --chunk-size 8 --max-parallel 4

# Launch a Megatron checkpoint without conversion; --megatron-iter defaults to latest
bash scripts/lm_eval_harness/launch_evaluations.sh olmo-easy --model /capstor/store/../apertus-.../checkpoints/ --backend megatron_lm --name Megatron-Test-260216 --megatron-iter 4000000

# Convert a Megatron checkpoint to HF first, then evaluate it with vLLM (recommended
# over --backend megatron_lm above: supports vllm/sglang/hf and the thinking metrics).
# --megatron-iter is required (no "latest" auto-resolution for conversion). The converted
# checkpoint lands in /capstor/.../infra01/hf_models/models/swiss-ai/<name>-iter<N>/ by
# default (override with --hf-output-dir); reconversion is skipped if it already exists.
# Works for checkpoints trained with --ckpt-fully-parallel-save too (tools/checkpoint's
# converter needs swiss-ai/Megatron-LM's fix/convert-fully-parallel-save-checkpoints
# branch, or a merged equivalent, for those).
bash scripts/lm_eval_harness/launch_evaluations.sh single --task gsm8k \
  --model /capstor/store/cscs/swissai/infra01/apertus_checkpoints/v1p5/megatron/8B/pretraining \
  --backend vllm --convert-to-hf --megatron-iter 430000 --name Apertus-v1p5-8B-pretraining

# Run just the conversion (no eval) to test it standalone, e.g. against a checkpoint's
# iteration directory directly. CHECK_EQ_HF diffs the result against a known-good HF
# checkpoint tensor-by-tensor if you have one to validate against; jobs need
# --reservation (or SBATCH_RESERVATION=<name>) on Clariden's shared partition.
SBATCH_RESERVATION=<reservation> CHECK_EQ_HF=/capstor/store/cscs/swissai/infra01/apertus_checkpoints/v1p5/hf/8B/pretraining/iter_0430000 \
  sbatch scripts/lm_eval_harness/convert_megatron_checkpoint.sbatch \
  /capstor/store/cscs/swissai/infra01/apertus_checkpoints/v1p5/megatron/8B/pretraining/iter_0430000 \
  /iopsstor/scratch/cscs/$USER/Apertus-v1p5-8B-pretraining-iter430000

# Launch with vllm backend - recommended!
bash scripts/lm_eval_harness/launch_evaluations.sh olmo-easy --model /capstor/store/../apertus-.../checkpoints/ --backend vllm

# Launch with the SGLang backend
bash scripts/lm_eval_harness/launch_evaluations.sh olmo-easy --model /capstor/store/../apertus-.../checkpoints/ --backend sglang

# Evaluate against an already-running OpenAI-compatible endpoint (vLLM serve, CSCS serving, ...)
# instead of loading the model in the job
bash scripts/lm_eval_harness/launch_evaluations.sh single --task gsm8k_cot --model Qwen/Qwen3-8B \
  --backend openai --api-base-url http://nid001234:8000

# Evaluate a base model with 5-shot and easy eval set (matching OLMo3 technical report settings)
bash scripts/lm_eval_harness/launch_evaluations.sh olmo-easy --model Qwen/Qwen2.5-7B --num-fewshot 5 --no-chat-template

# Evaluate a small model on a single task, useful for testing newly implemented tasks
bash scripts/lm_eval_harness/launch_evaluations.sh single --task multijail --model meta-llama/Llama-3.2-3B --backend vllm

# Thinking / reasoning eval with an isolated W&B run name. The tokens are explicit because
# --thinking only auto-detects them from Apertus-style templates (see "Thinking / Reasoning Metrics")
bash scripts/lm_eval_harness/launch_evaluations.sh posttrain --model Qwen/Qwen3-8B \
  --thinking --think-start-token '<think>' --think-end-token '</think>' --name Qwen3-8B-think
# ...then build the thinking-only table from that run (details: "Building a thinking-only table")
python scripts/results/make_html_table.py --thinking --metrics-file configs/apertus/tasks_posttrain_final.txt --entity apertus --project <project> --models Qwen3-8B-think --output thinking_table.html

# Reasoning level for models whose chat template reads one (gpt-oss: low/medium/high);
# the run is named gpt-oss-120b-think-effort-high (details: "Reasoning effort")
bash scripts/lm_eval_harness/launch_evaluations.sh single --task aime25 --model openai/gpt-oss-120b \
  --thinking --reasoning-effort high \
  --think-start-token '<|channel|>analysis<|message|>' --think-end-token '<|channel|>final<|message|>'
```


The other frameworks evaluate a model already served behind an OpenAI-compatible endpoint (the CSCS serving key is read from `scripts/cscs_serving_api_key.txt`, see [Credentials](#credentials)):

```bash
# Inspect AI: tau2-bench retail, a served model as the simulated customer
scripts/inspect_ai/run_inspect_eval.sh --task tau2_retail \
  --model RCP-AIaaS/Qwen/Qwen3.6-35B-A3B --api-base-url https://api.swissai.svc.cscs.ch/v1 \
  --model-role user=openai-api/swissai/RCP-AIaaS/Qwen/Qwen3.6-35B-A3B --limit 5

# Harbor: SWE-bench Verified and Terminal-Bench 2, two tasks each, task containers as k8s pods
scripts/harbor/run_harbor_eval.sh --benchmark swe_bench,terminal_bench_2 --sandbox k8s \
  --model RCP-AIaaS/Qwen/Qwen3.6-35B-A3B --api-base-url https://api.swissai.svc.cscs.ch/v1 --limit 2
```

---

## Credentials

Every credential below can be set either as an environment variable, or by creating a **plain-text file at the given path** (relative to the repo root, i.e. next to this README) **containing just the key/token itself** — no quotes, no `export`, no trailing newline needed (a trailing newline is stripped automatically if present). The env var, when set, always takes priority over the file. All of these paths are already listed in `.gitignore`, so a file placed there will never be committed.

```bash
# Example: save your W&B key so every run picks it up automatically, no export needed.
echo -n "wandb_v1_abc123..." > scripts/wandb_api_key.txt
```

| Env var | File (relative to repo root) | Required for |
|---|---|---|
| `HF_TOKEN` | `scripts/hf_token.txt` | Gated/rate-limited Hugging Face datasets and models (most task suites). |
| `CSCS_SERVING_API` | `scripts/cscs_serving_api_key.txt` | LLM-as-a-judge evals (e.g. AlpacaEval) and any `--api-base-url`/`openai` backend run against CSCS serving. Key can be obtained [here](https://serving.swissai.cscs.ch). |
| `OPENAI_API_KEY` | `scripts/openai_api_key.txt` | The optional `gpt` suite; falls back to `CSCS_SERVING_API` if unset. |
| `WANDB_API_KEY` | `scripts/wandb_api_key.txt` | Uploading results to Weights & Biases. |
| `CRITPT_API_KEY` | `scripts/critpt_api_key.txt` | Optional: the CritPt task's (`scripts/aaii/aa_critpt.py`) grading server. Access is granted case by case by the CritPt team (email critpt@artificialanalysis.ai); without it, a run still completes and just writes its submission batch to a local file instead of posting it for grading. |

---

## Repository Structure

```
evals-post-train/
├── scripts/
│   ├── lm_eval_harness/        # lm-evaluation-harness pipeline
│   │   ├── launch_evaluations.sh       # Main launcher (recommended entry point)
│   │   ├── evaluation_orchestrator.sh  # Resumable chunk arrays, retries, aggregation
│   │   ├── evaluation_controller.sbatch# CPU controller submitted after each chunk wave
│   │   ├── evaluate.sbatch             # SLURM job: serve the model (vLLM/SGLang/HF/Megatron), run lm-eval
│   │   ├── evaluate_api.sbatch         # CPU-only wrapper for the OpenAI-compatible API backend
│   │   ├── aggregate_chunks.sbatch     # Merges chunk results, uploads to W&B
│   │   ├── prepare_eval_env.sbatch / build_eval_env.sh  # Fingerprinted environment + harness overlay
│   │   ├── eval_state.py               # Task normalization, result scanning, chunking
│   │   ├── launch_judge.py             # Judge models for LLM-as-a-judge tasks
│   │   ├── convert_megatron_checkpoint.sbatch  # Megatron -> HF conversion
│   │   └── runners/                    # Multi-model scripts (hf_base_runner.sh, example_models.sh)
│   ├── inspect_ai/             # run_inspect_eval.sh/.sbatch, sandbox_patches/
│   ├── aaii/                   # Artificial Analysis protocol: Inspect tasks (*.py) and standalone runners (run_*.sh)
│   ├── harbor/                 # run_harbor_eval.sh/.sbatch, run.py, k8s/docker environments
│   ├── evalscope/              # run_evalscope_eval.sh, run.py, SWE-bench scorer
│   ├── common/                 # k8s_sandbox.py: pod-per-sandbox backend (Harbor, EvalScope)
│   ├── results/                # W&B upload (lm-eval, Inspect, Harbor), eval_export/, export_eval_results.py, make_*table.py
│   └── utils/                  # Log and W&B checks
├── configs/                    # Task lists, main-table metrics, model registry, export mappings
├── containers/                 # Enroot/Pyxis configs and the prebuilt evaluation image (Dockerfile.eval)
├── requirements/               # Pinned dependencies per framework
├── docs/                       # Per-framework documentation
└── tests/
```
