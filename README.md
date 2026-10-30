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

lm-eval-harness serves the model in the job (vLLM, SGLang, HF or Megatron-LM) and runs a suite. Results go to W&B, from which [docs/results.md](docs/results.md#reporting-building-result-tables) builds tables:

```bash
# A model on the default suite, with a custom W&B run name
bash scripts/lm_eval_harness/launch_evaluations.sh default --model meta-llama/Llama-3.1-8B-Instruct --name Llama-Baseline

# A checkpoint with vLLM, in resumable chunks of 8 tasks, at most 4 running at once
bash scripts/lm_eval_harness/launch_evaluations.sh olmo-easy --model /capstor/store/../apertus-.../checkpoints/ \
  --backend vllm --chunk-size 8 --max-parallel 4

# A reasoning model, recording the reasoning metrics (see "Thinking / Reasoning Metrics")
bash scripts/lm_eval_harness/launch_evaluations.sh posttrain --model Qwen/Qwen3-8B \
  --thinking --think-start-token '<think>' --think-end-token '</think>' --name Qwen3-8B-think
```

More in [docs/lm-eval-harness.md](docs/lm-eval-harness.md): base models and few-shot, an existing endpoint (`--backend openai`), [Megatron checkpoints](docs/lm-eval-harness.md#megatron-checkpoints) natively or converted to HF, reasoning effort, judges.

The other frameworks evaluate a model already served behind an OpenAI-compatible endpoint (the CSCS serving key is read from `scripts/cscs_serving_api_key.txt`, see [Credentials](#credentials)):

```bash
# Inspect AI: tau2-bench retail, a served model as the simulated customer
scripts/inspect_ai/run_inspect_eval.sh --task tau2_retail \
  --model RCP-AIaaS/Qwen/Qwen3.6-35B-A3B --api-base-url https://api.swissai.svc.cscs.ch/v1 \
  --model-role user=openai-api/swissai/RCP-AIaaS/Qwen/Qwen3.6-35B-A3B --limit 5

# Harbor: SWE-bench Verified and Terminal-Bench 2, two tasks each, task containers as k8s pods
scripts/harbor/run_harbor_eval.sh --benchmark swe_bench,terminal_bench_2 --sandbox k8s \
  --model RCP-AIaaS/Qwen/Qwen3.6-35B-A3B --api-base-url https://api.swissai.svc.cscs.ch/v1 --limit 2

# Harbor on Clariden: a CPU job, task containers as Breithorn pods (log in with kubelogin first)
sbatch -p debug -t 1:00:00 scripts/harbor/run_harbor_eval.sbatch --benchmark aider_polyglot \
  --model RCP-AIaaS/Qwen/Qwen3.6-35B-A3B --api-base-url https://api.swissai.svc.cscs.ch/v1 --limit 5
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

## Used by evals-svc

[evals-svc](https://github.com/swiss-ai/evals-svc) runs evaluations through this repo: it clones it at a pinned commit (`evals_post_train_ref`) and runs in the image built from it (`ghcr.io/swiss-ai/evals-post-train-eval:<commit>`, `containers/Dockerfile.eval`, built on every push). It calls these paths, so renaming or removing one breaks it; change evals-svc in step:

| evals-svc | Here |
|---|---|
| lm-eval-harness runs, checkpoint evals | `scripts/lm_eval_harness/launch_evaluations.sh` (and, through it, `evaluate.sbatch`), `convert_megatron_checkpoint.sbatch` |
| Inspect AI runs | `scripts/inspect_ai/run_inspect_eval.sh`, and the tasks `scripts/aaii/{hle,gpqa_diamond,scicode,aa_lcr,aa_omniscience,aa_critpt}.py` by path |
| EvalScope runs (SWE-bench Verified / Pro) | `scripts/evalscope/run_evalscope_eval.sh` |
| τ²-bench, GDPval, Terminal-Bench suites | `scripts/aaii/run_tau2_bench.sh`, `run_gdpval.sh`, `run_terminal_bench.sh` |

The Harbor runner (`scripts/harbor/`) is not used by evals-svc.

---

## Development

```bash
uv run --all-extras python -m unittest discover -s tests   # what CI's "Offline checks" runs (plus shellcheck, syntax and JSON checks)
```

The Harbor tests need Harbor installed (Python >= 3.12): `pip install -r requirements/harbor-runtime.txt`, then `python -m unittest tests.test_harbor_runner tests.test_harbor_wandb`; without it, the Harbor-only tests are skipped.

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
│   ├── harbor/                 # run_harbor_eval.sh/.sbatch, run.py (benchmarks), k8s/docker environments, task_patches.py
│   ├── evalscope/              # run_evalscope_eval.sh, run.py, SWE-bench scorer
│   ├── common/                 # k8s_sandbox.py: pod-per-sandbox backend (Harbor, EvalScope)
│   └── results/                # W&B upload (lm-eval, Inspect, Harbor), eval_export/, export_eval_results.py, make_*table.py
├── configs/                    # Task lists, main-table metrics, model registry, export mappings
├── containers/                 # Enroot/Pyxis configs and the prebuilt evaluation image (Dockerfile.eval)
├── requirements/               # Pinned dependencies per framework
├── docs/                       # Per-framework documentation
└── tests/
```
