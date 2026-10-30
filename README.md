# SwissAI Evaluation Pipeline

Evaluation infrastructure for LLMs on CSCS Alps (Clariden), with results in W&B. Four evaluation frameworks run side by side, each with its own folder in `scripts/`, entry point, pinned environment and doc:

| Framework | Benchmarks | Model | Entry point | Docs |
|---|---|---|---|---|
| [lm-evaluation-harness](https://github.com/swiss-ai/lm-evaluation-harness) | Standard benchmarks (MMLU, GSM8K, ...), task suites, LLM judges | served in the job (vLLM, SGLang, HF, Megatron-LM), or an endpoint | `scripts/lm_eval_harness/launch_evaluations.sh` | [docs/lm-eval-harness.md](docs/lm-eval-harness.md) |
| [Inspect AI](https://inspect.aisi.org.uk/) | Benchmarks with their own scorer, judge or simulated user (tau2-bench, HLE, AA-LCR, ...) | endpoint | `scripts/inspect_ai/run_inspect_eval.sh` | [docs/inspect-ai.md](docs/inspect-ai.md) |
| [Harbor](https://github.com/harbor-framework/harbor) | Agentic coding and terminal benchmarks (SWE-bench, Terminal-Bench, DeepSWE, Aider Polyglot, ...) | endpoint | `scripts/harbor/run_harbor_eval.sh` | [docs/harbor.md](docs/harbor.md) |
| [EvalScope](https://github.com/modelscope/evalscope) | SWE-bench Verified / Pro | endpoint | `scripts/evalscope/run_evalscope_eval.sh` | [docs/evalscope.md](docs/evalscope.md) |

**Where the model runs.** Only lm-evaluation-harness loads the model itself: its Slurm job starts vLLM, SGLang, HF or Megatron-LM on the checkpoint. Inspect AI, Harbor and EvalScope call a model that is already served behind an OpenAI-compatible endpoint (`--model` + `--api-base-url`): the CSCS serving API (`https://api.swissai.svc.cscs.ch/v1`), or a server you start yourself, e.g. `vllm serve <checkpoint>` on a Clariden node. To evaluate your own checkpoint with them, serve it first and pass that server's URL. Harbor's and EvalScope's `--gold` runs need no model at all.

Harbor and EvalScope run each task in its own container, in local Docker or as Kubernetes pods (Breithorn). `scripts/aaii/` has Artificial Analysis's versions of some benchmarks ([docs/inspect-ai.md](docs/inspect-ai.md)). Results: [docs/results.md](docs/results.md) (W&B, tables, Every Eval Ever / Hugging Face exports).

| | lm-eval-harness | Inspect AI | Harbor | EvalScope |
|---|---|---|---|---|
| On Clariden | `launch_evaluations.sh` submits the jobs | `run_inspect_eval.sbatch` | `run_harbor_eval.sbatch` (task pods on Breithorn) | `run_evalscope_eval.sbatch` (sandbox pods on Breithorn) |
| W&B upload | automatic | automatic | automatic | automatic |

---

## Quick Start

The CSCS serving key is read from `scripts/cscs_serving_api_key.txt` (see [Credentials](#credentials)).

**lm-evaluation-harness**: serves the model in the job and runs a task suite.

```bash
bash scripts/lm_eval_harness/launch_evaluations.sh default --model meta-llama/Llama-3.1-8B-Instruct --name Llama-Baseline
```

**Inspect AI**: tau2-bench retail, a served model as the simulated customer.

```bash
scripts/inspect_ai/run_inspect_eval.sh --task tau2_retail \
  --model RCP-AIaaS/Qwen/Qwen3.6-35B-A3B --api-base-url https://api.swissai.svc.cscs.ch/v1 \
  --model-role user=openai-api/swissai/RCP-AIaaS/Qwen/Qwen3.6-35B-A3B --limit 5 --no-wandb
```

**Harbor**: SWE-bench Verified and Terminal-Bench 2, two tasks each, task containers as Breithorn pods.

```bash
scripts/harbor/run_harbor_eval.sh --benchmark swe_bench,terminal_bench_2 --sandbox k8s \
  --model RCP-AIaaS/Qwen/Qwen3.6-35B-A3B --api-base-url https://api.swissai.svc.cscs.ch/v1 --limit 2 --no-wandb
```

**EvalScope**: SWE-bench Verified and Pro, two instances each, sandboxes in local Docker. The default agent works through tool calls, so the served model needs tool calling.

```bash
scripts/evalscope/run_evalscope_eval.sh --benchmark swe_bench,swe_bench_pro --limit 2 --no-wandb \
  --model <served model with tool calling> --api-base-url https://api.swissai.svc.cscs.ch/v1
```

The examples above are smoke tests, hence `--no-wandb`: all four frameworks upload the results to W&B whenever a W&B key is available, by default to `apertus/apertus-1.5-post-training-v0.1` (`--wandb-entity`/`--wandb-project` change it; `--no-wandb` or `WANDB_MODE=disabled` skips it, e.g. for a smoke test; `--gold` runs are never uploaded). Each doc has more: Clariden jobs, checkpoints, reference-solution (gold) checks, judges, reasoning metrics.

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
│   ├── lm_eval_harness/   # launch_evaluations.sh: Slurm jobs that serve the model and run lm-eval, in resumable chunks; judges, Megatron conversion
│   ├── inspect_ai/        # run_inspect_eval.sh/.sbatch, sandbox_patches/
│   ├── harbor/            # run_harbor_eval.sh/.sbatch, run.py (benchmarks), k8s/docker environments, task_patches.py
│   ├── evalscope/         # run_evalscope_eval.sh/.sbatch, run.py, SWE-bench scorer
│   ├── aaii/              # Artificial Analysis protocol: Inspect tasks (*.py) and standalone runners (run_*.sh)
│   ├── common/            # k8s_sandbox.py (task containers as pods: Harbor, EvalScope), wandb.sh (the W&B upload rule)
│   └── results/           # W&B upload, tables, exports
├── configs/               # lm-eval task lists, main-table metrics, model registry, export mappings
├── containers/            # Enroot/Pyxis configs and the prebuilt evaluation image (Dockerfile.eval)
├── requirements/          # Pinned dependencies, one file per framework environment
├── docs/                  # One doc per framework, plus results.md
└── tests/
```
