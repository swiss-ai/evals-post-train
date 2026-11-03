# Evals-Post-Train Evaluations

Evaluation infrastructure for LLMs on CSCS Alps (Clariden), with results in W&B. Four evaluation frameworks run side by side, each with its own folder in `scripts/`, entry point, environment and doc:

| Framework | Benchmarks | Entry point | Docs |
|---|---|---|---|
| [lm-evaluation-harness](https://github.com/swiss-ai/lm-evaluation-harness) | Standard benchmarks (MMLU, GSM8K, ...), task suites, LLM judges | `scripts/lm_eval_harness/launch_evaluations.sh` | [docs/lm-eval-harness.md](docs/lm-eval-harness.md) |
| [Inspect AI](https://inspect.aisi.org.uk/) | Benchmarks with their own scorer, judge or simulated user (tau2-bench, HLE, AA-LCR, ...) | `scripts/inspect_ai/run_inspect_eval.sh` | [docs/inspect-ai.md](docs/inspect-ai.md) |
| [Harbor](https://github.com/harbor-framework/harbor) | Agentic coding and terminal benchmarks (SWE-bench, Terminal-Bench, DeepSWE, Aider Polyglot, ...) | `scripts/harbor/run_harbor_eval.sh` | [docs/harbor.md](docs/harbor.md) |
| [EvalScope](https://github.com/modelscope/evalscope) | SWE-bench Verified / Pro | `scripts/evalscope/run_evalscope_eval.sh` | [docs/evalscope.md](docs/evalscope.md) |

`scripts/aaii/` has Artificial Analysis's versions of some benchmarks ([docs/inspect-ai.md](docs/inspect-ai.md)). Results: [docs/results.md](docs/results.md) (W&B, tables, Every Eval Ever / Hugging Face exports).

### How a run can be set up

| Framework | Model | Runner runs on | Task containers |
|---|---|---|---|
| lm-eval-harness | loaded in the job: `--backend vllm` (default), `sglang`, `hf`, `megatron_lm` | Clariden GPU job, in a CSCS container (`containers/env*.toml`) | — |
| lm-eval-harness | `--backend openai`: an OpenAI-compatible endpoint¹ (`completions` or `chat-completions`) | Clariden CPU job (`run_lm_eval_api.sbatch`) | — |
| Inspect AI | an OpenAI-compatible endpoint¹ (`--api-base-url`), or an Inspect provider (`--model anthropic/...`) | local machine, or Clariden CPU job in a CSCS container (`run_inspect_eval.sbatch`) | local Docker, or `--sandbox local` without Docker |
| Harbor | an OpenAI-compatible endpoint¹; `--gold`: no model | local machine, Clariden CPU job (`run_harbor_eval.sbatch`), or a Kubernetes pod | local Docker, Kubernetes pods (`--sandbox k8s`; Breithorn from Clariden), or Kata microVMs through OpenSandbox (`--sandbox opensandbox`) |
| EvalScope | an OpenAI-compatible endpoint¹ (tool calling for the default agent); `--gold`: no model | local machine, Clariden CPU job (`run_evalscope_eval.sbatch`), or a Kubernetes pod | local Docker, Kubernetes pods (`--sandbox k8s`; Breithorn from Clariden), or Kata microVMs through OpenSandbox (`--sandbox opensandbox`) |

¹ The CSCS serving API (`https://api.swissai.svc.cscs.ch/v1`), which serves only models launched and registered there, or a server you start yourself, e.g. `vllm serve <checkpoint>` on a Clariden node, passed by its own URL (no registration).

Every runner except the in-job lm-eval backends can also run from the prebuilt image (`containers/Dockerfile.eval`, [details](docs/lm-eval-harness.md#prebuilt-evaluation-image-no-installs-at-run-time)), which installs nothing at run time.

---

## Quick Start

The CSCS serving key is read from `scripts/secrets.env` (see [Credentials](#credentials)).

**lm-evaluation-harness**: serves the model in the job and runs the default task suite, 10 samples per task.

```bash
bash scripts/lm_eval_harness/launch_evaluations.sh default --model meta-llama/Llama-3.1-8B-Instruct \
  --name Llama-Baseline --limit 10 --no-wandb
```

**Inspect AI**: tau2-bench retail, a served model as the simulated customer.

```bash
scripts/inspect_ai/run_inspect_eval.sh --task tau2_retail \
  --model RCP-AIaaS/Qwen/Qwen3.6-35B-A3B --api-base-url https://api.swissai.svc.cscs.ch/v1 \
  --model-role user=openai-api/swissai/RCP-AIaaS/Qwen/Qwen3.6-35B-A3B --limit 5 --no-wandb
```

**Harbor**: SWE-bench Verified and Terminal-Bench 2, two tasks each, task containers as pods in Breithorn's `rob-poc` namespace (your kubeconfig must point at Breithorn).

```bash
SANDBOX_K8S_NAMESPACE=rob-poc scripts/harbor/run_harbor_eval.sh --benchmark swe_bench,terminal_bench_2 --sandbox k8s \
  --model RCP-AIaaS/Qwen/Qwen3.6-35B-A3B --api-base-url https://api.swissai.svc.cscs.ch/v1 --limit 2 --no-wandb
```

**EvalScope**: SWE-bench Verified and Pro, two instances each, sandboxes in local Docker. The default agent works through tool calls, so the served model needs tool calling.

```bash
scripts/evalscope/run_evalscope_eval.sh --benchmark swe_bench,swe_bench_pro --limit 2 --no-wandb \
  --model <served model with tool calling> --api-base-url https://api.swissai.svc.cscs.ch/v1
```

The examples above are smoke tests, hence `--no-wandb`. Otherwise all four frameworks upload the results to W&B whenever a W&B key is available, by default to `apertus/apertus-1.5-post-training-v0.1` (change it with `--wandb-entity`/`--wandb-project`, skip it with `--no-wandb` or `WANDB_MODE=disabled`; `--gold` runs are never uploaded). Each doc has more: Clariden jobs, checkpoints, reference-solution (gold) checks, judges, reasoning metrics.

---

## Credentials

All keys go in one file, `scripts/secrets.env` (in `.gitignore`), one `NAME=value` per line; fill in the ones you have:

```bash
cp scripts/secrets.env.example scripts/secrets.env
```

Every runner reads it (`scripts/common/secrets.sh`). A variable already set in the environment wins over the file, e.g. `WANDB_API_KEY=... bash scripts/...` for a one-off.

| Variable | Read by | Required for |
|---|---|---|
| `HF_TOKEN` | lm-eval jobs, `run_inspect_eval.sbatch` | Gated/rate-limited Hugging Face datasets and models (most task suites). |
| `CSCS_SERVING_API` | all four frameworks, the AA scripts | LLM-as-a-judge evals (e.g. AlpacaEval) and any `--api-base-url`/`openai` backend run against CSCS serving. Key can be obtained [here](https://serving.swissai.cscs.ch). |
| `OPENAI_API_KEY` | lm-eval jobs, `run_inspect_eval.sbatch` | The optional `gpt` suite (falls back to `CSCS_SERVING_API` in lm-eval jobs), and Inspect `openai/...` model roles. |
| `WANDB_API_KEY` | all four frameworks | Uploading results to Weights & Biases. |
| `CRITPT_API_KEY` | `run_inspect_eval.sh` | Optional: grading for CritPt (`scripts/aaii/aa_critpt.py`); ask critpt@artificialanalysis.ai for access. Without it, the submissions are written to a local file. |
| `OPENSANDBOX_API_KEY` | `run_harbor_eval.sh`, `run_evalscope_eval.sh` | Optional: `--sandbox opensandbox`. |

Moving from the old per-key files (`scripts/*_api_key.txt`, `scripts/hf_token.txt`), once, from the repo root:

```bash
for pair in CSCS_SERVING_API:cscs_serving_api_key WANDB_API_KEY:wandb_api_key HF_TOKEN:hf_token \
    OPENAI_API_KEY:openai_api_key CRITPT_API_KEY:critpt_api_key OPENSANDBOX_API_KEY:opensandbox_api_key; do
    f="scripts/${pair#*:}.txt"
    [ -f "$f" ] && printf '%s=%s\n' "${pair%%:*}" "$(tr -d '\r\n' < "$f")" >> scripts/secrets.env
done
```

Check `scripts/secrets.env`, then delete the old files.

---

## Contributing

Adding a benchmark or a framework, fixing a bug, running the tests, changing a dependency: [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Repository Structure

```
evals-post-train/
├── scripts/
│   ├── lm_eval_harness/   # launch_evaluations.sh: Slurm jobs that serve the model and run lm-eval, in resumable chunks; judges, Megatron conversion
│   ├── inspect_ai/        # run_inspect_eval.sh/.sbatch, sandbox_patches/
│   ├── harbor/            # run_harbor_eval.sh/.sbatch, run.py (benchmarks), k8s/docker/opensandbox environments, task_patches.py
│   ├── evalscope/         # run_evalscope_eval.sh/.sbatch, run.py, SWE-bench scorer
│   ├── aaii/              # Artificial Analysis protocol: Inspect tasks (*.py) and standalone runners (run_*.sh)
│   ├── common/            # k8s_sandbox.py (task containers as pods: Harbor, EvalScope), opensandbox_sandbox.py (as Kata microVMs: EvalScope; Harbor's cleanup), wandb.sh (the W&B upload rule), lock_requirements.sh and locked_venv.sh (dependency locks)
│   └── results/           # W&B upload, tables, exports
├── configs/               # lm-eval task lists, main-table metrics, model registry, export mappings
├── containers/            # Enroot/Pyxis configs and the prebuilt evaluation image (Dockerfile.eval)
├── requirements/          # Dependencies per framework environment: .txt (edited) and .lock (generated; see CONTRIBUTING.md for what installs which)
├── docs/                  # One doc per framework, plus results.md
└── tests/
```
