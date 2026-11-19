# Evals-Post-Train Evaluations

Evaluation infrastructure for LLMs on CSCS Alps (Clariden), with results in W&B. Four evaluation frameworks run side by side, each with its own folder in `scripts/`, entry point, environment and doc:

| Framework | Benchmarks | Entry point | Docs |
|---|---|---|---|
| [lm-evaluation-harness](https://github.com/swiss-ai/lm-evaluation-harness) | Standard benchmarks (MMLU, GSM8K, ...), task suites, LLM judges | `scripts/lm_eval_harness/launch_evaluations.sh` | [docs/lm-eval-harness.md](docs/lm-eval-harness.md) |
| [Inspect AI](https://inspect.aisi.org.uk/) | Benchmarks with their own scorer, judge or simulated user (tau2-bench, HLE, AA-LCR, ...) | `scripts/inspect_ai/run_inspect_eval.sh` | [docs/inspect-ai.md](docs/inspect-ai.md) |
| [Harbor](https://github.com/harbor-framework/harbor) | Agentic coding and terminal benchmarks (SWE-bench, Terminal-Bench, DeepSWE, Aider Polyglot, ...) | `scripts/harbor/run_harbor_eval.sh` | [docs/harbor.md](docs/harbor.md) |
| [EvalScope](https://github.com/modelscope/evalscope) | SWE-bench Verified / Pro, BFCL v4 (tool calls) | `scripts/evalscope/run_evalscope_eval.sh` | [docs/evalscope.md](docs/evalscope.md) |

`scripts/aaii/` has Artificial Analysis's versions of some benchmarks ([docs/inspect-ai.md](docs/inspect-ai.md)). Results: [docs/results.md](docs/results.md) (W&B, tables, Every Eval Ever / Hugging Face exports). Every launcher takes the same sampling flags (`--temperature`, `--top-p`, `--top-k`, `--min-p`, `--repetition-penalty`) over one default policy: [docs/sampling.md](docs/sampling.md). The same goes for thinking and reasoning effort (`--enable-thinking`, `--no-enable-thinking`, `--reasoning-effort`): [docs/thinking.md](docs/thinking.md). A comparison's benchmarks, repeats and per-setup flags are written down in a suite file: [docs/suites.md](docs/suites.md).

### How a run can be set up

| Framework | Model | Runner runs on | Task containers |
|---|---|---|---|
| lm-eval-harness | loaded in the job: `--backend vllm` (default), `sglang`, `hf`, `megatron_lm` | Clariden GPU job, in a CSCS container (`containers/env*.toml`) | — |
| lm-eval-harness | `--backend openai`: an OpenAI-compatible endpoint¹ (`completions` or `chat-completions`) | Clariden CPU job (`run_lm_eval_api.sbatch`) | — |
| Inspect AI | an OpenAI-compatible endpoint¹ (`--api-base-url`), or an Inspect provider (`--model anthropic/...`) | local machine, or Clariden CPU job in a CSCS container (`run_inspect_eval.sbatch`) | local Docker, `--sandbox local` without Docker, or Kata microVMs through OpenSandbox (`--sandbox opensandbox`) |
| Harbor | an OpenAI-compatible endpoint¹; `--gold`: no model | local machine, Clariden CPU job (`run_harbor_eval.sbatch`), or a Kubernetes pod | local Docker, Kubernetes pods (`--sandbox k8s`; Breithorn from Clariden), or Kata microVMs through OpenSandbox (`--sandbox opensandbox`) |
| EvalScope | an OpenAI-compatible endpoint¹ (tool calling for the default agent); `--gold`: no model | local machine, Clariden CPU job (`run_evalscope_eval.sbatch`), or a Kubernetes pod | local Docker, Kubernetes pods (`--sandbox k8s`; Breithorn from Clariden), or Kata microVMs through OpenSandbox (`--sandbox opensandbox`) |

¹ The CSCS serving API (`https://api.swissai.svc.cscs.ch/v1`), which serves only models launched and registered there, or a server you start yourself, e.g. `vllm serve <checkpoint>` on a Clariden node, passed by its own URL (no registration).

A model of your own can also be served and evaluated by one Slurm job, on several nodes if it needs them, with no gateway between: [the direct route](#serving-the-model-in-the-eval-job-direct-route).

Every runner except the in-job lm-eval backends can also run from the prebuilt image (`containers/Dockerfile.eval`, [details](docs/lm-eval-harness.md#prebuilt-evaluation-image-no-installs-at-run-time)), which installs nothing at run time.

### Chat template

- **lm-eval, post-trained models:** `--chat-template-by-type`: generative tasks with the chat template, log-likelihood tasks (multiple choice) without it. lm-eval applies the template to every task by default, and through it a log-likelihood task measures how likely the model is to start its reply with the answer, not the raw continuation.
- **lm-eval, base models:** never a template: `--no-chat-template`.
- **lm-eval, post-trained Apertus 2:** it has no Jinja template: `--chat-format apertus2 --tokenizer <dir or name> --max-length <max-model-len>` ([details](docs/lm-eval-harness.md#apertus-2-native-format)).
- **Inspect, Harbor, EvalScope:** they send chat requests, so the **server** applies the template, normally the model's own from `tokenizer_config.json` ([Inspect](docs/inspect-ai.md#chat-template)). The exception is EvalScope's `--completions` for base models: plain text to `/v1/completions`, no template ([details](docs/evalscope.md)). For Apertus 2 the server needs the `apertus-common` vLLM plugin ([setup](docs/thinking.md#apertus-2-on-v1chatcompletions-temporary)).

### Thinking models, judges, sampling

- **Thinking models:** no deterministic settings (keep the thinking profile in `configs/sampling_defaults.json`), and generally report repeated scores (`--repeat-config`, [docs/repeated_sampling.md](docs/repeated_sampling.md)).
- **Judges:** no self-grading (biased); use a separate, fixed judge model.
- **Sampling and thinking flags:** one policy for all launchers, changed in `scripts/common/` ([sampling](docs/sampling.md), [thinking](docs/thinking.md)).

---

## Quick Start

The CSCS serving key is read from `scripts/secrets.env` (see [Credentials](#credentials)).

**lm-evaluation-harness**: a model behind an OpenAI-compatible endpoint (a server you started, here a vLLM server on node `nid001234`; or the CSCS serving API with its URL and the served name as `--api-model-name`), GSM8K with chain of thought, 5 samples. `--model` is also the tokenizer's name, so it must be a Hugging Face name or a path.

```bash
bash scripts/lm_eval_harness/launch_evaluations.sh single --task gsm8k_cot --model Qwen/Qwen3-8B --limit 5 --no-wandb \
  --backend openai --api-base-url http://nid001234:8000
```

**lm-evaluation-harness, Apertus 2 native format**: a served Apertus 2 checkpoint through the CSCS serving API, 5 samples. `--tokenizer` is a local directory with the verified Apertus 2 instruct tokenizer that `apertus-common` names: [`tokenizers/Apertus_2_instruct`](https://github.com/swiss-ai/apertus-omni-tokenizer/tree/b801ff1b00d933d10bba90b0c22307157299faa7/tokenizers/Apertus_2_instruct) in `swiss-ai/apertus-omni-tokenizer` (evals-svc's Chonk/Megachonk evaluation tokenizer lacks the control tokens and is rejected), `--max-length` is the server's `--max-model-len`. Installs the pinned `apertus-common` commit ([details](docs/lm-eval-harness.md#apertus-2-native-format)).

```bash
bash scripts/lm_eval_harness/launch_evaluations.sh single --task gsm8k --limit 5 --no-wandb \
  --backend openai --api-base-url https://api.swissai.svc.cscs.ch/v1 \
  --api-model-name SwissAI-Research/swiss-ai/chonk-lc-16k-v0.1-ep1-iter_0019374 \
  --chat-format apertus2 --tokenizer <tokenizer dir> --max-length 32768
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

**Suites**: a whole comparison (every benchmark, for each model setup) from one suite file. Without `--submit` it only prints the launch commands ([docs/suites.md](docs/suites.md)).

```bash
python -m scripts.common.suites run apertus2/chonk_sft --api-base-url https://api.swissai.svc.cscs.ch/v1 \
  --model chonk-lc-16k-v0.2-sft=<served name> --limit 5 --no-wandb      # add --submit to launch
```

`--limit N` is a random subset of N samples per task (seed 42, so the same subset on every run) in all four frameworks, not the first N. lm-eval, EvalScope and Harbor also take a fraction below 1 (`--limit 0.5`: a random half); Inspect takes whole numbers only.

The examples above are smoke tests, hence `--no-wandb`. Otherwise all four frameworks upload the results to W&B whenever a W&B key is available, by default to `apertus/e-p-t-pool` (change it with `--wandb-entity`/`--wandb-project`, skip it with `--no-wandb` or `WANDB_MODE=disabled`; `--gold` runs are never uploaded). Each doc has more: Clariden jobs, checkpoints, reference-solution (gold) checks, judges, reasoning metrics.

---

## Serving the model in the eval job (direct route)

For a checkpoint that is not on the CSCS serving API, or too big for one node, one command starts **one Slurm job** that serves the model and runs the evals against it. The two parts run as steps of the job, each in its own container:

```
sbatch                       scripts/direct_serving/job.py
|- srun   model server(s)    serving image (vLLM), one step per node, Ray across the nodes of a replica
'- srun   eval command 1, 2  prebuilt eval image, one after another, against http://127.0.0.1:8080/v1
```

- **The server** is started by model-launch's own job script with OpenTela, metrics and telemetry off: nothing registers on the serving gateway, so its 503s, rate limits and header timeout never reach a run.
- **The evals** are launcher commands of any of the four frameworks (`--eval`, repeatable). The job waits for the server's `/health`, then runs them in order from this checkout in the [prebuilt eval image](docs/lm-eval-harness.md#prebuilt-evaluation-image-no-installs-at-run-time): the image of the last commit that changed what it is built from, which the job checks on ghcr before it starts. The model loads once for all of them, and a command finds the server's URL in `$EVAL_API_BASE_URL`.
- **The end:** the job ends with status 1 if the server died or any command failed; a failed command does not stop the next one.
- **A suite** can say how its model is served: a setup's `launch` block (checkpoint, serving EDF, nodes, time, parallelism) is read by `--suite` and `--setup`, and the setup's `serving` arguments are added ([docs/suites.md](docs/suites.md)).

Run it from the root of this checkout on a Clariden login node, in a Python that has `swiss_ai_model_launch` ([setup](docs/lm-eval-harness.md#judge-model-launching)); the checkout has to be on storage the compute nodes see. `--submit` submits the job and prints its id; without it the job script is printed. The job's logs are in `~/.sml/logs/<job id>/` (`--workdir` changes it): `log.out` for the commands, `replica_0.out`/`.err` for the server. Each runner writes its results where it always does.

**A model by hand**: Qwen3-32B on one node, Inspect and lm-eval, 3 samples each. `--framework-args` follow `--model`, `--served-model-name` and `--host`, which the launcher sets; lm-eval needs `--failure-policy fail-fast` here, and a Hugging Face name or a path as `--tokenizer`, not the served name.

```bash
python -m scripts.direct_serving.job \
  --model /capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3-32B --served-model-name qwen3-32b \
  --nodes 1 --time 01:00:00 --environment containers/env_vllm.toml \
  --framework-args "--tensor-parallel-size 4 --max-model-len 32768 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes" \
  --eval 'scripts/inspect_ai/run_inspect_eval.sh --task gsm8k --model qwen3-32b --api-base-url "$EVAL_API_BASE_URL" --limit 3 --no-wandb --no-enable-thinking' \
  --eval 'bash scripts/lm_eval_harness/launch_evaluations.sh single --task gsm8k --model qwen3-32b --backend openai --api-base-url "$EVAL_API_BASE_URL" --api-model-name qwen3-32b --tokenizer Qwen/Qwen3-32B --limit 3 --no-wandb --failure-policy fail-fast' \
  --submit
```

**A suite setup**: Chonk (`apertus2/chonk_sft`, setup `chonk-lc-16k-v0.2-sft`) on 2 nodes. The setup's `launch` block gives the checkpoint, the Apertus 2 serving EDF, 2 nodes, 24 h and the tensor-parallel 1 x data-parallel 8 layout over Ray; its `serving` arguments make the server render the Apertus 2 format. The server needs the `apertus-common` vLLM plugin, which its image does not contain: the install is `--pre-launch-cmds` (the commit is pinned in `requirements/apertus2.txt`, [details](docs/thinking.md#apertus-2-on-v1chatcompletions-temporary)). The model is served as `chonk-lc-16k-v0.2-sft`, the setup's name; `--time` overrides the suite's 24 h.

```bash
python -m scripts.direct_serving.job --suite apertus2/chonk_sft --setup chonk-lc-16k-v0.2-sft --time 01:00:00 \
  --pre-launch-cmds "pip install -q --no-deps --force-reinstall --ignore-requires-python 'git+https://github.com/swiss-ai/apertus-common.git@<pin of requirements/apertus2.txt>'" \
  --eval 'scripts/inspect_ai/run_inspect_eval.sh --task gsm8k --model chonk-lc-16k-v0.2-sft --api-base-url "$EVAL_API_BASE_URL" --limit 5 --no-wandb' \
  --submit
```

`--partition`, `--qos`, `--account` and `--reservation` set the job's Slurm options (the partition is `normal` unless you name another). The Slurm options, the eval image (`--eval-image-commit`), extra mounts and everything else: [docs/direct-serving.md](docs/direct-serving.md).

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
| `CSCS_SERVING_API` | all four frameworks, the AA scripts | LLM-as-a-judge evals (e.g. AlpacaEval) and any `--api-base-url`/`openai` backend run against CSCS serving. Key can be obtained [here](https://serving.swissai.cscs.ch). Not needed to serve and evaluate in the direct route, only for a judge on the CSCS serving API. |
| `OPENAI_API_KEY` | lm-eval jobs, `run_inspect_eval.sbatch` | The optional `gpt` suite (falls back to `CSCS_SERVING_API` in lm-eval jobs), and Inspect `openai/...` model roles. |
| `WANDB_API_KEY` | all four frameworks | Uploading results to Weights & Biases. |
| `CRITPT_API_KEY` | `run_inspect_eval.sh` | Optional: grading for CritPt (`scripts/aaii/aa_critpt.py`); ask critpt@artificialanalysis.ai for access. Without it, the submissions are written to a local file. |
| `OPENSANDBOX_API_KEY` | `run_harbor_eval.sh`, `run_evalscope_eval.sh`, `run_inspect_eval.sh`, `run_gdpval.sh` | Optional: `--sandbox opensandbox` (`--sandbox-backend opensandbox` for `run_gdpval.sh`). |
| `ANTHROPIC_API_KEY` or `PORTKEY_API_KEY` | `run_harbor_eval.sh` | Senior SWE-bench model runs: its LLM graders, part of the reward. Without either, the runner refuses such a run (`--gold` needs neither). |

---

## Contributing

Adding a benchmark or a framework, fixing a bug, running the tests, changing a dependency: [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Repository Structure

```
evals-post-train/
├── scripts/
│   ├── lm_eval_harness/   # launch_evaluations.sh: Slurm jobs that serve the model and run lm-eval, in resumable chunks; judges, Megatron conversion
│   ├── direct_serving/    # job.py: one Slurm job that serves a model and runs evals against it; inline_slurm/: sbatch and srun that run in place
│   ├── inspect_ai/        # run_inspect_eval.sh/.sbatch, sandbox_patches/, opensandbox_sandbox.py (the opensandbox sandbox provider)
│   ├── harbor/            # run_harbor_eval.sh/.sbatch, run.py (benchmarks), k8s/docker/opensandbox environments, task_patches.py
│   ├── evalscope/         # run_evalscope_eval.sh/.sbatch, run.py, SWE-bench scorer
│   ├── aaii/              # Artificial Analysis protocol: Inspect tasks (*.py) and standalone runners (run_*.sh)
│   ├── common/            # suites.py (suite files: `plan`, `flags`, `run`), sampling.py and thinking.py (the shared sampling and thinking policy), k8s_sandbox.py (task containers as pods: Harbor, EvalScope), opensandbox_sandbox.py (as Kata microVMs: EvalScope; Harbor's cleanup), dockerfile_replay.py (Dockerfiles replayed where images can't be built: Harbor, Inspect), wandb.sh (the W&B upload rule), lock_requirements.sh and locked_venv.sh (dependency locks)
│   └── results/           # W&B upload, tables, exports
├── configs/               # sampling_defaults.json, suites/ (comparison definitions), export mappings
├── containers/            # Enroot/Pyxis configs (serving images, env_vllm_apertus2.toml for Apertus 2) and the prebuilt evaluation image (Dockerfile.eval)
├── requirements/          # Dependencies per framework environment: .txt (edited) and .lock (generated; see CONTRIBUTING.md for what installs which)
├── docs/                  # One doc per framework, plus direct-serving.md, results.md, sampling.md, thinking.md, suites.md and repeated_sampling.md
└── tests/
```
