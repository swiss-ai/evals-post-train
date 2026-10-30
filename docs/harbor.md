# Harbor

`scripts/harbor/run_harbor_eval.sh` runs agentic benchmarks through [Harbor](https://github.com/harbor-framework/harbor), the harness behind Terminal-Bench. SWE-bench Verified and Pro can also run through [EvalScope](evalscope.md), with a different agent and scoring, so the two give different scores. Every task is a container the agent works in. The task's own verifier scores the result as a reward from 0 to 1.

| `--benchmark` | Dataset (pinned in `scripts/harbor/run.py`) | Tasks |
|---|---|---|
| `swe_bench` | SWE-bench Verified (`swebench-verified@1.0`) | 500 |
| `swe_bench_pro` | SWE-bench Pro **v1** (`swebenchpro@1.0`) | 731 |
| `multi_swe_bench` | Multi-SWE-bench, generated locally by Harbor's adapter (no published dataset) | 1,632 |
| `terminal_bench_2` | Terminal-Bench 2.0 | 89 |
| `terminal_bench_2_1` | Terminal-Bench 2.1, the version Artificial Analysis reports (their protocol: `terminus-2`, `--attempts 3`) | 89 |
| `terminal_bench_4` | Terminal-Bench 4.0.0 | 66 |
| `deep_swe` | DeepSWE v1.1 | 113 |
| `slop_code_bench` | SlopCodeBench | 36 |
| `senior_swe_bench` | Senior SWE-bench v2026.06 (public tasks); `--sandbox docker` only, grading needs `ANTHROPIC_API_KEY` (see below) | 50 |
| `aider_polyglot` | Aider Polyglot (`aider-polyglot@1.0`), patched to run every test (see below) | 225 |

Harbor Hub datasets are pinned by content digest and the older registry datasets by version. Harbor runs in its own venv (`requirements/harbor-runtime.txt`, Python ≥ 3.12, installed with uv). The prebuilt eval image (`containers/Dockerfile.eval`) carries that venv (`HARBOR_VENV`) and the Multi-SWE-bench adapter (`HARBOR_ADAPTERS_ROOT`), so with `EVAL_PREBUILT_ENV=1` nothing is installed or fetched with uvx at run time. The adapter reads Multi-SWE-bench from Hugging Face at a pinned revision (`hf_dataset` in `scripts/harbor/run.py`). The default agent is `terminus-2`, which drives a tmux session and needs no tool-call parser on the served model. It calls `--model` through litellm as `openai/<model>` at `--api-base-url`. `--gold` runs Harbor's `oracle` agent, which replays each task's reference solution, to check the environments and verifiers. Scores and run metadata go to `logs/harbor/<name>/summary.json`: per benchmark, the mean `reward`, or for verifiers that report several named metrics (SlopCodeBench) the mean of each. `results.json` next to it has the same scores flattened (`accuracy` = mean reward) plus one sample per trial: the shape evals-svc reports as is.

`--sandbox` picks where the task containers run:

- `docker`: Harbor's own Docker environment (`scripts/harbor/docker_environment.py` builds its network-allowlist proxy for the task images' platform). On Apple Silicon the linux/amd64 images run emulated; Erlang crashes under that emulation, so Senior SWE-bench's Elixir tasks only build on an amd64 host.
- `k8s`: one unprivileged pod per trial through `scripts/harbor/k8s_environment.py`, using the current kubeconfig, with no Docker daemon needed. The pods come from `scripts/common/k8s_sandbox.py` (shared with the EvalScope runner) and are pinned to amd64 nodes. Pod settings come from `SANDBOX_K8S_*` env vars (namespace, run label, CPU and memory requests, image pull secret, ...; see the module docstring). If the runner dies, delete leftover pods with `kubectl delete pods -l sandbox-run=<SANDBOX_K8S_RUN_LABEL>`.
  - **Images:** tasks with a prebuilt image (Terminal-Bench, DeepSWE) run it directly. For the others, the pod starts from the Dockerfile's `FROM` image and replays its steps (`RUN`/`ENV`/`WORKDIR`/`COPY`), since pods can't build images. In a multi-stage Dockerfile (Multi-SWE-bench's fetch a standalone Python in a helper stage), each stage that a `COPY --from` names is replayed in a short-lived pod of its own and the paths are copied across. A stage built `FROM` an earlier stage is rejected.
  - **Network:** `no-network` and allowlist tasks get a NetworkPolicy. A NetworkPolicy can't express wildcard hostnames, so Senior SWE-bench, whose agent allowlists contain them (`*.pythonhosted.org`, …), is Docker-only. Harbor's Docker environment enforces those with an egress proxy.
  - **Permissions:** the namespace needs `create`/`get`/`delete` on pods and networkpolicies and `create`/`get` on pods/exec, which `rob-poc` on Breithorn has.

**On Clariden:** the nodes are aarch64 and have no Docker, so `scripts/harbor/run_harbor_eval.sbatch` runs Harbor and the agent in a CPU job, with the task containers as Breithorn pods (`--sandbox k8s`, namespace `rob-poc` by default). The kubeconfig on Clariden authenticates through an OIDC login (kubelogin) that a batch job can't renew. Log in first (`kubectl get pods` works on the login node), and keep the run within the token's lifetime. Three minutes before the job's time limit the run is stopped (`summary.json` keeps the trials that finished, marked `interrupted`), and at the end the job deletes the pods and NetworkPolicies left with its label, `sandbox-run=harbor-<job id>`.

Senior SWE-bench's verifier runs the task's tests, then LLM judges and a validation agent (Claude Sonnet 4.6 and Haiku 4.5 by default) that are part of its reward. Export `ANTHROPIC_API_KEY` (or `PORTKEY_API_KEY`) before the run; it's passed to the verifier as is. Without one, every trial ends with an empty reward and is listed under `evaluation_errors`.

A trial that ends without a verifier reward (pod not scheduled, OOM-killed, build replay failed, API dropped) is listed under `evaluation_errors` in `summary.json`, and the run exits non-zero. A trial whose agent failed or timed out is still scored. Exception: Aider Polyglot's `tests/test.sh` exits without writing a reward when there is nothing to build or test (e.g. the agent deleted `Cargo.toml`), so there a missing reward file scores 0 (those trials are listed under `no_reward_file`).

**Aider Polyglot is patched.** Its exercises are Exercism's, which hide all but the first test until a learner un-hides them, and the published verifiers don't un-hide them for Rust (`#[ignore]`), JavaScript (`xtest`/`xit`), C++ (`EXERCISM_RUN_ALL_TESTS`) and 3 Java tasks (`@Disabled` variants), 107 of the 225 tasks: there a solution that passes the first test scores 1. The runner downloads the pinned dataset into `<work-dir>/tasks/aider_polyglot/` and patches it to run every test, as Aider's own benchmark does (`scripts/harbor/task_patches.py`), except Rust `macros`' 10 compile-fail tests, whose fixture files the dataset leaves out; `summary.json` records this as `patched_with`. A C++ test failure also fails the build there, which the patched `test.sh` scores 0 instead of leaving no reward. Scores from before this change (and other Harbor-based Aider Polyglot numbers) are not comparable.

`summary.json` is rewritten after each benchmark. If the run is stopped (SIGTERM, e.g. a Slurm time limit), the current benchmark is summarized from the trials that finished and marked `interrupted`.

The results go to W&B whenever a W&B key is available (`WANDB_API_KEY` or `scripts/wandb_api_key.txt`), by default to `apertus/apertus-1.5-post-training-v0.1`; `--wandb-entity`/`--wandb-project` change that, and `--no-wandb` (or `WANDB_MODE=disabled`) skips the upload, e.g. for a smoke test. They are uploaded as one run named `--name` (`--gold` runs are not uploaded), through the same `ModelEvaluation` structure and `upload_multi_model_results` as lm-eval-harness and Inspect runs. Each benchmark gets its `reward` (or its named verifier metrics), `n_trials`, `n_scored` and `n_errors`, plus a few passed, failed and unscored trials as samples. It uploads also when some trials were not scored, so check `n_errors`. To upload an existing run (e.g. one that was stopped): `python -m scripts.results.update_wandb_harbor --entity <e> --project <p> --name <name> --work-dir logs/harbor/<name>` from the Harbor venv.

```bash
# Pipeline check without a model: reference solutions, one task per benchmark, pods on Breithorn
SANDBOX_K8S_NAMESPACE=rob-poc scripts/harbor/run_harbor_eval.sh --sandbox k8s --gold --limit 1 \
  --benchmark swe_bench,swe_bench_pro,terminal_bench_2,terminal_bench_4,deep_swe,slop_code_bench,aider_polyglot

# Multi-SWE-bench on Breithorn pods (its helper build stage runs in a pod of its own)
SANDBOX_K8S_NAMESPACE=rob-poc scripts/harbor/run_harbor_eval.sh --sandbox k8s --gold --limit 1 --benchmark multi_swe_bench

# Senior SWE-bench needs a Docker daemon and a key for its LLM graders
ANTHROPIC_API_KEY=... scripts/harbor/run_harbor_eval.sh --benchmark senior_swe_bench --gold --limit 1 --sandbox docker

# A model on Clariden (debug partition smoke test)
sbatch -p debug -t 1:25:00 scripts/harbor/run_harbor_eval.sbatch --benchmark swe_bench,terminal_bench_2 \
  --model RCP-AIaaS/Qwen/Qwen3.6-35B-A3B --api-base-url https://api.swissai.svc.cscs.ch/v1 --limit 1 --no-wandb
```

`scripts/aaii/run_terminal_bench.sh` (the same protocol, which evals-svc's Terminal-Bench suite delegates to) is unchanged.

With `--attempts <k>` > 1, `summary.json` also has Harbor's `pass_at_k` per benchmark. Harbor computes it itself and counts every trial without a reward as a failure, including the evaluation errors that `mean_reward` leaves out. Artificial Analysis's Terminal-Bench protocol:

```bash
scripts/harbor/run_harbor_eval.sh --benchmark terminal_bench_2_1 --attempts 3 \
  --model CSCS-Inference/swiss-ai/Apertus-v1.5-8B --api-base-url https://api.swissai.svc.cscs.ch/v1
```
