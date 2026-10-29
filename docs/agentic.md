# Agentic coding and terminal benchmarks

Benchmarks where an agent works inside a task container: Harbor (the default) and EvalScope, with task containers in local Docker or as Kubernetes pods (`scripts/common/k8s_sandbox.py`).

## Harbor (SWE-bench, SWE-bench Pro and other agentic coding and terminal benchmarks)

`scripts/harbor/run_harbor_eval.sh` runs agentic benchmarks through [Harbor](https://github.com/harbor-framework/harbor), the harness behind Terminal-Bench. It is the default runner for agentic benchmarks here, SWE-bench and SWE-bench Pro included; EvalScope (next section) is the alternative for those two. Every task is a container the agent works in. The task's own verifier scores the result as a reward from 0 to 1.

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
| `aider_polyglot` | Aider Polyglot | 225 |

Harbor Hub datasets are pinned by content digest and the older registry datasets by version. Harbor runs in its own venv (`requirements/harbor-runtime.txt`, Python ≥ 3.12, installed with uv). The default agent is `terminus-2`, which drives a tmux session and needs no tool-call parser on the served model. It calls `--model` through litellm as `openai/<model>` at `--api-base-url`. `--gold` runs Harbor's `oracle` agent, which replays each task's reference solution, to check the environments and verifiers. Scores and run metadata go to `logs/harbor/<name>/summary.json`: per benchmark, the mean `reward`, or for verifiers that report several named metrics (SlopCodeBench) the mean of each.

`--sandbox` picks where the task containers run:

- `docker`: Harbor's own Docker environment (`scripts/harbor/docker_environment.py` builds its network-allowlist proxy for the task images' platform). On Apple Silicon the linux/amd64 images run emulated; Erlang crashes under that emulation, so Senior SWE-bench's Elixir tasks only build on an amd64 host.
- `k8s`: one unprivileged pod per trial through `scripts/harbor/k8s_environment.py`, using the current kubeconfig, with no Docker daemon needed. The pods come from `scripts/common/k8s_sandbox.py` (shared with the EvalScope runner) and are pinned to amd64 nodes. Pod settings come from `SANDBOX_K8S_*` env vars (namespace, run label, CPU and memory requests, image pull secret, ...; see the module docstring). If the runner dies, delete leftover pods with `kubectl delete pods -l sandbox-run=<SANDBOX_K8S_RUN_LABEL>`.
  - **Images:** tasks with a prebuilt image (Terminal-Bench, DeepSWE) run it directly. For the others, the pod starts from the Dockerfile's `FROM` image and replays its steps (`RUN`/`ENV`/`WORKDIR`/`COPY`), since pods can't build images. In a multi-stage Dockerfile (Multi-SWE-bench's fetch a standalone Python in a helper stage), each stage that a `COPY --from` names is replayed in a short-lived pod of its own and the paths are copied across. A stage built `FROM` an earlier stage is rejected.
  - **Network:** `no-network` and allowlist tasks get a NetworkPolicy. A NetworkPolicy can't express wildcard hostnames, so Senior SWE-bench, whose agent allowlists contain them (`*.pythonhosted.org`, …), is Docker-only. Harbor's Docker environment enforces those with an egress proxy.
  - **Permissions:** the namespace needs `create`/`get`/`delete` on pods and networkpolicies and `create`/`get` on pods/exec, which `rob-poc` on Breithorn has.

**On Clariden:** the nodes are aarch64 and have no Docker, so `scripts/harbor/run_harbor_eval.sbatch` runs Harbor and the agent in a CPU job, with the task containers as Breithorn pods (`--sandbox k8s`, namespace `rob-poc` by default). The kubeconfig on Clariden authenticates through an OIDC login (kubelogin) that a batch job can't renew. Log in first (`kubectl get pods` works on the login node), and keep the run within the token's lifetime.

Senior SWE-bench's verifier runs the task's tests, then LLM judges and a validation agent (Claude Sonnet 4.6 and Haiku 4.5 by default) that are part of its reward. Export `ANTHROPIC_API_KEY` (or `PORTKEY_API_KEY`) before the run; it's passed to the verifier as is. Without one, every trial ends with an empty reward and is listed under `evaluation_errors`.

A trial that ends without a verifier reward (pod not scheduled, OOM-killed, build replay failed, API dropped) is listed under `evaluation_errors` in `summary.json`, and the run exits non-zero. A trial whose agent failed or timed out is still scored.

`summary.json` is rewritten after each benchmark. If the run is stopped (SIGTERM, e.g. a Slurm time limit), the current benchmark is summarized from the trials that finished and marked `interrupted`.

`--wandb-entity <entity> --wandb-project <project>` uploads the results to W&B as one run named `--name`, through the same `ModelEvaluation` structure and `upload_multi_model_results` as lm-eval-harness and Inspect runs. Each benchmark gets its `reward` (or its named verifier metrics), `n_trials`, `n_scored` and `n_errors`, plus a few passed, failed and unscored trials as samples. It uploads also when some trials were not scored, so check `n_errors`. To upload an existing run (e.g. one that was stopped): `python -m scripts.results.update_wandb_harbor --entity <e> --project <p> --name <name> --work-dir logs/harbor/<name>` from the Harbor venv.

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
  --model RCP-AIaaS/Qwen/Qwen3.6-35B-A3B --api-base-url https://api.swissai.svc.cscs.ch/v1 --limit 1
```

`scripts/aaii/run_terminal_bench.sh` (the same protocol, which evals-svc's Terminal-Bench suite delegates to) is unchanged.

With `--attempts <k>` > 1, `summary.json` also has Harbor's `pass_at_k` per benchmark. Artificial Analysis's Terminal-Bench protocol:

```bash
scripts/harbor/run_harbor_eval.sh --benchmark terminal_bench_2_1 --attempts 3 \
  --model CSCS-Inference/swiss-ai/Apertus-v1.5-8B --api-base-url https://api.swissai.svc.cscs.ch/v1
```

---

## EvalScope (SWE-bench, SWE-bench Pro), the alternative to Harbor

`scripts/evalscope/run_evalscope_eval.sh` runs exactly two benchmarks through [EvalScope](https://github.com/modelscope/evalscope), as an alternative to Harbor (previous section) for these two:

| `--benchmark` | What | Images |
|---|---|---|
| `swe_bench` | SWE-bench Verified, 500 tasks (EvalScope's `swe_bench_verified_agentic`) | `swebench/sweb.eval.x86_64.*` (prebuilt, pulled) |
| `swe_bench_pro` | SWE-bench Pro **v1**, 731 tasks (see below) | `jefzda/sweap-images:<tag>` |

Both use the same protocol. The agent works through a multi-turn bash loop inside the instance's own container and submits a `git diff`. EvalScope then applies that patch in a fresh container and runs the instance's tests. An instance counts as resolved when every fail-to-pass and pass-to-pass test passes. Pass `--benchmark swe_bench,swe_bench_pro` to run both in one go.

The script installs EvalScope into its own venv from `requirements/evalscope-runtime.txt` (pinned exactly). It can't share the lm-eval/Inspect environment because EvalScope's `math-verify` pin conflicts with that environment's `antlr4` pin. The script takes `--model`/`--api-base-url` and uses the same serving-key fallback as `run_inspect_eval.sh`. It writes EvalScope's own outputs plus a `summary.json` to `logs/evalscope/<name>/`. Besides the scores, `summary.json` records what's needed to reproduce the run: the EvalScope version, the Hugging Face dataset revision for each benchmark (both pinned in `scripts/evalscope/run.py`, override with `--dataset-revision <benchmark>=<sha>`) and the `SWE-bench_Pro-os` commit (EvalScope pins `ca10a60`).

SWE-bench is scored by `scripts/evalscope/swe_bench_scoring.py`, not EvalScope's own scorer. The evaluation itself is the same: the same `swebench` test spec, eval script, patch-apply fallbacks and grading. EvalScope's version uses the Docker SDK directly, which never pulls the image and doesn't work without a Docker daemon. This one creates its container through EvalScope's sandbox layer, so it pulls the image and works on both `--sandbox` engines.

**This is SWE-bench Pro v1.** EvalScope implements v1: 731 tasks, `jefzda/sweap-images`, and the v1 run scripts. On 2026-09-22 Scale released V2, made it the dataset's `default` config, and kept v1 as config `v1`. V2 has 642 validated tasks shipped as [Harbor](https://github.com/harbor-framework/harbor) tasks with `ghcr.io/scaleapi/swe-bench_pro-v2` images. Scoring V2 rows with the v1 pipeline fails even the reference patches, because the test patch it applies doesn't exist in the v1 images. The runner therefore pins the dataset to its last v1 commit, `7ab5114` (override with `--dataset-revision swe_bench_pro=<sha>`). Its scores are v1 numbers and aren't comparable with V2 ones.

Every sample runs in its own sandbox container (`--sandbox`):

- `docker`: EvalScope's own path, using a local Docker daemon.
- `k8s`: one unprivileged pod per sandbox in the current namespace, via `scripts/common/k8s_sandbox.py`, with no Docker daemon needed. This is the same approach as evals-svc's Terminal-Bench trial pods. The service account needs `create`/`get`/`delete` on `pods` and `create`/`get` on `pods/exec`. The images are linux/amd64, so the pods are pinned to amd64 nodes. Pod settings come from `SANDBOX_K8S_*` env vars (see the module docstring). The runner itself doesn't have to run in the cluster: from a laptop, `--sandbox k8s` uses your current kubeconfig context, and `SANDBOX_K8S_NAMESPACE` picks the namespace (e.g. on Breithorn). Pods started that way have no owner pod, so if the runner dies, delete leftovers with `kubectl delete pods -l sandbox-run=<SANDBOX_K8S_RUN_LABEL>`.

Notes:

- The default `swe_bench_toolcall` strategy needs a served model with tool calling (vLLM `--enable-auto-tool-choice` plus a parser, sglang `--tool-call-parser`). `--strategy swe_bench_backticks` drives the agent through plain text instead.
- The step cap is 250 by default, and each instance's test run can take up to `eval_timeout` (3600 s). A full run is long.
- Each sandbox gets 2 CPUs. With `--sandbox docker` it gets 4 GiB (ms-enclave's default). Some SWE-bench Pro instances need more: teleport's tests peak at about 7.4 GiB, element-web fails the same way at 4 GiB, and EvalScope's docs mention NodeBB. So `--sandbox k8s` defaults to an 8 GiB limit with a 4 GiB request (`SANDBOX_K8S_MEMORY_REQUEST=4g`): an 8 GiB request often doesn't schedule on Breithorn's amd64 nodes, while this one does and still lets a pod grow to 8 GiB. Memory above the request isn't guaranteed: under node memory pressure such pods are evicted first. Setting `memory_limit` in `--sandbox-config` replaces both defaults, and the request then equals the limit unless `SANDBOX_K8S_MEMORY_REQUEST` is set.
- A sample whose evaluation fails for an infrastructure reason still scores as unresolved in EvalScope's report. Examples: an OOM-killed container, a pod that never schedules, or the cluster API dropping out. Such samples are listed under `evaluation_errors` in `summary.json`, and the run then exits non-zero. A patch that doesn't apply, or tests that time out, are the model's failures and aren't counted there.
- `--limit N` takes the first N instances, which are mostly from one repository. Add `--shuffle` (fixed seed 42) for a sample across repositories that is still the same on every run.
- The instance images total many GB and are pulled from DockerHub, so expect pull rate limits. Set `SANDBOX_K8S_IMAGE_PULL_SECRET` for an authenticated pull.

```bash
# Pipeline check without a model: score each instance's reference patch (every instance should resolve)
scripts/evalscope/run_evalscope_eval.sh --benchmark swe_bench,swe_bench_pro --gold --limit 20 --shuffle

# Smoke test: two instances of each benchmark, two sandboxes at a time
scripts/evalscope/run_evalscope_eval.sh --benchmark swe_bench,swe_bench_pro --model <served model with tool calling> \
  --api-base-url https://api.swissai.svc.cscs.ch/v1 --limit 2 --max-concurrency 2

# Inside a k8s Job (the default there): pods on amd64 nodes, more memory per sandbox
scripts/evalscope/run_evalscope_eval.sh --benchmark swe_bench_pro --model <served model> --api-base-url <url> \
  --sandbox k8s --sandbox-config '{"memory_limit": "8g", "cpu_limit": 4}' --max-concurrency 8

# From a laptop, sandboxes as pods in a Breithorn namespace; patient retries for a shared, rate-limited key.
# Keep the laptop awake (caffeinate -i on macOS): a sleeping laptop loses DNS and the run's samples error out.
SANDBOX_K8S_NAMESPACE=<namespace> SANDBOX_K8S_RUN_LABEL=my-run \
  caffeinate -i scripts/evalscope/run_evalscope_eval.sh --benchmark swe_bench,swe_bench_pro \
  --model CSCS-Inference/swiss-ai/Apertus-v1.5-8B --api-base-url https://api.swissai.svc.cscs.ch/v1 \
  --sandbox k8s --limit 1 --max-concurrency 1 \
  --generation-config '{"retries": 100, "retry_interval": 60}'
```
