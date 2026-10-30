# EvalScope

`scripts/evalscope/run_evalscope_eval.sh` runs two benchmarks through [EvalScope](https://github.com/modelscope/evalscope); [Harbor](harbor.md) runs them too, with a different agent and scoring, so the scores differ.

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
- `k8s`: one unprivileged pod per sandbox in the current namespace, via `scripts/common/k8s_sandbox.py`, with no Docker daemon needed. The service account needs `create`/`get`/`delete` on `pods` and `create`/`get` on `pods/exec`. The images are linux/amd64, so the pods are pinned to amd64 nodes. Pod settings come from `SANDBOX_K8S_*` env vars (see the module docstring). The runner itself doesn't have to run in the cluster: from a laptop, `--sandbox k8s` uses your current kubeconfig context, and `SANDBOX_K8S_NAMESPACE` picks the namespace (e.g. on Breithorn). Pods started that way have no owner pod, so if the runner dies, delete leftovers with `kubectl delete pods -l sandbox-run=<SANDBOX_K8S_RUN_LABEL>`.

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
  --api-base-url https://api.swissai.svc.cscs.ch/v1 --limit 2 --max-concurrency 2 --no-wandb

# Inside a k8s Job (the default there): pods on amd64 nodes, more memory per sandbox
scripts/evalscope/run_evalscope_eval.sh --benchmark swe_bench_pro --model <served model> --api-base-url <url> \
  --sandbox k8s --sandbox-config '{"memory_limit": "8g", "cpu_limit": 4}' --max-concurrency 8

# From a laptop, sandboxes as pods in a Breithorn namespace; patient retries for a shared, rate-limited key.
# Keep the laptop awake (caffeinate -i on macOS): a sleeping laptop loses DNS and the run's samples error out.
SANDBOX_K8S_NAMESPACE=<namespace> SANDBOX_K8S_RUN_LABEL=my-run \
  caffeinate -i scripts/evalscope/run_evalscope_eval.sh --benchmark swe_bench,swe_bench_pro \
  --model CSCS-Inference/swiss-ai/Apertus-v1.5-8B --api-base-url https://api.swissai.svc.cscs.ch/v1 \
  --sandbox k8s --limit 1 --max-concurrency 1 --no-wandb \
  --generation-config '{"retries": 100, "retry_interval": 60}'
```

**On Clariden:** the nodes are aarch64 and have no Docker, so `scripts/evalscope/run_evalscope_eval.sbatch` runs EvalScope and the agent in a CPU job, with the sandboxes as Breithorn pods (`--sandbox k8s`, namespace `rob-poc` by default). The kubeconfig on Clariden authenticates through an OIDC login (kubelogin) that a batch job can't renew. Log in first (`kubectl get pods` works on the login node), and keep the run within the token's lifetime. The pods are labelled `sandbox-run=evalscope-<job id>`, and the job deletes the ones left at the end. Three minutes before the time limit it stops the run; EvalScope writes `summary.json` only when a run finishes, so a stopped run has none.

```bash
sbatch -p debug -t 1:00:00 scripts/evalscope/run_evalscope_eval.sbatch --benchmark swe_bench,swe_bench_pro --gold --limit 2
```

The results go to W&B whenever a W&B key is available (`WANDB_API_KEY` or `scripts/wandb_api_key.txt`), by default to `apertus/apertus-1.5-post-training-v0.1`; `--wandb-entity`/`--wandb-project` change that, and `--no-wandb` (or `WANDB_MODE=disabled`) skips the upload, e.g. for a smoke test. They are uploaded as one run named `--name` (`--gold` runs are not uploaded), through the same `ModelEvaluation` structure and `upload_multi_model_results` as the other frameworks. Each benchmark gets its `accuracy` (the share of instances resolved), `n_samples` and `n_errors`, plus a few resolved, unresolved and failed-evaluation instances as samples (tests passed of those required, the failing tests, tokens, the patch). `accuracy` counts failed evaluations as unresolved, so with `n_errors` > 0 it is too low. To upload an existing run: `python -m scripts.results.update_wandb_evalscope --entity <e> --project <p> --name <name> --work-dir logs/evalscope/<name>` from the EvalScope venv.
