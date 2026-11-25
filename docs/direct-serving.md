# Serving a model and evaluating it in one job

`scripts/direct_serving/job.py` writes one Slurm job that serves a model and runs eval commands against it on the same nodes. The model loads once, however many commands follow, and the evals call the server on `127.0.0.1`: no serving gateway sits between them, so its 503s, rate limits and header timeout never reach a run.

```
sbatch
|- srun   model server(s)        model-launch's own job script
'- srun   eval command 1, 2, ..  in the eval image, one after another
```

The serving half is model-launch's rendering of a serving job (head and follower scripts, Ray across the nodes of a replica) with OpenTela, metrics and telemetry off, so a model on several nodes works. Every serving node runs with `RAY_TMPDIR=/tmp/ray-$SLURM_JOB_ID` (before `--pre-launch-cmds`, which can override it): Ray records its cluster address in a file under its temp dir, and with a temp dir shared between jobs (`TMPDIR` on scratch) a second job overwrites the first one's address and the first one's vLLM fails with `Failed to connect to Ray cluster`. Only the end of that script is replaced: this job waits for `/health`, runs the commands, and ends, where model-launch waits for the server to exit. If the server dies before it is ready or during a command, the job stops that command and ends with status 1.

## Preflight

A model takes up to half an hour to load, and a missing key used to show only after that: the runner checks its keys when it starts. So the job starts by running every `--eval` command once with `EVAL_PREFLIGHT=1`, in the eval image (which it pulls there, on the first run), before the model loads. Each runner (`scripts/common/preflight.sh`: `preflight_done`) stops right after its own checks of arguments and keys, before it installs, downloads or submits anything, and exits 0 (`preflight ok: <runner>`). The checks that matter most: a W&B upload that was asked for (`--wandb-entity`/`--wandb-project`, which the suites give) needs `WANDB_API_KEY`, `--sandbox opensandbox` needs `OPENSANDBOX_API_KEY`, and an lm-eval task whose Hugging Face dataset is gated (`gpqa*`, `flores_plus*`, `bouquet*`, `harmbench`, listed in `GATED_TASKS_PATTERN` of `launch_evaluations.sh`: add a task when its dataset is gated) needs an `HF_TOKEN` (environment or `scripts/secrets.env`) or a logged-in Hugging Face token file (not needed with `HF_HUB_OFFLINE=1` or `HF_DATASETS_OFFLINE=1`). If any command fails, the job logs every failure, never loads the model and ends with status 1. There is no switch to skip it: it takes seconds once the image is on the node.

A command of your own has to source `scripts/common/preflight.sh` and call `preflight_done` after its checks, or the preflight would run the whole eval before the model is there: each command gets 15 minutes (the first includes the image pull), then it is killed and counted as failed (exit 124).

## Run it

On a Clariden login node, from the root of this checkout; `scripts/direct_serving/job.sh` runs `job.py` from a cached venv of `requirements/direct-serving.lock` (the model-launch commit it renders its serving job from, pinned; `uv` finds or fetches the Python, 3.11 or newer), so there is nothing to install by hand. `DIRECT_SERVING_VENV` points it at another venv instead. 2-node Apertus 2 Chonk, one benchmark:

```bash
scripts/direct_serving/job.sh \
  --model /capstor/store/cscs/swissai/infra01/apertus_checkpoints/v2/hf/chonk/<checkpoint> \
  --served-model-name chonk --nodes 2 --time 04:00:00 \
  --environment containers/env_vllm_apertus2.toml \
  --framework-args "--trust-remote-code --dtype bfloat16 --max-model-len 4096 --tensor-parallel-size 1 --data-parallel-size 8 --data-parallel-size-local 4 --data-parallel-backend ray --enable-expert-parallel --all2all-backend allgather_reducescatter --moe-backend triton --numa-bind --load-format runai_streamer --gpu-memory-utilization 0.95 --max-num-seqs 8 --max-num-batched-tokens 512 --api-server-count 4" \
  --eval 'scripts/inspect_ai/run_inspect_eval.sh --task gsm8k --model chonk --api-base-url "$EVAL_API_BASE_URL"' \
  --submit
```

`--submit` prints the job id (`sbatch --parsable`); without it the job script is printed instead, to submit another way. The checkout is mounted into the eval image and the commands run from it, so it must be on storage the compute nodes see. Each `--eval` is first run once with `EVAL_PREFLIGHT=1` before the model loads (see [Preflight](#preflight)), then with `bash -c` from this checkout inside the eval image, in the order given; pass as many as you like, from any framework. Every command finds the server's URL in `$EVAL_API_BASE_URL`.

## The frameworks' commands

Use each framework's own launcher with `--api-base-url "$EVAL_API_BASE_URL"` and the name the server answers to (`--served-model-name`):

| Framework | Command |
|---|---|
| lm-eval-harness | `bash scripts/lm_eval_harness/launch_evaluations.sh single --task gsm8k --model <name> --api-base-url "$EVAL_API_BASE_URL" --api-model-name <name> --tokenizer <HF name or path> --failure-policy fail-fast` |
| Inspect AI | `scripts/inspect_ai/run_inspect_eval.sh --task gsm8k --model <name> --api-base-url "$EVAL_API_BASE_URL"` |
| EvalScope | `scripts/evalscope/run_evalscope_eval.sh --benchmark swe_bench --sandbox <k8s or opensandbox> --model <name> --api-base-url "$EVAL_API_BASE_URL"` |
| Harbor | `scripts/harbor/run_harbor_eval.sh --benchmark swe_bench --sandbox <k8s or opensandbox> --model <name> --api-base-url "$EVAL_API_BASE_URL"` |

- **lm-eval submits Slurm jobs of its own**, and a job on another node could not reach the server on `127.0.0.1`. So every command finds `sbatch` and `srun` stand-ins first on `PATH` (`scripts/direct_serving/inline_slurm/`), which run the batch script or command in place, in the job's allocation. They can't emulate job arrays, so give lm-eval `--failure-policy fail-fast`. They are a bridge for every command; an inline executor in lm-eval's launcher would replace them. The tokenizer must be a Hugging Face name or a path, not the served name.
- **Harbor and EvalScope's SWE-bench** run their task containers in a sandbox, which the compute node must be able to reach (Kubernetes access, or an OpenSandbox key in the environment or `scripts/secrets.env`, as for any run). The job does not provide one.
- **Keys:** the runners want a key for `--api-base-url` and the server in the job has none, so the job sets `TARGET_API_KEY` (the runners' variable for that endpoint's key) to `local` unless it is already set. `CSCS_SERVING_API` is not needed to serve or evaluate; it remains the key of a judge model on the CSCS serving API, read from the environment or `scripts/secrets.env`.
- **Apertus 2 over chat requests** needs the `apertus-common` plugin in the server ([setup](thinking.md#apertus-2-on-v1chatcompletions-temporary)).

## From a suite

A [suite](suites.md) setup with a `launch` block says how its model is served: `--suite` and `--setup` fill in the checkpoint, the serving EDF, the nodes, the time and the framework arguments (the launch block's, then the setup's `serving` arguments, then `--framework-args`). The model is served under the setup's name unless `--served-model-name` says otherwise. What the command line names wins.

```bash
scripts/direct_serving/job.sh --suite apertus2/chonk_sft --setup chonk-lc-16k-v0.2-sft \
  --pre-launch-cmds "<install the apertus-common plugin>" \
  --eval 'scripts/inspect_ai/run_inspect_eval.sh --task gsm8k --model chonk-lc-16k-v0.2-sft --api-base-url "$EVAL_API_BASE_URL" --limit 5' \
  --submit
```

The Chonk setups' `serving` arguments need the `apertus-common` plugin, which the Apertus 2 image does not contain. Pass its install as `--pre-launch-cmds` ([how](thinking.md#apertus-2-on-v1chatcompletions-temporary)); without it the server exits at start-up with `invalid tool call parser: apertus2`.

## Options

| Option | |
|---|---|
| `--account`, `--partition`, `--qos`, `--reservation` | The job's Slurm options. The partition is `normal` unless you name another; the account defaults to your primary group. |
| `--environment` | EDF of the serving image. `containers/env_vllm_apertus2.toml` has the Apertus 2 vLLM image; `containers/env_vllm_0.30.0.toml` is the default one (model-launch's vLLM 0.30.0 image, arm64, with its multi-node network settings; Ray included). For Beverin (AMD MI300A, gfx942) use the ROCm EDFs: `env_vllm_apertus2_rocm.toml` (Chonk), `env_vllm_apertus1p5_rocm.toml` (Apertus 1.5), `env_vllm_rocm_0.29.toml` and `env_vllm_rocm_0.30.0.toml` (stock vLLM). They carry no RCCL/CXI hooks (with them RCCL hangs under model-launch). |
| `--framework-args` | After `--model`, `--served-model-name` and `--host`, which the launcher sets. The layout above is the validated 2-node one for Chonk: it needs `--api-server-count 4` (more front ends race at startup), and the engine takes about 8.5 minutes to be ready. |
| `--suite`, `--setup` | A suite setup's `launch` block ([above](#from-a-suite)); then `--model`, `--environment` and `--served-model-name` are optional, and so are `--nodes` and `--time`. |
| `--pre-launch-cmds` | A shell command every serving node runs before the server starts. |
| `--eval-image-commit` | The [prebuilt eval image](lm-eval-harness.md#prebuilt-evaluation-image-no-installs-at-run-time) the commands run in. Default: the last commit of this checkout that changed what the image is built from (`containers/Dockerfile.eval`, `requirements/`, the workflow), since CI builds an image only for those. The job refuses to start when ghcr does not have it yet (its build is still running, or the commit was never pushed), and writes the EDF that starts it. |
| `--mount` | A directory to mount in the eval image besides this checkout, `/users/$USER`, the scratch directories and the model's directory (`--model`, or the setup's `launch.model_path`; skipped when one of those already holds it, so the evals can read its tokenizer). Repeatable. |
| `--time` | The model's load plus every command: the job has no other limit. |
| `--ready-timeout` | Seconds to wait for the server (default 3600). |

All commands share one server, so they share its `--max-model-len`, chat template and plugins. A benchmark that needs other server settings gets its own job.

## Exit status

The job ends with status 1 if the server died or any command failed. A failed command does not stop the next one.
