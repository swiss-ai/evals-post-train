# Serving a model and evaluating it in one job

`scripts/direct_serving/job.py` writes one Slurm job that serves a model and runs eval commands against it on the same nodes. The model is loaded once, however many commands follow, and the evals call the server on `127.0.0.1`: no serving gateway between them, so its 503s, rate limits and header timeout never reach a run.

```
sbatch
|- srun   model server(s)        model-launch's own job script
'- srun   eval command 1, 2, ..  in the eval image, one after another
```

The serving half is model-launch's rendering of a serving job (head and follower scripts, Ray across the nodes of a replica), with OpenTela, metrics and telemetry off, so a model on several nodes works. Only the end of that script is replaced: model-launch waits for the server to exit, this job waits for `/health`, runs the commands, and ends. If the server dies before it is ready or during a command, the job stops that command and ends with status 1.

## Run it

On a Clariden login node, with `swiss_ai_model_launch` installed ([setup](lm-eval-harness.md#judge-model-launching)). 2-node Apertus 2 Chonk, one benchmark:

```bash
python -m scripts.direct_serving.job \
  --model /capstor/store/cscs/swissai/infra01/apertus_checkpoints/v2/hf/chonk/<checkpoint> \
  --served-model-name chonk --nodes 2 --time 04:00:00 \
  --environment containers/env_vllm_apertus2.toml \
  --framework-args "--trust-remote-code --dtype bfloat16 --max-model-len 4096 --tensor-parallel-size 1 --data-parallel-size 8 --data-parallel-size-local 4 --data-parallel-backend ray --enable-expert-parallel --all2all-backend allgather_reducescatter --moe-backend triton --numa-bind --load-format runai_streamer --gpu-memory-utilization 0.95 --max-num-seqs 8 --max-num-batched-tokens 512 --api-server-count 4" \
  --eval 'scripts/inspect_ai/run_inspect_eval.sh --task gsm8k --model chonk --api-base-url "$EVAL_API_BASE_URL"' \
  --submit
```

Without `--submit` it prints the job script instead, to submit another way. Each `--eval` is a command run with `bash -c` from this checkout inside the eval image, in the order given; give it as many as you like. Every command finds the server's URL in `$EVAL_API_BASE_URL`. The commands run inline in the job, so use the runners that do (`run_inspect_eval.sh`, `run_harbor_eval.sh`, `run_evalscope_eval.sh`); lm-eval's `launch_evaluations.sh` submits Slurm jobs of its own.

| Option | |
|---|---|
| `--environment` | EDF of the serving image. `containers/env_vllm_apertus2.toml` has the Apertus 2 vLLM image; `containers/env_vllm.toml` is the default one. |
| `--framework-args` | After `--model`, `--served-model-name` and `--host`, which the launcher sets. The layout above is the validated 2-node one for Chonk: it needs `--api-server-count 4` (more front ends race at startup) and the engine takes about 8.5 minutes to be ready. |
| `--eval-image-commit` | The [prebuilt eval image](lm-eval-harness.md#prebuilt-evaluation-image-no-installs-at-run-time) the commands run in. Default: the last commit of this checkout that changed what the image is built from (`containers/Dockerfile.eval`, `requirements/`, the workflow), since CI builds an image only for those. The job refuses to start when ghcr does not have it yet (the push of that commit is still building, or was never pushed), and writes the EDF that starts it. |
| `--mount` | A directory to mount in the eval image besides this checkout, `/users/$USER` and the scratch directories. Repeatable. |
| `--time` | The model's load plus every command: the job has no other limit. |
| `--ready-timeout` | Seconds to wait for the server (default 3600). |

All commands share one server, so they share its `--max-model-len`, chat template and plugins. A benchmark that needs other server settings gets its own job.

## Exit status

The job ends with status 1 if the server died or any command failed. A failed command does not stop the next one.
