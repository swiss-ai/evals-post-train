# Suites

A suite writes down a comparison once: its benchmarks, how each is repeated and scored, and the setups a model can run under. It lives in `configs/suites/<name>.json` (Apertus 2 suites in `configs/suites/apertus2/`, named `apertus2/<name>`); `scripts/common/suites.py` reads it and knows no model.

```bash
python -m scripts.common.suites show apertus2/chonk_sft   # title and setups
python -m scripts.common.suites plan apertus2/chonk_sft --setup apertus-1.5-70b-thinking
python -m scripts.common.suites flags apertus2/chonk_sft --setup apertus-1.5-70b-thinking --task ifbench
# --thinking --think-start-token '<|inner_prefix|>' --think-end-token '<|inner_suffix|>' --max-length 32768 --repeats 4
```

`show` prints the title and the setups. `plan` prints one row per benchmark (label, task, launcher flags, what it reports, the results key). Give `flags` output to the launcher next to `--task`. `run` starts a suite: see [Running a suite](#running-a-suite).

## File

| Part | Fields |
|---|---|
| `benchmarks` (list, table order) | `label` (the table row), `task`, `framework`, `repeats {n, via}`, `metric {label, key}`, optional `sampling`, optional `task_args` (inspect only), optional `flags` |
| `setups` (by name) | `max_length` (the model's maximal window), `sampling` (default profile), `flags` (by framework: thinking, chat format, tokenizer) |

- `repeats.via` is `flag` when the task has no repeats of its own and the launcher gets `--repeats n` (lm_eval) or `-- --epochs n` (inspect), or `task` when its yaml already repeats `n` times (AIME 8, the self-consistency tasks 4). The flag would apply the repeats twice, so nothing is passed.
- `metric.label` is what the table shows (`mean@4`); it must agree with `repeats.n`. `metric.key` is the key in a run's results that holds the value, `null` until it has been checked against a finished run.
- `sampling` names a profile of `configs/sampling_defaults.json` (`thinking`, `non_thinking`, `greedy`); a suite holds no temperature or top-p. A setup names its default and a benchmark may name another. A setup with a thinking flag (`--thinking`, `--enable-thinking`) is refused if a benchmark would sample with a temperature of 0. `plan` shows the profile per row; the launcher does not read it yet.
- `flags` is an object keyed by framework, with an entry (`[]` for none) for every framework the suite's benchmarks use: a thinking switch means `--thinking ...` to lm_eval and `--enable-thinking` to inspect. `flags_for` returns the setup's flags of the benchmark's framework, then what the suite adds: lm_eval `--max-length` and `--repeats`; inspect one `--task-arg key=value` per `task_args` entry and `-- --epochs n`. A setup's flags may not contain those. Inspect has no window flag, so `max_length` is only passed to lm_eval; keep the served `--max-model-len` equal to it.
- **Harbor rows** (`swe_bench`, `framework: harbor`) take the setup's `flags.harbor` (every setup needs the entry): the thinking switch where the setup has no `reasoning_effort`, and `--agent-kwarg model_info={"max_input_tokens": <window>}` so terminus-2 knows the served window and summarises before it fills (Chonk's is 16384; at that size an unaided agent ran out of room in the SWE-bench smoke runs). The row's own `flags` carry `--sandbox opensandbox`. `run` launches it with `sbatch scripts/harbor/run_harbor_eval.sbatch --benchmark swe_bench ...`.
- `serving` (optional) lists the arguments the server itself needs, such as Apertus 2's `--tokenizer-mode apertus2 --reasoning-parser apertus2 ...` ([thinking](thinking.md#apertus-2-on-v1chatcompletions-temporary)); nothing in a suite runs them; `scripts/direct_serving/job.py --suite` appends them to the setup's `launch` arguments.
- `launch` (optional, `{model_path, environment, nodes, time, framework_args}`) says how the setup's model is served by [`scripts/direct_serving/job.py`](direct-serving.md#from-a-suite): the checkpoint, the serving EDF (a path in this repository), the nodes of the one replica, the wall time and the layout arguments (`framework_args`, a list; not `--model`, `--served-model-name`, `--host` or `--port`, which the launcher sets). Only the Chonk setups have one so far. They are served with a 32768 window (`--max-model-len`, `max_length`), which BFCL v4 needs: its memory categories send prompts over 16k tokens and a 16384 server answers them with HTTP 400. The checkpoint's `max_position_embeddings` is 16384, so vLLM only starts with `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` (the setup's `launch.env`) and what the model does past 16k is untrained. `env` (optional, variable name to string) is exported on every serving node before the server starts, ahead of `--pre-launch-cmds`. The partition, the plugin install and the sandbox are not part of it.
- `tokenizer` (optional, `{repo, revision, path}`) pins a tokenizer in a git repository by full commit sha. lm_eval rows get `--tokenizer <local copy>` (a setup's lm_eval flags may not hold `--tokenizer`); `flags` fetches the directory once into `$EVAL_ENV_CACHE_ROOT/tokenizers/<revision>/` (needs git access to the repository) and `plan` only shows where it will be. Apertus 2 needs this: its format has no Jinja template, and the verified tokenizer is `tokenizers/Apertus_2_instruct` of `swiss-ai/apertus-omni-tokenizer` at `b801ff1`, the revision `apertus-common` names.
- **Chat template** ([per framework](../README.md#chat-template)): a suite holds no template flag of its own; it is part of the setup's `flags`. lm_eval rows of a thinking setup get `--thinking`, which forces the template on; every lm_eval row here is generative, so no `--chat-template-by-type` is needed (add it to a setup that also runs log-likelihood tasks). Inspect, Harbor and EvalScope send chat requests and the server applies the template, so a setup's `inspect` flags carry only the thinking switch (`--enable-thinking`, which reaches the template as `chat_template_kwargs`).
- `reasoning_effort` (optional, per setup) becomes `--reasoning-effort <level>` on every lm_eval, inspect, evalscope and harbor row run under that setup, and the setup's flags may not set it. `chonk-lc-16k-v0.2-sft` uses `high` for all benchmarks: Chonk only reasons at `high` on non-math tasks (at `medium` its think frame is empty). Apertus 2 takes `low`, `medium` or `high`; the other setups set none.
- `flags` (optional, per benchmark) are launcher flags for that row under every setup, after the setup's own: `--sandbox opensandbox` for SWE-bench Verified. They may not hold what the suite produces (`--max-length`, `--repeats`, `--reasoning-effort`).
- `task_args` are the `-T` arguments of an inspect task, as strings: `{"cot": "true"}` for `mmlu_0_shot`, `{"fewshot": "0"}` for zero-shot `gsm8k`.
- `name`, `title` and `description` are required. A malformed file is refused with the field named (`benchmarks[5] (aime24): repeats.via must be one of ...`).

## Running a suite

```bash
python -m scripts.common.suites run apertus2/chonk_sft --api-base-url https://api.swissai.svc.cscs.ch/v1 \
  --model chonk-lc-16k-v0.2-sft=<served name> --model qwen3-32b-thinking=<served name> \
  --qos highprio --partition highprio --limit 5 --no-wandb           # prints the commands
... --submit                                                          # runs them
```

Every (model, benchmark) pair is its own launch and Slurm job: 12 benchmarks and 2 models are 24 launches. `--model SETUP=SERVED_NAME` names a setup and the name its model is served under, once per model. An lm_eval row is one `launch_evaluations.sh single` launch (with `--backend openai` and `--api-model-name` when `--api-base-url` is given); an inspect row is one `sbatch scripts/inspect_ai/run_inspect_eval.sbatch`; an evalscope row (`bfcl_v3`, `bfcl_v4`) is one `sbatch scripts/evalscope/run_evalscope_eval.sbatch`, with the setup's `evalscope` flags and its reasoning effort (every setup lists `evalscope: []`; bfcl_v3's requests carry them through EvalScope's own client, bfcl_v4's through `scripts/evalscope/bfcl_request.py`, which makes bfcl-eval's handler send them), and the score is a tool-call signal, not the leaderboard's. `--qos` and `--partition` go to the launcher or to `sbatch`. The first line says how many launches there are (`24 launches: 2 models x 12 benchmarks`). Without `--submit` nothing runs; with it the launches start in order and the first failure stops the rest. Every launch uploads to the W&B project named after the suite (`apertus/apertus2-chonk-sft`; `--no-wandb` skips it). Inspect, evalscope and Harbor rows write to `logs/<framework>/<suite>/<served name>/<task>` (`--logs-dir` / `--work-dir`), so two runs of one model no longer share a directory; lm_eval's logs already sit under the W&B project. Other options (`--limit`, `--no-wandb`, `--api-model-name`) are passed to every launch. Other frameworks have no launcher yet and are refused.

Whoever launches decides where and how a model runs beyond its `launch` block (partition, account). To add a comparison, copy `apertus2/chonk_sft.json`; to add a kind of model, add a setup. The `tasks_*.txt` lists the launcher modes read are separate and unchanged.
