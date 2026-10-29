# lm-eval-harness pipeline

Launching lm-evaluation-harness evaluations on Clariden: the launcher, resumable chunked jobs, task suites, thinking metrics, the sbatch scripts and the container environment. Entry point: `scripts/lm_eval_harness/launch_evaluations.sh`.

## The Launch Script

`scripts/lm_eval_harness/launch_evaluations.sh` is the primary entry point for running evaluations. It supports three model selection modes and multiple benchmark suites.

### Benchmark Suites

The launcher selects a benchmark suite from its optional first positional argument (`<mode>`); omitting it defaults to `posttrain`. Apertus suites are defined in `configs/apertus/`, OLMo3 suites in `configs/olmo/`.

**Apertus suites** (`configs/apertus/`)

| Mode | Task list | Tasks | Description |
|------|-----------|-------|-------------|
| `default` | `tasks_default.txt` | 48 | Full Apertus 1.5 post-training suite: knowledge, math, code, reasoning, multilingual, instruction following, bias, cultural knowledge, safety |
| `posttrain` | `tasks_posttrain_final.txt` | 50 | Post-training suite incl. chat/arena (alpaca_eval, arena_hard_v01/v2), AIME, MATH-500, BFCL v3, Swiss AI Charter Alignment, o(r)bench, and system-prompt adherence (RealGuardrails: S-IFEval, TensorTrust, S-RuLES) |
| `best-of-k` | `tasks_best_of_k.txt` | 7 | Multi-repeat/self-consistency suite for math and code, with mean@k, majority-vote, and pass@k metrics |
| `gpt` | `tasks_gpt.txt` | 3 | Experimental AlpacaEval and Arena-Hard path for a future OpenAI GPT judge type in the Swiss-AI harness |
| `multi-lingual` | `tasks_multilingual.txt` | 12 | Multilingual-only subset (global_mmlu, mgsm, hellaswag_multilingual, include, cultural_bench, multi-if, aya_redteaming, ...) |
| `apertus-previous` | `tasks_english.txt` | 19 | Previous (Apertus 1.0) English benchmark suite |
| `pretrain` | `tasks_pretrain.txt` | 30 | Pretraining suite (base-model loglikelihood/MC variants, few-shot MMLU); logs to W&B project `apertus-1.5-pre-training-v0.0` |
| `eval-debug` | `eval_debug.txt` | 8 | Small mix of loglikelihood + generative tasks to smoke-test the eval pipeline |
| `custom` | `--task-file` | — | User-specified task file and optional `--table-metrics` file |
| `single` | `--task` | 1 | One task, user-specified through `--task` (comma-separated tasks allowed) |

**OLMo3 suites** (`configs/olmo/`)

| Mode | Tasks | Description |
|------|-------|-------------|
| `olmo-easy` | 17 tasks | Base Easy Suite: perplexity/BPB-style evaluation (mmlu, hellaswag, arc, etc.) |
| `olmo-main` | 21 tasks | Base Main Suite: generation + MC (gsm8k_cot, humaneval, drop, etc.) |
| `olmo-heldout` | 2 tasks | Held-out Suite: mmlu_pro, bbh |
| `olmo-safety` | 4 tasks | Safety Suite: harmbench, toxigen, wmdp, bbq |
| `olmo-longcontext` | 1 task | Long-Context: RULER (8192 tokens) |
| `olmo-complete` | 24 tasks | Curated combined suite (excludes long-context; see the task file for intentionally omitted tasks) |

Each mode maps to a task list and a metric config (`*_main_table.txt`) in the same directory. OLMo3 modes log to a per-mode W&B project (the base `WANDB_PROJECT` with a `-olmo-<suite>` suffix, e.g. `<project>-olmo-easy`); the `single` mode appends `-single`. The launcher defaults `WANDB_PROJECT` to `apertus-1.5-post-training-v0.0` (not `evaluate.sbatch`'s own `swissai-evals-test` default — see [SBATCH Scripts](#sbatch-scripts)); export `WANDB_PROJECT` yourself for test/smoke runs to avoid logging into the production project.

### Model Selection Modes

**Mode 1: Single model** (recommended for quick evaluations)
```bash
bash scripts/lm_eval_harness/launch_evaluations.sh <mode> --model <hf_path_or_local_path> [options]
```
Automatically derives the run name; the chat template is applied by default for every model (pass `--no-chat-template` to disable it).

The mode may be omitted for the default post-training suite:
```bash
bash scripts/lm_eval_harness/launch_evaluations.sh --model <hf_path_or_local_path> [options]
```

**Mode 2: Model-list script** (for batch evaluation of predefined model sets)
```bash
bash scripts/lm_eval_harness/launch_evaluations.sh <mode> --script scripts/lm_eval_harness/runners/example_models.sh
```
Runs a script that defines a `MODEL_CHECKPOINTS` associative array and sources `hf_base_runner.sh`.

### Options / Key Hyperparameters

| Flag | Description |
|------|-------------|
| `--name <name>` | Override the auto-derived evaluation run name |
| `--task <task>` | Task name(s) for `single` mode (single task or comma-separated list) |
| `--chat-template` | Force enable chat template (applied by default for every model) |
| `--no-chat-template` | Force disable chat template |
| `--tokenizer <path>` | Custom tokenizer (default: same as model) |
| `--num-fewshot N` | Override num_fewshot globally. Tasks with explicit `num_fewshot: 0` in their YAML are never overridden. OLMo3 paper uses 5-shot for most MC tasks. |
| `--backend <hf\|vllm\|sglang\|megatron_lm\|openai>` | Inference backend (default: from sbatch script). `vllm` recommended for in-job inference; `openai` evaluates against an already-running OpenAI-compatible endpoint instead of loading the model. |
| `--api-base-url <url>` | Endpoint for the `openai` backend (required with it). Accepts a bare host (`http://host:8000`), a `/v1` root, or a full `/v1/(chat/)completions` URL. |
| `--api-model-name <name>` | `model` field sent in API requests (default: the `--model` value) |
| `--api-requests-per-minute N` | Endpoint-wide request limit for the benchmarked model when using `--backend openai` (for example, `30`) |
| `--chunk-size N` | Maximum tasks per resumable Slurm-array element (default: 8) |
| `--max-parallel N` | Cap concurrently running chunks; omitted means all generated chunks may run |
| `--max-retries N` | Retry waves for missing tasks (default: 1); retry chunk size is halved |
| `--failure-policy <resume\|fail-fast>` | `resume` scans existing results, runs missing tasks, retries, and aggregates (default); `fail-fast` runs the full suite once |
| `--force-tasks <patterns>` | Comma-separated task substrings to evaluate again even when results exist |
| `--task-file <path>` / `--table-metrics <path>` | Task and main-metric files for `custom` mode |
| `--logs-root <path>` | Override the evaluation output root |
| `--wandb-entity <name>` / `--wandb-project <name>` | Override the W&B destination |
| `--account <name>` | Override the Slurm account used by submitted jobs |
| `--merge-only` | Skip evaluation and aggregate the already completed task outputs |
| `--debug` | Resolve and print the orchestration plan without submitting jobs |
| `--max-length N` | Backend context limit in tokens, exported as `MAX_LENGTH` (vLLM/SGLang `max_model_len`, Megatron `seq_length`, hf/openai `max_length`). Default: the backend's own. `loglikelihood_rolling` tasks such as `uncheatable_eval` cut each document into windows of this size, so set it to the model's context to score documents whole. |
| `--limit N` | Limit number of samples per task (forwarded as `--limit` to lm-eval-harness; default: no limit). Useful for quick sanity checks. |
| `--megatron-iter <iter>` | For Megatron-LM checkpoints, the iteration to evaluate (e.g. `8926`); defaults to `latest`. Exported as `CKPT_ITERATION`. |
| `--harness-branch B` | Resolve lm-evaluation-harness from a branch, tag, or full commit SHA in the task-selected repository (default: repository HEAD); the resolved commit is shared by every chunk |
| `--reservation <name>` | Submit evaluation jobs and any automatically launched judge under this SLURM reservation. |
| `--qos <name>` | Submit every job, including any automatically launched judge, under this SLURM QOS (e.g. `highprio`). Exported as `SBATCH_QOS`. |
| `--partition <name>` | Submit every job, including any automatically launched judge, to this SLURM partition (e.g. `highprio`; default: `normal`). Exported as `SBATCH_PARTITION`. |
| `--judge <none\|auto\|preset>` | Judge-model control for LLM-as-a-judge tasks. `none` (default) disables auto-launch; `auto` scans the task list using the mapping in `scripts/lm_eval_harness/launch_judge.py`; a preset name (e.g. `qwen3.5-27b`, `llama-3.3-70b`) launches that judge. |
| `--judge-args <str>` | Extra arguments forwarded to `scripts/lm_eval_harness/launch_judge.py` |
| `--judge-requests-per-minute N` | Endpoint-wide request limit applied separately to each hosted judge model (for example, `30`) |
| `--keep-judge` | Do not auto-cancel the judge model after evaluation finishes (otherwise a cleanup job cancels it via `afterany` dependency) |
| `--thinking` | Umbrella flag for reasoning models: make the model think **and** record the thinking metrics. See [Thinking / Reasoning Metrics](#thinking--reasoning-metrics). |
| `--enable-thinking` / `--no-enable-thinking` | Chat-template argument deciding whether the model reasons. `--enable-thinking` **on its own records nothing** — a reasoning close token must also be known. |
| `--think-end-token <str>` | Force the reasoning close token, e.g. `'</think>'`. A known close token arms the trace strip **and** the thinking metrics. |
| `--think-start-token <str>` | Force the reasoning open token, e.g. `'<think>'`. Needed for `thinking_format_has_open`. |
| `--autodetect-think-tokens` | Read the reasoning open/close tokens from the model's chat template. Only templates that declare them as `inner_token = '…'` / `outer_token = '…'` (Apertus-style) are recognised; for other models pass the tokens explicitly. |
| `--track-thinking-metrics <true\|false>` / `--no-track-thinking-metrics` | Force the thinking metrics on or off (default: on iff a close token is known). |
| `--log-length-metrics` | Aggregate `response_length_*` / `thinking_length_*` into results and W&B. `thinking_format_*` is aggregated regardless. |
| `--reasoning-effort <level>` | Chat-template argument `reasoning_effort` (e.g. `low`/`medium`/`high`) for models whose template reads it, such as gpt-oss. hf, vllm, sglang and openai backends. See [Reasoning effort](#reasoning-effort). |

> [!TIP]
> Inference hyperparameters such as batch size (`BS`) and `MAX_NEW_TOKENS` are not exposed as launcher flags (`MAX_LENGTH` is: `--max-length`) — set them as environment variables consumed by `evaluate.sbatch` (see [SBATCH Scripts](#sbatch-scripts)). `SIZE` is retained as informational/legacy metadata. vLLM keeps the compatible TP=4/DP=1 topology unless the model name unambiguously identifies a model below 30B, in which case it uses TP=1/DP=4; explicit topology variables always win.

### Judge Model Launching

Some LLM-as-a-judge tasks require a separate model to be available through the CSCS serving API. Pass `--judge auto` to launch the required judge models after the resume scan, and only when judge-dependent tasks are still missing:

```bash
bash scripts/lm_eval_harness/launch_evaluations.sh posttrain \
  --model /capstor/store/.../checkpoint \
  --judge auto \
  --judge-requests-per-minute 30
```

The task-to-judge mapping is defined by `TASK_TO_JUDGE` in `scripts/lm_eval_harness/launch_judge.py`:

| Tasks | Judge preset |
|-------|--------------|
| `alpaca_eval`, `multijail`, `aya_redteaming` | `llama-3.3-70b` |
| `arena_hard_v01`, `arena_hard_v2`, `hallulens` | `qwen3.5-27b` |
| `harmbench` | `cais-llama-harmbench` |
| `realtoxicitypromptsllama` | `llama-guard` |

Automatic judge launching runs on the login node, so the `python3` used to invoke the evaluation launcher must have the [Swiss AI Model Launch](https://github.com/swiss-ai/model-launch) (`swiss_ai_model_launch`) package installed. With a local `model-launch/` checkout:

```bash
python3 -m pip install -e ./model-launch
python3 -c "import swiss_ai_model_launch"
```

`CSCS_SERVING_API` must also be exported or stored in `scripts/cscs_serving_api_key.txt` so the launcher can verify that the judge is healthy.

CSCS exposes hosted models with a host scope, for example `<user>/cais/HarmBench-Llama-2-13b-cls`. The launcher defaults `JUDGE_MODEL_PREFIX` to `$USER`; the harness prefers `$JUDGE_MODEL_PREFIX/<model>` and checks `/v1/models` for a `CSCS-Inference/<model>` provider name or another unique matching scope. Unscoped hosted IDs are not accepted. Set `JUDGE_MODEL_PREFIX` explicitly when the desired host differs from the submitting user. Fully scoped task-specific judge-model overrides are preserved.

The judge limit is shared by all task threads and Slurm chunks in one launch and is keyed by endpoint/model, so two judge models configured at 30 RPM each do not unnecessarily share one 30 RPM budget. Set `LM_EVAL_RATE_LIMIT_STATE_DIR` to the same shared-filesystem directory for separate launcher invocations that must coordinate a common endpoint limit.

An explicit preset can be launched through the evaluation launcher or directly:

```bash
# Evaluation launcher; --reservation is forwarded to the judge job
bash scripts/lm_eval_harness/launch_evaluations.sh single \
  --task arena_hard_v2 \
  --model /capstor/store/.../checkpoint \
  --judge qwen3.5-27b \
  --reservation my-reservation

# Direct judge launch
python3 scripts/lm_eval_harness/launch_judge.py \
  --preset qwen3.5-27b \
  --reservation my-reservation
```

By default the evaluation launcher submits a cleanup job that cancels automatically launched judges after all evaluation jobs finish. Pass `--keep-judge` to leave them running. Additional judge-specific options, such as `--health-timeout 1800`, can be supplied through `--judge-args`.

#### Experimental OpenAI GPT judge path

The separate `gpt` suite reserves a stable task/config path for the planned
OpenAI GPT judge implementation in `swiss-ai/lm-evaluation-harness`:

```bash
export OPENAI_API_KEY="..."
bash scripts/lm_eval_harness/launch_evaluations.sh gpt \
  --model /capstor/store/.../checkpoint
```

The key may instead be stored in the ignored `scripts/openai_api_key.txt` file.
This is experimental scaffolding: the launcher deliberately does not pass a
speculative `--judge-type` (or similar) flag yet. Until the corresponding
Swiss-AI harness support lands, the mode prints a warning and is not expected
to provide the future GPT-judge behavior. The ordinary `posttrain` suite
continues to use the CSCS judge setup described above.


### Examples

```bash
# OLMo3 paper-faithful 5-shot evaluation
bash scripts/lm_eval_harness/launch_evaluations.sh olmo-complete --model allenai/OLMo-2-1124-7B --num-fewshot 5 --no-chat-template

# Large model with vLLM, eight tasks per chunk and four concurrent chunks
bash scripts/lm_eval_harness/launch_evaluations.sh default \
  --model Qwen/Qwen2.5-72B-Instruct --backend vllm \
  --chunk-size 8 --max-parallel 4

# Run all models from a batch script on the safety suite
bash scripts/lm_eval_harness/launch_evaluations.sh olmo-safety \
  --script scripts/lm_eval_harness/runners/example_models.sh --chunk-size 4
```

`--bos` explicitly prepends a BOS token. Normally leave it unset unless the model requires it; chat-template detection does not infer this option.

---

## Graceful / Resumable Launcher

Graceful execution is the default behavior of `scripts/lm_eval_harness/launch_evaluations.sh`. The launcher scans existing harness outputs, evaluates only missing tasks, and uploads one aggregate W&B run.

```bash
bash scripts/lm_eval_harness/launch_evaluations.sh custom \
  --task-file configs/apertus/tasks_posttrain_final.txt \
  --model /capstor/store/.../apertus-1.5-checkpoint \
  --table-metrics configs/apertus/tasks_posttrain_final_main_table.txt \
  --wandb-entity apertus --wandb-project apertus-1.5-post-training-v0.0 \
  --chunk-size 8 --max-parallel 4 --max-retries 1
```

### How it works

1. The task list and a normalized run configuration are recorded, then existing `eval_*` result directories are scanned. Only outputs with the same model/checkpoint, backend, harness ref, limit, few-shot, tokenizer/chat, and generation settings are eligible for resume; legacy or differently configured outputs are rerun.
2. Missing tasks are grouped into `--chunk-size N` chunks and submitted as one Slurm array. `--max-parallel` adds the array `%N` concurrency limit; by default every chunk may run.
3. A CPU controller runs with an `afterany` dependency, so it runs even when an array element fails.
4. The controller rescans results. If tasks remain and retry budget is available, it submits only those tasks again with the chunk size halved.
5. Completed outputs are merged and uploaded once. When retries are exhausted, successful tasks are still aggregated and `final_failed_tasks.txt` is retained in the run's controller state directory.

Use `--failure-policy fail-fast` to recover the original single-job behavior. `--force-tasks`, `--merge-only`, and `--debug` support targeted reruns, aggregation-only recovery, and submission dry runs.

---

## Task Configuration

Task lists are plain text files under `configs/` (principally `configs/apertus/` and `configs/olmo/`) with one task name per line. Comments (`#`) and blank lines are supported:

```
# Math
gsm8k_cot
minerva_math

# Code
humaneval
mbpp
```

The corresponding `*_main_table.txt` file specifies which `task/metric` pairs appear in the W&B summary table:

```
gsm8k_cot/exact_match,strict-match
mmlu/acc
arc_challenge/acc_norm
```

### Few-Shot Configuration

lm-eval-harness uses a three-level hierarchy for `num_fewshot`:

1. **Task YAML default** -- each task defines its own default (typically 0 for MC tasks)
2. **CLI `--num_fewshot N`** -- overrides the task default globally
3. **Explicit `num_fewshot: 0`** -- tasks like `coqa`, `lambada_openai` that explicitly set 0 are **never** overridden by the CLI flag

Use `--num-fewshot 5` to match the OLMo3 paper settings. Tasks with hardcoded examples (e.g., `gsm8k_cot` has 8 chain-of-thought examples baked into its prompt template) are unaffected.

### Adding Custom Task Suites

1. Create `configs/my_suite.txt` with task names (one per line)
2. Create `configs/my_suite_main_table.txt` with `task_name/metric_name` entries
3. Add a new case in the `launch_evaluations.sh` mode selector, or export `TASKS` and `TABLE_METRICS` directly:

```bash
export TASKS=./configs/my_suite.txt
export TABLE_METRICS=./configs/my_suite_main_table.txt
bash scripts/lm_eval_harness/launch_evaluations.sh olmo-complete --model my-model
```

Available task names can be found in the selected lm-evaluation-harness checkout under `lm_eval/tasks/` or by running `lm_eval --tasks list` in the prepared evaluation environment.

---

## Parallel Task Chunking

`--chunk-size N` controls failure isolation and the amount of cross-task lm-eval batching within each job. `--max-parallel N` independently limits resource usage. If `--max-parallel` is omitted, it defaults to the number of generated chunks.

### How It Works

1. The launcher computes chunks from the currently missing tasks.
2. A CPU preparation job resolves one harness commit and creates or reuses its environment.
3. The chunks run as a Slurm job array after environment preparation succeeds.
4. An `afterany` controller scans outputs, retries missing tasks, and submits aggregation.
5. The aggregation job combines `results_*.json` files and sample JSONL files, then uploads one W&B run.

```
prepare environment ──> chunk array ──> afterany controller ──> aggregate ──> W&B
                                      └── retry missing chunks, if needed
```

No manual dependency management is needed -- the launcher handles everything via `sbatch --parsable` and `--dependency`.

### Race Condition Safety

- Chunk jobs do **not** upload to W&B individually. Only the aggregation job does the upload, avoiding concurrent `wandb.init(resume="allow")` conflicts.
- Output directories are unique per job ID (`eval_<timestamp>_$SLURM_JOBID`), so file writes never collide.

---

## Thinking / Reasoning Metrics

For reasoning models, the harness strips the reasoning trace before scoring the answer, and records
how long that trace was and whether it was well-formed. Enable all of it with `--thinking`, plus
the model's reasoning tokens unless its chat template declares them (see the note below):

```bash
bash scripts/lm_eval_harness/launch_evaluations.sh single --task gsm8k_cot --model Qwen/Qwen3-8B \
  --thinking --think-start-token '<think>' --think-end-token '</think>'
```

> [!WARNING]
> **`--enable-thinking` on its own records nothing.** It is purely a chat-template argument that
> decides whether the model reasons. The trace strip and every thinking metric are armed by a known
> reasoning **close token** — supplied with `--think-end-token '</think>'` or discovered with
> `--autodetect-think-tokens`. `--thinking` turns on auto-detection unless you pass
> `--think-end-token`, and the launcher refuses a job that has neither.

> [!IMPORTANT]
> **Auto-detection only recognises templates that declare the tokens** as Jinja variables,
> `inner_token = '…'` (open) and `outer_token = '…'` (close), as Apertus templates do. Templates
> that write the tokens literally (e.g. Qwen3's `<think>` … `</think>`) or use another format
> (e.g. gpt-oss's channels) declare nothing, so detection finds no close token: the job still runs,
> but the trace is not stripped and no thinking metrics are recorded. The only sign is the job-log
> warning `no reasoning close token is known`. For such models always pass `--think-start-token`
> and `--think-end-token`. Detection fails loudly only when a template declares the tokens but the
> close can't be read as a literal.

### The four independent switches

| Question                                    | Flag                                             | Default                                                                                 |
|---------------------------------------------|--------------------------------------------------|-----------------------------------------------------------------------------------------|
| Does the model reason?                      | `--enable-thinking`                              | the chat template's own default (`enable_thinking` is not sent unless set)              |
| Are the reasoning tokens discovered?        | `--autodetect-think-tokens` (declared tokens only) | off — the template is never scanned                                                   |
| Does the trace get stripped before scoring? | *(implicit)* whenever a **close** token is known | off                                                                                     |
| Are the thinking metrics recorded?          | `--track-thinking-metrics`                       | on iff a close token is known                                                           |

`--thinking` sets the first, second and fourth, adds `--log-length-metrics`, and forces the chat
template on (the reasoning tokens live in it). Any granular flag you pass overrides the umbrella.

Thinking runs also change **generation**: sampling is forced (`do_sample=true`,
`THINK_TEMPERATURE=0.6`, `THINK_TOP_P=0.95` — reasoning degrades under greedy), and every task's
generation budget becomes 32768 tokens, replacing the task YAML's `max_gen_toks`: those were sized
for a direct answer (mmlu_pro 2048, ifeval 1280) and the trace counts against them. 32768 is the
largest task budget (AIME), so no task is lowered. Runs with `--reasoning-effort` get the same
budget. Override via the `THINK_*` / `MAX_NEW_TOKENS` env vars (see
[SBATCH Scripts](#sbatch-scripts)); `NOTHINK_TEMPERATURE` enables the same sampling for no-think
ablations.

### Reasoning effort

Some models take a reasoning *level* instead of (or on top of) the on/off switch. gpt-oss's chat
template reads `reasoning_effort` and writes `Reasoning: <level>` into its system block (default
`medium`). Pass it with `--reasoning-effort`:

```bash
bash scripts/lm_eval_harness/launch_evaluations.sh single --task aime25 --model openai/gpt-oss-120b \
  --thinking --reasoning-effort high \
  --think-start-token '<|channel|>analysis<|message|>' --think-end-token '<|channel|>final<|message|>'
```

It goes to the harness as `chat_template_args={"reasoning_effort":"<level>"}` (hf, vllm, sglang and
openai; the launcher refuses megatron_lm), forces the chat template on, is part of the run
configuration (results of another level are never resumed into this one), and adds
`-effort-<level>` to an auto-derived run name. A template that doesn't read `reasoning_effort`
ignores it silently -- check the rendered prompt in the job log. It does not turn reasoning on or
set the think tokens by itself, and `--thinking` can't auto-detect gpt-oss's: its template declares
no `inner_token`/`outer_token`, so without the explicit tokens above the whole `analysis` channel stays in the
scored response and the thinking metrics stay off (the job log warns `no reasoning close token is
known`). With them, only the `final` channel is scored.

### Emitted metrics

Recorded per task, and uploaded to W&B as `<task>/<metric>` alongside a `_stderr` companion:

| Metric                                 | Kind         | Gated by                                   |
|----------------------------------------|--------------|--------------------------------------------|
| `thinking_format_has_open`             | rate `[0,1]` | tracking on **and** an open token is known |
| `thinking_format_has_close`            | rate `[0,1]` | tracking on                                |
| `thinking_format_correct`              | rate `[0,1]` | tracking on                                |
| `response_length_{words,chars,tokens}` | raw count    | `--log-length-metrics`                     |
| `thinking_length_{words,chars,tokens}` | raw count    | `--log-length-metrics`                     |

> [!NOTE]
> These exist **only for generative tasks** (`generate_until` / `multi_turn_generate`).
> Multiple-choice and loglikelihood tasks — `mmlu`, `hellaswag`, `arc_challenge` — emit nothing,
> and are dropped automatically from the thinking table.

Two caveats worth internalising:

- **The two length families use different denominators.** `response_length_*` averages over *all*
  responses; `thinking_length_*` averages over *well-formed* responses only (those with
  `thinking_format_correct == 1`). So an aggregate `thinking_length` is **not** bounded by the
  aggregate `response_length` — it can exceed it when the well-formed responses are the long ones.
- **Prefer `_chars` when comparing across backends.** `_tokens` provenance differs per backend
  (vLLM counts the stop-string and EOS tokens; a re-encoded thinking span can land 1–2 tokens over).

### Backend support

| Backend              | Support         | `enable_thinking`                                                                |
|----------------------|-----------------|----------------------------------------------------------------------------------|
| `vllm` (recommended) | full            | forwarded **only when explicitly set**; otherwise the template's default applies |
| `hf`                 | full            | forwarded **only when explicitly set**; otherwise the template's default applies |
| `sglang`             | full            | forwarded **only when explicitly set**; uses its dedicated container             |
| `megatron_lm`        | **unsupported** | requesting thinking metrics is a hard error                                      |
| `openai`             | full (`local-completions`) | rendered client-side; forwarded **only when explicitly set**. `--enable-thinking` on its own only switches the template; `--thinking` or a think token also strips and measures the reasoning |

On `openai` the harness renders the chat template itself before sending the prompt to
`/v1/completions`, so the template switches reach the model. With `--thinking` or a think token it
also holds the task's stop sequences back while the model reasons (they often occur inside the
reasoning), strips the reasoning before scoring and records the thinking metrics, as on vllm.
Reasoning markers made of special tokens (gpt-oss's `<|channel|>…<|message|>`) are kept in the
server's text for that (`skip_special_tokens: false`). `--enable-thinking` on its own only switches
the template: the reasoning is then scored with the answer, so prefer `--thinking`. A reasoning
effort raises the default token budget like `--enable-thinking` does (gpt-oss reasons at every
level). With `--api-endpoint chat-completions` the server renders the template: these settings go
with each request as `chat_template_kwargs`, and the server returns the reasoning in its own field,
which the harness puts back between `--think-start-token`/`--think-end-token` (when given) so it is
stripped and measured the same way.

> [!NOTE]
> Templates such as Qwen3's default to reasoning when `enable_thinking` is not passed. For a
> non-thinking baseline of such a model, pass `--no-enable-thinking` explicitly.

### Examples

```bash
# Reasoning model, everything on, quick smoke test
bash scripts/lm_eval_harness/launch_evaluations.sh single --task gsm8k_cot --model Qwen/Qwen3-8B --limit 20 \
  --thinking --think-start-token '<think>' --think-end-token '</think>'

# A model whose template declares inner_token/outer_token (Apertus-style): --thinking alone
bash scripts/lm_eval_harness/launch_evaluations.sh single --task gsm8k_cot --model my/apertus-reasoner --thinking

# Explicit tokens rather than template auto-detection
bash scripts/lm_eval_harness/launch_evaluations.sh posttrain --model my/reasoner --backend vllm \
  --enable-thinking --think-start-token '<think>' --think-end-token '</think>' \
  --log-length-metrics

# Measure format well-formedness only, no length aggregation
bash scripts/lm_eval_harness/launch_evaluations.sh olmo-main --model my/reasoner --autodetect-think-tokens

# Response length of a NON-reasoning model: no close token needed, since response_length_*
# is recorded for every generative response whether the model thinks or not
bash scripts/lm_eval_harness/launch_evaluations.sh posttrain --model meta-llama/Llama-3.1-8B-Instruct --log-length-metrics
```

The main launcher forwards all of these flags to every chunk:

```bash
bash scripts/lm_eval_harness/launch_evaluations.sh posttrain \
  --model /capstor/.../my-reasoner --thinking
```

Thinking runs get an isolated run name (`<model-basename>-think`, override with `--name`), so their
results and W&B run never collide with the same model's non-thinking eval — see
[Graceful / Resumable Launcher](#graceful--resumable-launcher).

### Building a thinking-only table

`scripts/results/make_html_table.py --thinking` creates a separate, thinking-only table: one group per metric family, one row per task. It reuses
the suite's **existing task list** as the metric source.

```bash
python scripts/results/make_html_table.py --thinking \
  --metrics-file configs/apertus/tasks_posttrain_final.txt \
  --entity apertus --project apertus-1.5-post-training-v0.0 \
  --models my-reasoner-run another-run \
  --output thinking_table.html
```

```
Category / Benchmark          Reasoner-A   Reasoner-B
▼ Thinking Format Correct (%)       97.4         99.1
    gsm8k_cot                       98.2         99.5
    aime24                          96.6         98.7
▼ Thinking Length (tokens)           413          918
    gsm8k_cot                        287          602
    aime24                           538         1235
▼ Response Length (tokens)           499         1021
```

Behaviour specific to `--thinking`:

- Lengths render **raw** and never receive a "best" badge — a shorter trace is not automatically a
  better one. Only the format-correctness rates are scored and highlighted.
- There is **no Overall Average**: averaging a `[0,1]` rate with a ~600-token count is meaningless.
  Each group still shows a macro-mean over its tasks.
- Tasks with no thinking metrics, and models that never ran with thinking enabled, are dropped.
- `--length-unit {tokens,words,chars}` swaps the unit (default `tokens`);
  `--thinking-format-detail` adds the `has_open` / `has_close` groups.
- Metric keys are resolved by **exact match** (`get_metric(..., fuzzy=False)`), since they are
  synthesized rather than hand-written. The main table's substring fallback would happily let
  `gsm8k/...` be answered by `gsm8k_cot/...`.

---

## SBATCH Scripts

### `scripts/lm_eval_harness/evaluate.sbatch`

Primary SLURM job script for HuggingFace-compatible model evaluation.

**Resources**: local-model backends use `evaluate.sbatch` with 1 node, 4 GPUs, 200 CPUs, 460GB memory, and an 11h59m limit. The `openai` backend is selected automatically through the CPU-only `evaluate_api.sbatch` wrapper with 16 CPUs and 64GB memory.

**Positional arguments**: `<model_path> <name>`

**Environment variables** (all optional, with defaults):

| Variable | Default | Description |
|----------|---------|-------------|
| `TASKS` | `configs/apertus/tasks_default.txt` | Task list file or comma-separated task names |
| `TABLE_METRICS` | `configs/apertus/tasks_default_main_table.txt` | Metrics for W&B summary table |
| `LM_EVAL_BACKEND` | `vllm` | Backend: `hf` (accelerate), `vllm`, `sglang`, `megatron_lm`, `openai` (OpenAI-compatible API) |
| `API_BASE_URL` | (unset) | `openai` backend only, **required**: the OpenAI-compatible endpoint. Bare host / `/v1` root / full endpoint URL all accepted. |
| `API_MODEL_NAME` | same as model | `openai` backend only: the `model` field sent in requests, when the server registers the model under a different id |
| `API_NUM_CONCURRENT` | `8` | `openai` backend only: concurrent requests (batch size is pinned to 1; this is the throughput knob) |
| `API_REQUESTS_PER_MINUTE` | (unset) | `openai` backend only: endpoint-wide request limit; set by `--api-requests-per-minute` |
| `API_MAX_RETRIES` | `3` | `openai` backend only: retries per failed request |
| `JUDGE_REQUESTS_PER_MINUTE` | (unset) | Endpoint-wide request limit applied separately to every LLM judge model; set by `--judge-requests-per-minute` |
| `JUDGE_MODEL_PREFIX` | `$USER` | Preferred CSCS host scope prepended to unscoped judge model IDs |
| `LM_EVAL_JUDGE_MODEL_DISCOVERY` | `1` | Query the judge endpoint's `/models` list and fall back to another matching hosted scope; set to `0` to disable |
| `LM_EVAL_JUDGE_MODEL_DISCOVERY_TIMEOUT` | `10` | Timeout in seconds for hosted judge-model discovery |
| `LM_EVAL_RATE_LIMIT_STATE_DIR` | per-launch controller directory | Shared request-slot state used to coordinate threads, processes, and Slurm nodes; point independent launches at the same directory to share budgets |
| `OPENAI_API_KEY` | `scripts/openai_api_key.txt`, then `CSCS_SERVING_API` | OpenAI GPT judge or `openai` backend bearer token |
| `LM_EVAL_HARNESS_BRANCH` | repository HEAD | Branch, tag, or commit installed from the task-selected harness repository (not allowed with `EVAL_PREBUILT_ENV=1`, which uses `requirements/lm-eval-harness.txt`) |
| `APPLY_CHAT_TEMPLATE` | `true` | Apply chat template for instruct models |
| `TOKENIZER` | same as model | Custom tokenizer path |
| `BOS` | `false` | Prepend BOS token |
| `BS` | `auto:20` | Batch size |
| `SIZE` | `1` | Informational/legacy model-size metadata; topology inference uses the model name/path instead |
| `MAX_LENGTH` | (backend/model default) | Optional total context limit, passed as the backend-native setting (`max_model_len` for vLLM/SGLang, `seq_length` for Megatron, `max_length` for HF/API) |
| `MAX_NEW_TOKENS` | `32768` with thinking or a reasoning effort; otherwise unused | Generation budget of thinking/reasoning-effort runs, passed as `--gen_kwargs max_gen_toks=...` so it replaces every task's own `max_gen_toks`. An explicit value sets that budget. |
| `VLLM_TP_SIZE` / `VLLM_DP_SIZE` | auto | Explicit vLLM topology; either value can be omitted and is derived from the four allocated GPUs. Both values must multiply to 4. |
| `VLLM_AUTO_PARALLELISM` | `true` | With no explicit topology, use TP=1/DP=4 only when the model name unambiguously identifies `<30B`; otherwise retain TP=4/DP=1. |
| `VLLM_MEMORY_FRACTION` | `0.8` | vLLM `gpu_memory_utilization` / SGLang `mem_fraction_static` |
| `MEGATRON_MICRO_BATCH_SIZE` | numeric `BS`, otherwise `1` | Megatron micro-batch override. A numeric `BS` is forwarded automatically; `BS=auto:*` cannot be inferred by the adapter and falls back to 1. |
| `THINK_TEMPERATURE` / `THINK_TOP_P` | `0.6` / `0.95` | Sampling for thinking runs (`do_sample=true` is forced — reasoning degrades under greedy) |
| `THINK_REPETITION_PENALTY` | (unset) | Optional knob against degenerate looping in thinking runs (e.g. `1.05`) |
| `NOTHINK_TEMPERATURE` / `NOTHINK_TOP_P` | (unset) / `0.95` | Optional sampling for NO-think runs (greedy-vs-sampling ablations); inert unless `NOTHINK_TEMPERATURE` is set |
| `HARNESS_LIMIT` | (unset) | Limit number of samples per task (set by launcher flag `--limit`) |
| `NUM_FEWSHOT` | (unset) | Global few-shot override |
| `EVAL_ENV_MANIFEST` | required | Immutable base-environment and harness-overlay paths produced by `prepare_eval_env.sbatch` |
| `EVAL_PREBUILT_ENV` | (unset) | `1` in the prebuilt image: no environment preparation or installs; the launcher writes the manifest (see [Prebuilt evaluation image](#prebuilt-evaluation-image-no-installs-at-run-time)) |
| `EVAL_RUN_CONFIG` | required | Launcher-generated normalized provenance used to exclude incompatible results during resume |
| `EVAL_CHUNKS_FILE` | (unset) | One comma-separated task chunk per line; indexed by `SLURM_ARRAY_TASK_ID` |
| `LOGS_ROOT` | `$SCRATCH/eval_logs_start/` | Root directory for evaluation logs |
| `WANDB_ENTITY` | `apertus` | W&B entity |
| `WANDB_PROJECT` | `swissai-evals-test` | W&B project. This is `evaluate.sbatch`'s own default when invoked directly; `launch_evaluations.sh` sets its own default of `apertus-1.5-post-training-v0.0` before the sbatch script ever runs (see [The Launch Script](#the-launch-script)), so export `WANDB_PROJECT` explicitly to keep test/smoke runs out of the production project. |
| `ENABLE_THINKING` | *(unset)* | Chat-template argument: whether the model reasons. Emitted for `hf`, `vllm` and `sglang` **only when set explicitly**; unset leaves the template's default. |
| `AUTODETECT_THINK_TOKENS` | `false` | Read the reasoning open/close tokens from the chat template |
| `THINK_START_TOKEN` | (unset) | Force the reasoning open token, e.g. `<think>` |
| `THINK_END_TOKEN` | (unset) | Force the reasoning close token, e.g. `</think>`. Arms the strip and the metrics. |
| `TRACK_THINKING_METRICS` | (unset → derive) | `true`/`false` to force the thinking metrics on/off |
| `LOG_LENGTH_METRICS` | `false` | Add `--log_length_metrics` (aggregates `response_length_*` / `thinking_length_*`) |

The script auto-detects RULER long-context tasks, adds 600 tokens of engine headroom, and passes the resulting limit using the selected backend's native argument.

Requesting any thinking metric with `LM_EVAL_BACKEND=megatron_lm` aborts the job: the harness has no
reasoning-token support for that backend, so it would otherwise record nothing silently. See
[Thinking / Reasoning Metrics](#thinking--reasoning-metrics).

---

## Multi-Model Scripts

A model-list script defines a `MODEL_CHECKPOINTS` associative array and sources `hf_base_runner.sh`; copy `scripts/lm_eval_harness/runners/example_models.sh` and list your models:

```bash
declare -A MODEL_CHECKPOINTS=(
    ["Llama-3.1-8B-Instruct"]="meta-llama/Llama-3.1-8B-Instruct"
    ["OLMo-2-1124-7B-Instruct"]="allenai/OLMo-2-1124-7B-Instruct"
)
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-true}
source scripts/lm_eval_harness/runners/hf_base_runner.sh "models"
```

Base models take the launcher's `--no-chat-template` (Apertus base checkpoints also `--tokenizer alehc/swissai-tokenizer --bos`).

`hf_base_runner.sh` handles the model loop and delegates resumable chunk orchestration to `scripts/lm_eval_harness/evaluation_orchestrator.sh`. It respects `EVAL_CHUNK_SIZE`, `EVAL_MAX_PARALLEL`, `EVAL_MAX_RETRIES`, `SBATCH_SCRIPT`, and `WANDB_*`.

### Model Registry

See `configs/models.md` for a reference list of models with their HF paths, local checkpoint paths, and required special flags. Key model families:

- **Apertus** (1.0, 1.5, and aligned checkpoints)
- **Meta Llama** (3.1, 3.3)
- **OLMo** (2-1124, 2-0325, 3)
- **Qwen** (2.5, 3)
- **Gemma** (3), **EuroLLM**, **Mistral**, **SmolLM**, **Marin**, and others

---

## Container Setup

The pipeline runs inside containers managed by enroot/pyxis on SLURM. The available container configurations:

| Config | Base Image | Use Case |
|--------|-----------|----------|
| `env.toml` | CSCS NGC PyTorch 25.12 image | Standard HF evals |
| `env_vllm.toml` | CSCS vLLM 0.22.1 runtime image | vLLM and Megatron-LM evals |
| `env_sglang.toml` | CSCS SGLang CUDA 13 image | SGLang evals |

Evaluation dependencies use a shared two-tier cache under `${EVAL_ENV_CACHE_ROOT:-$SCRATCH/eval-envs}`:

1. A base virtual environment is keyed by backend, Python/container configuration, and `requirements/eval-runtime.txt`.
2. A lightweight harness overlay is keyed by the base environment, harness repository, and resolved commit SHA.

`prepare_eval_env.sbatch` first verifies that the cache and per-run manifest
directories are visible from both the host job and the selected container. It
then builds missing tiers on a CPU node. Each base and overlay fingerprint has
its own `flock`: simultaneous launches with the same fingerprint wait for the
first builder, re-check the completed environment, and reuse it; unrelated
fingerprints can build concurrently. Packages are published only after import
validation, and manifests are replaced atomically. The builder also creates one
uncompressed tar archive per immutable harness overlay. At evaluation startup,
each chunk reads that single archive from the shared filesystem and extracts it
under `$SLURM_TMPDIR` (or container-local `/tmp`), then prepends the local copy
to `PYTHONPATH`. If staging fails, the job logs a warning and safely falls back
to the shared overlay. This avoids thousands of small-file operations against
`/iopsstor`, including Python imports and the harness's task discovery fallback.

Recent harness commits also ship a persistent task index keyed implicitly by
the resolved commit. `TaskManager` reads that one JSON file for bundled tasks
and lazily opens only selected task YAMLs; explicit `--include_path`
directories remain dynamically scanned. Missing or invalid indexes retain the
old full-scan behavior, which remains reasonably fast because the overlay has
already been staged locally. A moving branch is therefore resolved once per
model run: pushed changes propagate on the next launch, while all chunks and
repairs in one run use exactly the same commit.

The cache is disposable. If `$SCRATCH` retention removes either tier, the next
launch rebuilds it automatically. Cache hits validate the environment rather
than trusting the completion marker alone. Evaluation jobs use a fast
in-container structural preflight (completion markers, Python executable, and
harness package) and perform a last-resort rebuild if retention removes those
paths between the CPU preparation job and GPU-job startup. They do not repeat
the expensive Python import validation by default; set
`EVAL_VERIFY_ENV_IMPORTS=true` to enable it for diagnosis. Set
`EVAL_ENV_CACHE_ROOT` to a longer-lived shared filesystem if retaining
environments beyond the cluster's scratch window is preferable. Successful
cache use refreshes the completion-marker and archive timestamps so recently
used entries remain active under age-based scratch retention policies.

### Prebuilt evaluation image (no installs at run time)

`containers/Dockerfile.eval` bakes everything an OpenAI-compatible-backend run
needs into one image: CPU PyTorch, `requirements/eval-runtime.txt`,
`requirements/inspect-runtime.txt`, and each lm-evaluation-harness fork at the
commit pinned in `requirements/lm-eval-harness.txt` (as `--no-deps` overlays
under `/opt/lm-eval-harness/<owner>/<repo>`, the same layout `build_eval_env.sh`
produces). CI (`.github/workflows/eval-image.yml`) publishes it for every
pushed commit as `ghcr.io/swiss-ai/evals-post-train-eval:<commit>`.

The image sets `EVAL_PREBUILT_ENV=1`. With it, the launcher submits no
`prepare_eval_env.sbatch` job and writes the manifest itself;
`evaluate.sbatch` checks that the container's harness commit matches the pin
in the checkout and that the packages import, then runs, never installing
anything. `run_inspect_eval.sh` skips its install the same way. A mismatch
(the scripts come from a different commit than the image) fails the job with
a message naming both commits, so run the image with the scripts from the
commit it was built from. evals-svc's k8s Jobs do exactly that, and fall back
to the environment-building path when no image exists for a commit yet.

The harness is pinned, not the moving `HEAD` the building path follows: to
ship a newer harness, bump its commit in `requirements/lm-eval-harness.txt`.
`--harness-branch`/`LM_EVAL_HARNESS_BRANCH` is refused with
`EVAL_PREBUILT_ENV=1`.

```bash
docker build -f containers/Dockerfile.eval -t evals-post-train-eval .
```

---

## Notes

> [!NOTE]
> **vLLM vs HF inference**: Generation task results (gsm8k, squadv2) may differ slightly between backends (for instruction-tuned models). Only compare results across models using the same backend. We recommend performing all evaluations with the `vllm` backend (default) to ensure reproducibility.
- **OpenAI-compatible API backend (`--backend openai`)**: evaluates against an already-running endpoint (e.g. `vllm serve`, the CSCS serving platform) instead of loading the model inside the job. It uses lm-eval's `local-completions` against `/v1/completions`, which serves **both** generative and loglikelihood/MC tasks (mixed suites work) *provided* the server returns prompt logprobs with echo (vLLM does; most commercial APIs do not). With the chat template on, the harness renders the model's template client-side via the HF tokenizer — so the tokenizer must be resolvable (use `--tokenizer` when the served model name is not a pullable HF repo). `--api-endpoint chat-completions` (or `API_CHAT_ENDPOINT=true`) switches to `/v1/chat/completions` (server-side template; generative tasks ONLY, so not with `--no-chat-template`/`--chat-template-by-type`; required by FC tasks such as `bfcl_v3_fc`, which send the functions in the request's `tools` field). The run records it as `api_chat_endpoint`. Auth uses `OPENAI_API_KEY` (defaults to the CSCS serving key). Use `--api-requests-per-minute 30` when the endpoint is rate limited; all chunks in the launch coordinate that budget. The launcher automatically submits this backend through the CPU-only `evaluate_api.sbatch` wrapper.
- **Megatron-LM**: To run Megatron-LM models natively, clone the [NVIDIA Megatron-LM repository](https://github.com/NVIDIA/Megatron-LM) into the evals-post-train directory (or change the location via the launch script).
- **Time limits**: The default 11h59m SLURM limit applies to each chunk. Adjust `--chunk-size` to keep individual jobs below it and `--max-parallel` to control concurrent nodes.

---

## Extending the Pipeline

### Adding a New Inference Backend

The sbatch scripts support `hf`, `vllm`, `sglang`, `megatron_lm`, and `openai` backends. To add a new one:

1. Add a new `elif` block in `evaluate.sbatch` at the `LM_EVAL_BACKEND` dispatch section
2. Set appropriate `COMMON_MODEL_ARGS` for the new backend
3. Add backend-independent Python dependencies to `requirements/eval-runtime.txt`; provide backend runtimes through the corresponding container configuration
4. Include the new backend/container identity in `scripts/lm_eval_harness/build_eval_env.sh` fingerprinting if it needs distinct cached environments

### Adding a New Task

If the task exists in lm-eval-harness:
1. Add the task name to your task list config file
2. Add the `task_name/metric_name` entry to the corresponding `*_main_table.txt`

If you need a custom task:
1. Create a YAML task config in `lm_eval/tasks/your_task/` in the selected harness repository
2. Register it following the [lm-eval-harness task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)

If lm-eval-harness (YAML config) isn't a fit -- e.g. the benchmark needs its own scorer/grading
model, like an LLM-as-judge, rather than a built-in metric -- write it as an Inspect task instead:
drop a `@task`-decorated Python file in `custom_tasks/` (see `scripts/aaii/aa_omniscience.py` and
`scripts/aaii/aa_lcr.py` for from-scratch examples with a judge-model scorer -- Artificial Analysis's own
AAII benchmarks live in `scripts/aaii/` specifically, not `custom_tasks/`, since they're a fixed,
AAII-branded protocol rather than an arbitrary one-off task) and run it with
`scripts/inspect_ai/run_inspect_eval.sh --task custom_tasks/your_task.py ...` -- see
[Alternative: Inspect AI evals](inspect-ai.md#inspect-ai-evals) for the full flag reference and
runnable examples.
