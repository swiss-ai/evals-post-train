# SwissAI Evaluation Pipeline

Evaluation infrastructure for benchmarking Large Language Models on SLURM clusters (CSCS Alps). Built on top of [lm-evaluation-harness](https://github.com/swiss-ai/lm-evaluation-harness) with W&B integration for results tracking.

**Contents**

1. [Quick Start](#quick-start)
2. [The Launch Script](#the-launch-script)
3. [Graceful / Resumable Launcher](#graceful--resumable-launcher)
4. [Task Configuration](#task-configuration)
5. [Parallel Task Chunking](#parallel-task-chunking)
6. [Thinking / Reasoning Metrics](#thinking--reasoning-metrics)
7. [Every Eval Ever and Hugging Face exports](#every-eval-ever-and-hugging-face-exports)
8. [W&B Integration](#wb-integration)
9. [Reporting: Building Result Tables](#reporting-building-result-tables)
10. [SBATCH Scripts](#sbatch-scripts)
11. [Multi-Model Scripts](#multi-model-scripts)
12. [Container Setup](#container-setup)
13. [Notes](#notes)
14. [Extending the Pipeline](#extending-the-pipeline)
15. [Repository Structure](#repository-structure)

---

## Quick Start

This pipeline **launches evaluations** on the cluster (see [The Launch Script](#the-launch-script)) and turns their W&B results into **tables** (see [Reporting: Building Result Tables](#reporting-building-result-tables)). For reasoning models, the `--thinking` option additionally makes the model reason and records the reasoning metrics — see [Thinking / Reasoning Metrics](#thinking--reasoning-metrics).

```bash
# Evaluate a single model on the benchmark suite (with custom name)
bash scripts/launch_evaluations.sh default --model meta-llama/Llama-3.1-8B-Instruct --name Llama-Baseline

# Resumable chunks of 8 tasks, with at most 4 chunks running concurrently
bash scripts/launch_evaluations.sh default --model meta-llama/Llama-3.1-8B-Instruct \
  --chunk-size 8 --max-parallel 4

# Launch Megatron checkpoint without conversion (TODO: Verify), Megatron-iter defaults to: latest
bash scripts/launch_evaluations.sh olmo-easy --model /capstor/store/../apertus-.../checkpoints/ --backend megatron_lm --name Megatron-Test-260216 --megatron-iter 4000000

# Launch with vllm backend - recommended!
bash scripts/launch_evaluations.sh olmo-easy --model /capstor/store/../apertus-.../checkpoints/ --backend vllm

# Launch with the SGLang backend
bash scripts/launch_evaluations.sh olmo-easy --model /capstor/store/../apertus-.../checkpoints/ --backend sglang

# Evaluate against an already-running OpenAI-compatible endpoint (vLLM serve, CSCS serving, ...)
# instead of loading the model in the job
bash scripts/launch_evaluations.sh single --task gsm8k_cot --model Qwen/Qwen3-8B \
  --backend openai --api-base-url http://nid001234:8000

# Evaluate a base model with 5-shot and easy eval set (matching OLMo3 technical report settings)
bash scripts/launch_evaluations.sh olmo-easy --model Qwen/Qwen2.5-7B --num-fewshot 5

# Evaluate a small model on a single task, useful for testing newly implemented tasks
bash scripts/launch_evaluations.sh single --task multijail --model meta-llama/Llama-3.2-3B --backend vllm

# Thinking / reasoning eval with an isolated W&B run name
bash scripts/launch_evaluations.sh posttrain --model Qwen/Qwen3-8B \
  --thinking --name Qwen3-8B-think
# ...then build the thinking-only table from that run (details: "Building a thinking-only table")
python make_html_table.py --thinking --metrics-file configs/apertus/tasks_posttrain_final.txt --entity apertus --project <project> --models Qwen3-8B-think --output thinking_table.html
```

---

## The Launch Script

`scripts/launch_evaluations.sh` is the primary entry point for running evaluations. It supports three model selection modes and multiple benchmark suites.

### Benchmark Suites

The launcher selects a benchmark suite from its first positional argument (`<mode>`). Apertus suites are defined in `configs/apertus/`, OLMo3 suites in `configs/olmo/`.

**Apertus suites** (`configs/apertus/`)

| Mode | Task list | Tasks | Description |
|------|-----------|-------|-------------|
| `default` | `tasks_default.txt` | 48 | Full Apertus 1.5 post-training suite: knowledge, math, code, reasoning, multilingual, instruction following, bias, cultural knowledge, safety |
| `posttrain` | `tasks_posttrain_final.txt` | 50 | Post-training suite incl. chat/arena (alpaca_eval, arena_hard_v01/v2), AIME, MATH-500, BFCL v3, Swiss AI Charter Alignment, o(r)bench, and system-prompt adherence (RealGuardrails: S-IFEval, TensorTrust, S-RuLES) |
| `best-of-k` | `tasks_best_of_k.txt` | 8 | Multi-repeat/self-consistency suite for math and code, with mean@k, majority-vote, and pass@k metrics |
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
| `olmo-easy` | 21 tasks | Base Easy Suite: perplexity/BPB-style evaluation (mmlu, hellaswag, arc, etc.) |
| `olmo-main` | 23 tasks | Base Main Suite: generation + MC (gsm8k_cot, humaneval, drop, etc.) |
| `olmo-heldout` | 2 tasks | Held-out Suite: mmlu_pro, bbh |
| `olmo-safety` | 4 tasks | Safety Suite: harmbench, toxigen, wmdp, bbq |
| `olmo-longcontext` | 1 task | Long-Context: RULER (8192 tokens) |
| `olmo-complete` | 30 tasks | Union of all above (excludes long-context), deduplicated |

Each mode maps to a task list and a metric config (`*_main_table.txt`) in the same directory. OLMo3 modes log to a per-mode W&B project (the base `WANDB_PROJECT` with a `-olmo-<suite>` suffix, e.g. `swissai-evals-test-olmo-easy`); the `single` mode appends `-single`.

### Model Selection Modes

**Mode 1: Single model** (recommended for quick evaluations)
```bash
bash scripts/launch_evaluations.sh <mode> --model <hf_path_or_local_path> [options]
```
Automatically derives the run name and detects whether to apply a chat template based on the model name (patterns: `-Instruct`, `-Chat`, `-SFT`, `-DPO`, `-it`, `-aligned`).

**Mode 2: Model-list script** (for batch evaluation of predefined model sets)
```bash
bash scripts/launch_evaluations.sh <mode> --script runners/hf_eval_multiple_other_models.sh
```
Runs a script that defines a `MODEL_CHECKPOINTS` associative array and sources `hf_base_runner.sh`.

**Mode 3: Default scripts** (edit the `EVALUATION_SCRIPTS` array inside the launcher)
```bash
bash scripts/launch_evaluations.sh <mode>
```

### Options / Key Hyperparameters

| Flag | Description |
|------|-------------|
| `--name <name>` | Override the auto-derived evaluation run name |
| `--task <task>` | Task name(s) for `single` mode (single task or comma-separated list) |
| `--chat-template` | Force enable chat template (auto-detected for Instruct/Chat/SFT/DPO/-it/-aligned models) |
| `--no-chat-template` | Force disable chat template |
| `--tokenizer <path>` | Custom tokenizer (default: same as model) |
| `--num-fewshot N` | Override num_fewshot globally. Tasks with explicit `num_fewshot: 0` in their YAML are never overridden. OLMo3 paper uses 5-shot for most MC tasks. |
| `--backend <hf\|vllm\|sglang\|megatron_lm\|openai>` | Inference backend (default: from sbatch script). `vllm` recommended for in-job inference; `openai` evaluates against an already-running OpenAI-compatible endpoint instead of loading the model. |
| `--api-base-url <url>` | Endpoint for the `openai` backend (required with it). Accepts a bare host (`http://host:8000`), a `/v1` root, or a full `/v1/(chat/)completions` URL. |
| `--api-model-name <name>` | `model` field sent in API requests (default: the `--model` value) |
| `--chunk-size N` | Maximum tasks per resumable Slurm-array element (default: 8) |
| `--max-parallel N` | Cap concurrently running chunks; omitted means all generated chunks may run |
| `--max-retries N` | Retry waves for missing tasks (default: 1); retry chunk size is halved |
| `--failure-policy <resume\|fail-fast>` | `resume` scans existing results, runs missing tasks, retries, and aggregates (default); `fail-fast` runs the full suite once |
| `--force-tasks <patterns>` | Comma-separated task substrings to evaluate again even when results exist |
| `--limit N` | Limit number of samples per task (forwarded as `--limit` to lm-eval-harness; default: no limit). Useful for quick sanity checks. |
| `--megatron-iter <iter>` | For Megatron-LM checkpoints, the iteration to evaluate (e.g. `8926`); defaults to `latest`. Exported as `CKPT_ITERATION`. |
| `--harness-branch B` | Resolve lm-evaluation-harness from a branch, tag, or full commit SHA (default: repository HEAD); the resolved commit is shared by every chunk |
| `--reservation <name>` | Submit evaluation jobs and any automatically launched judge under this SLURM reservation. |
| `--judge <none\|auto\|preset>` | Judge-model control for LLM-as-a-judge tasks. `none` (default) disables auto-launch; `auto` scans the task list using the mapping in `scripts/launch_judge.py`; a preset name (e.g. `qwen3.5-27b`, `llama-3.3-70b`) launches that judge. |
| `--judge-args <str>` | Extra arguments forwarded to `scripts/launch_judge.py` |
| `--keep-judge` | Do not auto-cancel the judge model after evaluation finishes (otherwise a cleanup job cancels it via `afterany` dependency) |
| `--thinking` | Umbrella flag for reasoning models: make the model think **and** record the thinking metrics. See [Thinking / Reasoning Metrics](#thinking--reasoning-metrics). |
| `--enable-thinking` / `--no-enable-thinking` | Chat-template argument deciding whether the model reasons. `--enable-thinking` **on its own records nothing** — a reasoning close token must also be known. |
| `--think-end-token <str>` | Force the reasoning close token, e.g. `'</think>'`. A known close token arms the trace strip **and** the thinking metrics. |
| `--think-start-token <str>` | Force the reasoning open token, e.g. `'<think>'`. Needed for `thinking_format_has_open`. |
| `--autodetect-think-tokens` | Read the reasoning open/close tokens from the model's chat template. |
| `--track-thinking-metrics <true\|false>` / `--no-track-thinking-metrics` | Force the thinking metrics on or off (default: on iff a close token is known). |
| `--log-length-metrics` | Aggregate `response_length_*` / `thinking_length_*` into results and W&B. `thinking_format_*` is aggregated regardless. |

> [!TIP]
> Inference hyperparameters such as batch size (`BS`), `MAX_LENGTH`, `MAX_NEW_TOKENS`, and `SIZE` (model size in billions, for parallelism) are not exposed as launcher flags — set them as environment variables consumed by `evaluate.sbatch` (see [SBATCH Scripts](#sbatch-scripts)).

### Judge Model Launching

Some LLM-as-a-judge tasks require a separate model to be available through the CSCS serving API. Pass `--judge auto` to scan the selected task list and launch the required judge models before submitting the evaluations:

```bash
bash scripts/launch_evaluations.sh posttrain \
  --model /capstor/store/.../checkpoint \
  --judge auto
```

The task-to-judge mapping is defined by `TASK_TO_JUDGE` in `scripts/launch_judge.py`:

| Tasks | Judge preset |
|-------|--------------|
| `alpaca_eval`, `multijail`, `aya_redteaming` | `llama-3.3-70b` |
| `arena_hard_v01`, `arena_hard_v2`, `hallulens` | `qwen3.5-27b` |
| `harmbench` | `cais-llama-harmbench` |
| `realtoxicitypromptsllama` | `llama-guard` |

Automatic judge launching runs on the login node, so the `python3` used to invoke the evaluation launcher must have the [Swiss AI Model Launch](https://github.com/swiss-ai/model-launch) (`swiss_ai_model_launch`) package installed. With the `model-launch/` checkout included in this repository:

```bash
python3 -m pip install -e ./model-launch
python3 -c "import swiss_ai_model_launch"
```

`CSCS_SERVING_API` must also be exported or stored in `scripts/cscs_serving_api_key.txt` so the launcher can verify that the judge is healthy.

An explicit preset can be launched through the evaluation launcher or directly:

```bash
# Evaluation launcher; --reservation is forwarded to the judge job
bash scripts/launch_evaluations.sh single \
  --task arena_hard_v2 \
  --model /capstor/store/.../checkpoint \
  --judge qwen3.5-27b \
  --reservation my-reservation

# Direct judge launch
python3 scripts/launch_judge.py \
  --preset qwen3.5-27b \
  --reservation my-reservation
```

By default the evaluation launcher submits a cleanup job that cancels automatically launched judges after all evaluation jobs finish. Pass `--keep-judge` to leave them running. Additional judge-specific options, such as `--health-timeout 1800`, can be supplied through `--judge-args`.

#### Experimental OpenAI GPT judge path

The separate `gpt` suite reserves a stable task/config path for the planned
OpenAI GPT judge implementation in `swiss-ai/lm-evaluation-harness`:

```bash
export OPENAI_API_KEY="..."
bash scripts/launch_evaluations.sh gpt \
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
bash scripts/launch_evaluations.sh olmo-complete --model allenai/OLMo-2-1124-7B --num-fewshot 5

# Large model with vLLM, eight tasks per chunk and four concurrent chunks
bash scripts/launch_evaluations.sh default \
  --model Qwen/Qwen2.5-72B-Instruct --backend vllm \
  --chunk-size 8 --max-parallel 4

# Run all models from a batch script on the safety suite
bash scripts/launch_evaluations.sh olmo-safety \
  --script runners/hf_eval_multiple_other_models.sh --chunk-size 4
```

#### Deprecated option:

| `--bos` | Prepend BOS token (deprecated: previously for Apertus models, now automatically infered from chat temlate) |

---

## Graceful / Resumable Launcher

Graceful execution is the default behavior of `scripts/launch_evaluations.sh`. The launcher scans existing harness outputs, evaluates only missing tasks, and uploads one aggregate W&B run. The old `scripts/launch_evaluations_gracefuly.sh` command remains as a deprecated compatibility wrapper.

```bash
bash scripts/launch_evaluations.sh custom \
  --task-file configs/apertus/tasks_posttrain_final.txt \
  --model /capstor/store/.../apertus-1.5-checkpoint \
  --table-metrics configs/apertus/tasks_posttrain_final_main_table.txt \
  --wandb-entity apertus --wandb-project apertus-1.5-post-training-v0.0 \
  --chunk-size 8 --max-parallel 4 --max-retries 1
```

### How it works

1. The task list is normalized and existing `eval_*` result directories are scanned.
2. Missing tasks are grouped into `--chunk-size N` chunks and submitted as one Slurm array. `--max-parallel` adds the array `%N` concurrency limit; by default every chunk may run.
3. A CPU controller runs with an `afterany` dependency, so it runs even when an array element fails.
4. The controller rescans results. If tasks remain and retry budget is available, it submits only those tasks again with the chunk size halved.
5. Completed outputs are merged and uploaded once. When retries are exhausted, successful tasks are still aggregated and `final_failed_tasks.txt` is retained in the run's controller state directory.

Use `--failure-policy fail-fast` to recover the original single-job behavior. `--force-tasks`, `--merge-only`, and `--debug` support targeted reruns, aggregation-only recovery, and submission dry runs.

---

## Task Configuration

Task lists are plain text files in `configs/apertus/...` with one task name per line. Comments (`#`) and blank lines are supported:

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
bash scripts/launch_evaluations.sh olmo-complete --model my-model
```

Available task names can be found in `lm_eval_reference/tasks/` or by running `lm_eval --tasks list`.

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
how long that trace was and whether it was well-formed. Enable all of it with one flag:

```bash
bash scripts/launch_evaluations.sh single --task gsm8k_cot --model Qwen/Qwen3-8B --thinking
```

> [!WARNING]
> **`--enable-thinking` on its own records nothing.** It is purely a chat-template argument that
> decides whether the model reasons. The trace strip and every thinking metric are armed by a known
> reasoning **close token** — supplied with `--think-end-token '</think>'` or discovered with
> `--autodetect-think-tokens`. `--thinking` wires both up for you; the launcher refuses to submit a
> job that would silently record nothing. If the model's chat template declares no reasoning
> tokens, autodetection fails loudly *inside the job* (after queueing, at model construction) —
> for such models pass `--think-end-token` explicitly or skip `--thinking`.

### The four independent switches

| Question                                    | Flag                                             | Default                                                                                 |
|---------------------------------------------|--------------------------------------------------|-----------------------------------------------------------------------------------------|
| Does the model reason?                      | `--enable-thinking`                              | vLLM: off (this repo pins `enable_thinking=False`); hf: the chat template's own default |
| Are the reasoning tokens discovered?        | `--autodetect-think-tokens`                      | off — the template is never scanned                                                     |
| Does the trace get stripped before scoring? | *(implicit)* whenever a **close** token is known | off                                                                                     |
| Are the thinking metrics recorded?          | `--track-thinking-metrics`                       | on iff a close token is known                                                           |

`--thinking` sets the first, second and fourth, adds `--log-length-metrics`, and forces the chat
template on (the reasoning tokens live in it). Any granular flag you pass overrides the umbrella.

Thinking runs also change **generation**: sampling is forced (`do_sample=true`,
`THINK_TEMPERATURE=0.6`, `THINK_TOP_P=0.95` — reasoning degrades under greedy), and the default
generation budget rises to 8192 tokens, floored at each task's own YAML `max_gen_toks` (AIME keeps
its 32768). Override via the `THINK_*` / `MAX_NEW_TOKENS` env vars (see
[SBATCH Scripts](#sbatch-scripts)); `NOTHINK_TEMPERATURE` enables the same sampling for no-think
ablations.

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
| `vllm` (recommended) | full            | forwarded always; this repo defaults it to `False`                               |
| `hf`                 | full            | forwarded **only when explicitly set**; otherwise the template's default applies |
| `sglang`             | available       | uses the SGLang backend and its dedicated container                              |
| `megatron_lm`        | **unsupported** | requesting thinking metrics is a hard error                                      |
| `openai`             | **unsupported** | requesting thinking metrics is a hard error                                      |

### Examples

```bash
# Reasoning model, everything on, quick smoke test
bash scripts/launch_evaluations.sh single --task gsm8k_cot \
  --model Qwen/Qwen3-8B --thinking --limit 20

# Explicit tokens rather than template auto-detection
bash scripts/launch_evaluations.sh posttrain --model my/reasoner --backend vllm \
  --enable-thinking --think-start-token '<think>' --think-end-token '</think>' \
  --log-length-metrics

# Measure format well-formedness only, no length aggregation
bash scripts/launch_evaluations.sh olmo-main --model my/reasoner --autodetect-think-tokens

# Response length of a NON-reasoning model: no close token needed, since response_length_*
# is recorded for every generative response whether the model thinks or not
bash scripts/launch_evaluations.sh posttrain --model meta-llama/Llama-3.1-8B-Instruct --log-length-metrics
```

The main launcher forwards all of these flags to every chunk:

```bash
bash scripts/launch_evaluations.sh posttrain \
  --model /capstor/.../my-reasoner --thinking
```

Thinking runs get an isolated run name (`<model-basename>-think`, override with `--name`), so their
results and W&B run never collide with the same model's non-thinking eval — see
[Graceful / Resumable Launcher](#graceful--resumable-launcher).

### Building a thinking-only table

`make_html_table.py --thinking` creates a separate, thinking-only table: one group per metric family, one row per task. It reuses
the suite's **existing task list** as the metric source.

```bash
python make_html_table.py --thinking \
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

## Every Eval Ever and Hugging Face exports

Completed lm-evaluation-harness runs can be converted into:

- [Every Eval Ever (EEE)](https://huggingface.co/datasets/evaleval/EEE_datastore) schema `0.2.2` aggregate records
- optional EEE instance-level JSONL records from lm-eval sample logs
- reviewable Hugging Face `.eval_results/*.yaml` previews for registered Hub benchmarks

The exporter is deliberately local-only: it does not upload files, open pull
requests, or read an HF token.

### Export a completed run

Point the exporter at either a single `results_*.json` file or a directory that
contains exactly one result file. For chunked runs, use the merged directory
produced by `merge_split_results.py`.

```bash
python scripts/export_eval_results.py export \
  /path/to/merged_eval/results_2026-07-27T12-00-00.json \
  --output-dir eval-results/Apertus-release-2026-07 \
  --model-id swiss-ai/Apertus-8B-Instruct-2607 \
  --include-samples
```

`--model-id` is optional when the result log contains an `owner/model` ID, but
it is required when the evaluation used a local checkpoint path.
Evaluator provenance defaults to `first_party` for `swiss-ai/*` models and
`third_party` for other owners; override it with `--evaluator-relationship`
when a run was collaborative or has a different relationship.

The output is organized for review and later submission:

```text
eval-results/Apertus-release-2026-07/
├── manifest.yaml
├── eee/data/
│   └── <canonical-benchmark>/<developer>/<model>/
│       ├── <uuid>.json
│       └── <uuid>_samples.jsonl
└── huggingface/.eval_results/
    ├── gsm8k.yaml
    └── mmlu-pro.yaml
```

The generated Hugging Face YAML links to the EEE record's expected immutable
`flat/objects/<uuid-prefix>/<uuid>.json` location. Submit and merge the EEE
records before copying those YAML previews into a model repository, otherwise
the source links will not resolve yet.

### Validate and review

```bash
# Check generated records, instance checksums, and expected files.
python scripts/export_eval_results.py validate \
  eval-results/Apertus-release-2026-07

# Verify EEE collection names and Hugging Face benchmark task IDs remotely.
python scripts/export_eval_results.py check-mappings
```

The export manifest (`manifest.yaml`, containing JSON-compatible YAML) lists:

- generated EEE records and HF previews
- lm-eval tasks skipped because they have no reviewed mapping
- `_self_consistency` task names resolved to a reviewed base-task mapping
- missing run-level metadata such as generation temperature or maximum tokens
- `publishing_performed: false`, confirming that the exporter only wrote local files

The YAML extension is intentional: the official EEE validator recursively
treats every `.json` file as an aggregate `EvaluationLog`. Keeping exporter
bookkeeping out of `.json` means the complete output directory can be passed
directly to `every_eval_ever validate`. The local validator still accepts
legacy exports containing `manifest.json`, and re-exporting migrates that file
to `manifest.yaml`.

Use `--strict-mappings` when every numeric task in a run must be mapped. Without
it, unmapped tasks are skipped and reported rather than guessed.

### Task mappings

Mappings live in `configs/eval_export/task_mappings.json`. Each entry maps an
exact lm-eval task name to:

- the canonical EEE datastore collection directory
- the EEE `evaluation_name`
- the source dataset ID
- optional ordered EEE metric candidates and canonical metric-ID overrides
- optionally, a registered Hugging Face benchmark dataset, task ID, and ordered
  metric candidates

The repository's `_self_consistency` suffix is handled as a controlled task
variant: the exporter removes only that exact suffix and requires the remaining
base task to have a reviewed mapping. Its mapping can define
`self_consistency_metric_candidates`; `{repeats}` is expanded from the lm-eval
task config so, for example, `exact_match,mean@{repeats}` selects
`exact_match,mean@32`. The original task name, repeat count, and
`self_consistency` variant remain recorded in the exported metadata.

Do not otherwise use fuzzy matching for benchmark variants. For example,
`gpqa_main_cot_zeroshot` is not mapped to the datastore's `gpqa_diamond`
collection, and `gsm8k_platinum` is not mapped to `gsm8k`. Add mappings only
after confirming that the dataset and evaluation protocol are equivalent.
For the same reason, `bfcl_v3` and `swiss_ai_charter_alignment` currently remain
explicitly unmapped: the EEE datastore has no reviewed one-to-one mapping for
these exact lm-eval protocols and native score scales. They are reported in the
export manifest's `skipped_unmapped_tasks` list.

The exporter preserves scores in their native lm-eval scale. It never
automatically multiplies proportions by 100.

---

## W&B Integration

### Metrics Upload

Results are automatically uploaded to W&B after evaluation completes (or after aggregation for chunked jobs). Each model gets a W&B run with:

- **`main_results`** table: summary metrics specified in the `*_main_table.txt` config
- **Flat metrics**: all task metrics logged as `task_name/metric_name`
- **`eval_duration`**: wall-clock time for the evaluation

Because *every* flat metric is uploaded — not just the `*_main_table.txt` subset — the
[thinking metrics](#thinking--reasoning-metrics) reach W&B as `task_name/thinking_format_correct`
without any uploader configuration. The length families (`task_name/thinking_length_tokens`,
`task_name/response_length_tokens`, …) arrive the same way, but only once the eval was launched
with `--log-length-metrics`, which is what makes the harness aggregate them in the first place.

### Sample Upload (Stratified)

Per task, example prompts are uploaded as W&B tables below `samples/{task_name}`.
The model name is not repeated in the table key because the table already belongs
to that model's W&B run. Long run IDs and task names are shortened with stable
hashes, leaving room for W&B's generated artifact suffix and avoiding its
128-character artifact-name limit. The run keeps its full display name.

- **2 positive samples** by default (correctly answered, metric = 1.0)
- **3 negative samples** by default (incorrectly answered, metric = 0.0)

Samples are classified using binary metrics (`acc`, `exact_match`, `em`, `pass@1`). Each sample includes an `is_correct` field (`true`/`false`/`null`) for downstream filtering. If a task has no binary metric (e.g., perplexity), 10 random samples are uploaded instead.

If one group is underrepresented (e.g., a model gets almost everything right), the remaining slots are filled from the other group.

The stratified counts are configurable via `n_positive` and `n_negative` parameters in `create_model_evaluation_from_results()`.

### Retrieving Samples via API

Samples are stored as W&B Tables, retrievable via the W&B API:

```python
import wandb

api = wandb.Api()
run = api.run("entity/project/run_id")

# Get a specific task's samples. This also works when its table key was shortened.
task_name = "mmlu"
table_key = run.summary["sample_table_keys"][task_name]
table = run.summary[table_key]
```

Each row in the table is a flattened sample dict containing:

| Field | Description |
|-------|-------------|
| `task_name` | Full lm-eval task name, even if the W&B table key was shortened |
| `doc/*` | Original question/document fields from the dataset |
| `target` | Expected answer |
| `arguments/*` | The prompt sent to the model |
| `filtered_resps` | Model's response after filtering |
| `is_correct` | Stratification label: `true`, `false`, or `null` (non-binary task) |
| `acc`, `exact_match`, etc. | Task-specific metric values |

### Manual Upload

```bash
# Upload a single model's results
python -m scripts.alignment.update_wandb_alignment \
  --entity apertus --project swissai-evals \
  --name Llama-3.1-8B-Instruct \
  --logs_root /path/to/harness/eval_20250726_003542_12345 \
  --main_metrics mmlu/acc arc_challenge/acc_norm gsm8k_cot/exact_match \
  --eval_duration 3600

# Batch upload all models from a logs directory
python -m scripts.alignment.update_wandb_all_models \
  --entity apertus --project swissai-evals \
  --logs_root /path/to/eval-logs
```

---

## Reporting: Building Result Tables

Both table scripts must be run manually. They read every score from **W&B run summaries** (not from disk); the metrics file
selects *which* `task/metric` keys to pull and how to group them.

### `make_html_table.py` — interactive HTML results table

A self-contained, collapsible HTML table: one column per model, benchmarks grouped by the `#`
headers in the metrics file.

```bash
python make_html_table.py \
  --metrics-file configs/apertus/tasks_posttrain_final_main_table.txt \
  --entity apertus --project apertus-1.5-post-training-v0.0 \
  --models my-run-a my-run-b \
  --output eval_table.html
```

- **`--metrics-file`** (required): a `*_main_table.txt` — each `#` line is a category group, each
  other line a `task/metric[,filter]` W&B summary key.
- **Models**: the pinned Apertus baselines are included automatically; add yours with `--models`
  and/or `--models-file` (one run name per line). `--no-baselines` drops the pins.
- **Layout**: `--no-split` (single table, no train/test), `--flat` (plain rows, no group headers),
  `--title`, `--rename 'run-name=Display Name'`.
- **Baseline controls**: `--sft-baseline <run>`, `--no-sft-baseline`, `--instruct-baseline`.
- **Extra info rows**: `--show-word-count` (AlpacaEval avg word count), `--more-details` (averaged
  degeneration), `--overrefusal` (ORBench over-refusal rate).
- **`--json-output <path>`**: also dump the combined metrics (grouped, with group + overall
  averages) as JSON — only the `--models`, no baselines.
- **`--thinking`**: switches to the thinking-only table — see
  [Building a thinking-only table](#building-a-thinking-only-table).
- `--debug` prints the fetched W&B summary keys per run (useful when a cell is blank).

### `make_table.py` — hyperparameter-sweep table (PNG + CSV)

Targets sweep comparisons rather than a leaderboard: it enumerates the run sub-directories
under `--base-model-path`, parses each name for `beta` / batch-size / `lr` / length-norm, fetches
its W&B summary, and renders **delta improvements over a baseline** (`Avg_Imp`; or absolute scores
with `--absolute`). Writes `eval_table.png` **and** `eval_table.csv` into the `--output` directory.

```bash
python make_table.py \
  --base-model-path /path/to/sweep_runs \
  --entity apertus --project apertus-1.5-post-training-v0.0 \
  --metrics-file configs/apertus/tasks_posttrain_main_table.txt \
  --output ./eval_tables
```

- **`--base-model-path`** (required, one or more dirs): each contains the run sub-directories to
  look up in W&B by name.
- **Baselines**: the base run defaults to `baseline-apertus-1-sft` (override `--baseline`); a DPO
  baseline row is shown unless `--no-dpo-baseline`. Add `--extra-baseline` / `--extra-run` (with
  matching `-name` flags) for extra rows.
- **Benchmark selection**: `--include` / `--exclude` (substring match), `--metrics-file` (one or
  more, merged).
- **Rows**: `--top-n` (best N by `Avg_Imp`), `--complete-only` (drop rows with any N/A),
  `--model-order`, `--raw-names` (don't parse lr/beta/ebs), `--rename`, `--label`, `--filter`.
- **Columns**: `--absolute` (raw scores instead of deltas), `--wordcount` (append avg word count).

---

## SBATCH Scripts

### `scripts/evaluate.sbatch`

Primary SLURM job script for HuggingFace-compatible model evaluation.

**Resources**: 1 node, 4 GPUs, 288 CPUs, 460GB memory, 12h time limit.

**Positional arguments**: `<model_path> <name>`

**Environment variables** (all optional, with defaults):

| Variable | Default | Description |
|----------|---------|-------------|
| `TASKS` | `configs/apertus/tasks_constrained.txt` | Task list file or comma-separated task names |
| `TABLE_METRICS` | `configs/apertus/tasks_constrained_main_table.txt` | Metrics for W&B summary table |
| `LM_EVAL_BACKEND` | `hf` | Backend: `hf` (accelerate), `vllm`, `sglang`, `megatron_lm`, `openai` (OpenAI-compatible API) |
| `API_BASE_URL` | (unset) | `openai` backend only, **required**: the OpenAI-compatible endpoint. Bare host / `/v1` root / full endpoint URL all accepted. |
| `API_MODEL_NAME` | same as model | `openai` backend only: the `model` field sent in requests, when the server registers the model under a different id |
| `API_NUM_CONCURRENT` | `8` | `openai` backend only: concurrent requests (batch size is pinned to 1; this is the throughput knob) |
| `API_MAX_RETRIES` | `3` | `openai` backend only: retries per failed request |
| `OPENAI_API_KEY` | `scripts/openai_api_key.txt`, then `CSCS_SERVING_API` | OpenAI GPT judge or `openai` backend bearer token |
| `LM_EVAL_HARNESS_BRANCH` | repository default | Branch/ref installed from the task-selected harness repository |
| `APPLY_CHAT_TEMPLATE` | `false` | Apply chat template for instruct models |
| `TOKENIZER` | same as model | Custom tokenizer path |
| `BOS` | `false` | Prepend BOS token |
| `BS` | `auto:20` | Batch size |
| `SIZE` | `1` | Model size in billions (for model parallelism) |
| `MAX_LENGTH` | `4096` | Maximum input sequence length |
| `MAX_NEW_TOKENS` | `2048` (`8192` with thinking) | Generation-budget *floor*: raised to the largest per-task YAML `max_gen_toks` (e.g. AIME's 32768), never lowered. Thinking raises the default; an explicit value always wins. |
| `THINK_TEMPERATURE` / `THINK_TOP_P` | `0.6` / `0.95` | Sampling for thinking runs (`do_sample=true` is forced — reasoning degrades under greedy) |
| `THINK_REPETITION_PENALTY` | (unset) | Optional knob against degenerate looping in thinking runs (e.g. `1.05`) |
| `NOTHINK_TEMPERATURE` / `NOTHINK_TOP_P` | (unset) / `0.95` | Optional sampling for NO-think runs (greedy-vs-sampling ablations); inert unless `NOTHINK_TEMPERATURE` is set |
| `LIMIT` | (unset) | Limit number of samples per task |
| `NUM_FEWSHOT` | (unset) | Global few-shot override |
| `EVAL_ENV_MANIFEST` | required | Immutable base-environment and harness-overlay paths produced by `prepare_eval_env.sbatch` |
| `EVAL_CHUNKS_FILE` | (unset) | One comma-separated task chunk per line; indexed by `SLURM_ARRAY_TASK_ID` |
| `LOGS_ROOT` | `/capstor/.../eval-logs` | Root directory for evaluation logs |
| `WANDB_ENTITY` | `apertus` | W&B entity |
| `WANDB_PROJECT` | `swissai-evals-test` | W&B project |
| `ENABLE_THINKING` | `false` | Chat-template argument: whether the model reasons. Emitted for `hf` **only when set explicitly**. |
| `AUTODETECT_THINK_TOKENS` | `false` | Read the reasoning open/close tokens from the chat template |
| `THINK_START_TOKEN` | (unset) | Force the reasoning open token, e.g. `<think>` |
| `THINK_END_TOKEN` | (unset) | Force the reasoning close token, e.g. `</think>`. Arms the strip and the metrics. |
| `TRACK_THINKING_METRICS` | (unset → derive) | `true`/`false` to force the thinking metrics on/off |
| `LOG_LENGTH_METRICS` | `false` | Add `--log_length_metrics` (aggregates `response_length_*` / `thinking_length_*`) |

The script auto-detects RULER long-context tasks and adjusts `MAX_LENGTH` and `max_model_len` accordingly.

Requesting any thinking metric with `LM_EVAL_BACKEND=megatron_lm` aborts the job: the harness has no
reasoning-token support for that backend, so it would otherwise record nothing silently. See
[Thinking / Reasoning Metrics](#thinking--reasoning-metrics).

---

## Multi-Model Scripts

Scripts in `runners/` define `MODEL_CHECKPOINTS` associative arrays and source `hf_base_runner.sh`:

```bash
# runners/hf_eval_multiple_other_models.sh
declare -A MODEL_CHECKPOINTS=(
    ["Llama-3.1-8B-Instruct"]="meta-llama/Llama-3.1-8B-Instruct"
    ["OLMo-2-1124-7B-Instruct"]="allenai/OLMo-2-1124-7B-Instruct"
    # Uncomment models as needed...
)
export APPLY_CHAT_TEMPLATE=true
source runners/hf_base_runner.sh "SFT models"
```

`hf_base_runner.sh` handles the model loop and delegates resumable chunk orchestration to `scripts/evaluation_orchestrator.sh`. It respects `EVAL_CHUNK_SIZE`, `EVAL_MAX_PARALLEL`, `EVAL_MAX_RETRIES`, `SBATCH_SCRIPT`, and `WANDB_*`.

### Model Registry

See `configs/models.md` for the full list of available models with their HF paths, local checkpoint paths, and required special flags. Key model families:

- **Apertus** (1.0)
- **Meta Llama** (3.1, 3.3)
- **OLMo** (2-1124, 2-0325, 3)
- **Qwen** (2.5, 3)
- **Gemma** (3), **EuroLLM**, **Mistral**, **SmolLM**, **Marin**, and others

---

## Container Setup

The pipeline runs inside containers managed by enroot/pyxis on SLURM. The available container configurations:

| Config | Base Image | Use Case |
|--------|-----------|----------|
| `env.toml` | Based on CSCS container image | Standard HF evals |
| `env_vllm.toml` | CSCS base image + vLLM 0.16 built from source | vLLM evals |
| `env_sglang.toml` | CSCS SGLang CUDA 13 image | SGLang evals |

Evaluation dependencies use a shared two-tier cache under `${EVAL_ENV_CACHE_ROOT:-$SCRATCH/eval-envs}`:

1. A base virtual environment is keyed by backend, Python/container configuration, and `requirements/eval-runtime.txt`.
2. A lightweight harness overlay is keyed by the base environment, harness repository, and resolved commit SHA.

`prepare_eval_env.sbatch` builds missing tiers once on a CPU node using a file lock. Evaluation chunks only activate the immutable base and prepend the overlay to `PYTHONPATH`. A moving branch is therefore resolved once per model run: pushed changes propagate on the next launch, while all chunks in one run use exactly the same commit.

The cache is disposable. If `$SCRATCH` retention removes either tier, the next
launch rebuilds it automatically. Cache hits validate the environment rather
than trusting the completion marker alone. Evaluation jobs also re-check the
prepared paths and perform a last-resort rebuild if retention removes them
between the CPU preparation job and GPU-job startup. Set `EVAL_ENV_CACHE_ROOT`
to a longer-lived shared filesystem if retaining environments beyond the
cluster's scratch window is preferable.

---

## Notes

> [!NOTE]
> **vLLM vs HF inference**: Generation task results (gsm8k, squadv2) may differ slightly between backends (for instruction-tuned models). Only compare results across models using the same backend. We recommend performing all evaluations with the `vllm` backend (default) to ensure reproducibility.
- **OpenAI-compatible API backend (`--backend openai`)**: evaluates against an already-running endpoint (e.g. `vllm serve`, the CSCS serving platform) instead of loading the model inside the job. It uses lm-eval's `local-completions` against `/v1/completions`, which serves **both** generative and loglikelihood/MC tasks (mixed suites work) *provided* the server returns prompt logprobs with echo (vLLM does; most commercial APIs do not). With the chat template on, the harness renders the model's template client-side via the HF tokenizer — so the tokenizer must be resolvable (use `--tokenizer` when the served model name is not a pullable HF repo). `API_CHAT_ENDPOINT=true` switches to `/v1/chat/completions` (server-side template; generative tasks ONLY). Auth uses `OPENAI_API_KEY` (defaults to the CSCS serving key). Note the job still requests the resources declared in `evaluate.sbatch` even though no GPU is used.
- **Megatron-LM**: To run Megatron-LM models natively, clone the [NVIDIA Megatron-LM repository](https://github.com/NVIDIA/Megatron-LM) into the evals-post-train directory (or change the location via the launch script).
- **Time limits**: The default 12h SLURM limit applies to each chunk. Adjust `--chunk-size` to keep individual jobs below it and `--max-parallel` to control concurrent nodes.
- **WANDB_API_KEY**: Must be available either as an environment variable or in `scripts/wandb_api_key.txt`.
- **HF_TOKEN**: Must be available either as an environment variable or in  `scripts/hf_token.txt`.
- **OPENAI_API_KEY**: Required for the optional `gpt` suite, either as an environment variable or in `scripts/openai_api_key.txt`.
- **CSCS_SERVING_API**: Must be available either as an environment variable or in `scripts/cscs_serving_api_key.txt` to run LLM-as-a-judge evals (e.g. AlpacaEval). Key can be optained [here](https://serving.swissai.cscs.ch).

---

## Extending the Pipeline

### Adding a New Inference Backend

The sbatch scripts support `hf`, `vllm`, `sglang`, `megatron_lm`, and `openai` backends. To add a new one:

1. Add a new `elif` block in `evaluate.sbatch` at the `LM_EVAL_BACKEND` dispatch section
2. Set appropriate `COMMON_MODEL_ARGS` for the new backend
3. Add any required pip install commands to `INSTALL_CMD`

### Adding a New Task

If the task exists in lm-eval-harness:
1. Add the task name to your task list config file
2. Add the `task_name/metric_name` entry to the corresponding `*_main_table.txt`

If you need a custom task:
1. Create a YAML task config in `lm_eval_reference/tasks/your_task/`
2. Register it following the [lm-eval-harness task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)

### Customizing Sample Upload

The stratified sample selection in `scripts/alignment/wandb_alignment_utils.py` can be adjusted:

```python
# In create_model_evaluation_from_results():
model_eval = create_model_evaluation_from_results(
    model_name="my-model",
    eval_dir=Path("/path/to/eval_dir"),
    n_positive=5,   # number of correct samples to upload (default: 2)
    n_negative=15,  # number of incorrect samples to upload (default: 3)
)
```

The binary metrics used for correctness classification are defined in `BINARY_METRICS` at the top of `wandb_alignment_utils.py`. Add new metric names there if your tasks use different correctness indicators.

---

### Task Separation (Legacy)

The `swissai_eval` hierarchy and approximate time distribution:

```
swissai_eval (100%)
├── english  (~46%)
│   ├── english_pt1 (~10%)
│   └── english_pt2 (~36%)
└── multilingual (~54%)
    ├── multilingual_pt1 (~8%)
    └── multilingual_pt2 (~46%)
```

Rule of thumb for fitting within the 12h limit: ensure `2.5 * percentage * model_size_B < 100`.

---

## Repository Structure

```
evals/
├── configs/                         # Task lists and model registry
│   ├── _*.txt                       # actual task lists
│   ├── _*_main_table.txt            # Corresponding metric specs for W&B summary tables
│   ├── models.md                    # Model registry with paths and special flags
│   ├── apertus/                     # Apertus task lists (english, multilingual, etc.)
│   ├── olmo/                        # OLMo3 benchmark suites (easy, main, heldout, safety, longcontext, complete)
├── scripts/
│   ├── launch_evaluations.sh  # Main launcher (recommended entry point)
│   ├── launch_evaluations_gracefuly.sh # Deprecated compatibility wrapper
│   ├── evaluation_orchestrator.sh # Resumable chunk arrays, retries, aggregation
│   ├── prepare_eval_env.sbatch # Builds/reuses the two-tier environment
│   ├── launch_judge.py        # Launches judge models for LLM-as-a-judge tasks
│   ├── evaluate.sbatch        # SLURM job script for HF/vLLM model evaluation
│   ├── aggregate_splits.sbatch   # Aggregation job for chunked evaluations
│   └── alignment/                   # Python package for W&B upload and data handling
│       ├── wandb_alignment_utils.py # Core upload logic with stratified sample selection
│       ├── update_wandb_alignment.py       # Per-model W&B upload script
│       ├── update_wandb_all_models.py      # Batch upload for all models
│       ├── merge_split_results.py          # Merges results from chunk jobs
│       └── data_structures.py              # Sample, Metric, Task, ModelEvaluation classes
├── make_html_table.py                # Reporting: interactive HTML results table (reads W&B)
├── make_table.py                     # Reporting: hyperparameter-sweep table, PNG + CSV (reads W&B)
├── runners/              # Multi-model evaluation scripts
│   ├── hf_base_runner.sh            # Generic runner (delegates chunk orchestration)
│   ├── hf_eval_multiple_other_models.sh
│   ├── hf_eval_multiple_other_base_models.sh
│   ├── hf_eval_multiple_apertus_models.sh
│   └── hf_eval_multiple_apertus_base_models.sh
├── containers/                      # Container specs (Docker, env.toml for enroot/pyxis)
│   ├── Dockerfile                   # CUDA 9.0+PTX, vLLM, FlashAttention-3
│   ├── env.toml                     # Standard container config
└── └── env_vllm.toml                # VLLM-based container config
```
