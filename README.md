# SwissAI Evaluation Pipeline

Evaluation infrastructure for benchmarking Large Language Models on SLURM clusters (CSCS Alps). Built on top of [lm-evaluation-harness](https://github.com/swiss-ai/lm-evaluation-harness) with W&B integration for results tracking.

## Quick Start

```bash
# Evaluate a single model on the benchmark suite (with custom name)
bash scripts/launch_evaluations.sh default --model meta-llama/Llama-3.1-8B-Instruct --name Llama-Baseline

# Same, but split tasks across 4 parallel nodes for faster evaluation, name automatically infered
bash scripts/launch_evaluations.sh default --model meta-llama/Llama-3.1-8B-Instruct --splits 4

# Launch Megatron checkpoint without conversion (TODO: Verify), Megatron-iter defaults to: latest
bash scripts/launch_evaluations.sh olmo-easy --model /capstor/store/../apertus-.../checkpoints/ --backend megatron_lm --name Megatron-Test-260216 --megatron-iter 4000000

# Launch with vllm backend - recommended!
bash scripts/launch_evaluations.sh olmo-easy --model /capstor/store/../apertus-.../checkpoints/ --backend vllm

# Evaluate a base model with 5-shot and easy eval set (matching OLMo3 technical report settings)
bash scripts/launch_evaluations.sh olmo-easy --model Qwen/Qwen2.5-7B --num-fewshot 5

# Evaluate a small model on a single task, useful for testing newly implemented tasks
bash scripts/launch_evaluations.sh single --task multijail --model meta-llama/Llama-3.2-3B --backend vllm
```

## The Launch Script

`scripts/launch_evaluations.sh` is the primary entry point for running evaluations. It supports three model selection modes and multiple benchmark suites.

### Benchmark Suites

The launcher selects a benchmark suite from its first positional argument (`<mode>`). Apertus suites are defined in `configs/apertus/`, OLMo3 suites in `configs/olmo/`.

**Apertus suites** (`configs/apertus/`)

| Mode | Task list | Tasks | Description |
|------|-----------|-------|-------------|
| `default` | `tasks_default.txt` | 48 | Full Apertus 1.5 post-training suite: knowledge, math, code, reasoning, multilingual, instruction following, bias, cultural knowledge, safety |
| `posttrain` | `tasks_posttrain_final.txt` | 48 | Post-training suite incl. chat/arena (alpaca_eval, arena_hard_v01/v2), AIME, MATH-500, o(r)bench, system-prompt adherence (RealGuardrails: S-IFEval, TensorTrust, S-RuLES) |
| `multi-lingual` | `tasks_multilingual.txt` | 12 | Multilingual-only subset (global_mmlu, mgsm, hellaswag_multilingual, include, cultural_bench, multi-if, aya_redteaming, ...) |
| `apertus-previous` | `tasks_english.txt` | 19 | Previous (Apertus 1.0) English benchmark suite |
| `pretrain` | `tasks_pretrain.txt` | 30 | Pretraining suite (base-model loglikelihood/MC variants, few-shot MMLU); logs to W&B project `apertus-1.5-pre-training-v0.0` |
| `eval-debug` | `eval_debug.txt` | 8 | Small mix of loglikelihood + generative tasks to smoke-test the eval pipeline |
| `custom` | (none) | — | No suite configured; export `TASKS` and `TABLE_METRICS` yourself before launching |
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
| `--backend <hf\|vllm\|megatron_lm>` | Inference backend (default: from sbatch script). `vllm` recommended. |
| `--splits K` | Split task list across K parallel SLURM nodes per model (auto-clamped to the task count) |
| `--limit N` | Limit number of samples per task (forwarded as `--limit` to lm-eval-harness; default: no limit). Useful for quick sanity checks. |
| `--megatron-iter <iter>` | For Megatron-LM checkpoints, the iteration to evaluate (e.g. `8926`); defaults to `latest`. Exported as `CKPT_ITERATION`. |
| `--harness-branch B` | Install lm-evaluation-harness from a specific branch/ref (default: repo default branch) |
| `--judge <none\|auto\|preset>` | Judge-model control for LLM-as-a-judge tasks (alpaca_eval, arena_hard_v01/v2, multijail, aya_redteaming). `none` (default) disables auto-launch; `auto` scans the task list and launches needed judges; a preset name (e.g. `qwen3.5-27b`, `llama-3.3-70b`) launches that judge. |
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


### Examples

```bash
# OLMo3 paper-faithful 5-shot evaluation
bash scripts/launch_evaluations.sh olmo-complete --model allenai/OLMo-2-1124-7B --num-fewshot 5

# Large model with vLLM and 8-way task splitting
bash scripts/launch_evaluations.sh default \
  --model Qwen/Qwen2.5-72B-Instruct --backend vllm --splits 8

# Run all models from a batch script on the safety suite
bash scripts/launch_evaluations.sh olmo-safety \
  --script runners/hf_eval_multiple_other_models.sh --splits 4
```

#### Deprecated option:

| `--bos` | Prepend BOS token (deprecated: previously for Apertus models, now automatically infered from chat temlate) |

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

| Question | Flag | Default |
|---|---|---|
| Does the model reason? | `--enable-thinking` | vLLM: off (this repo pins `enable_thinking=False`); hf: the chat template's own default |
| Are the reasoning tokens discovered? | `--autodetect-think-tokens` | off — the template is never scanned |
| Does the trace get stripped before scoring? | *(implicit)* whenever a **close** token is known | off |
| Are the thinking metrics recorded? | `--track-thinking-metrics` | on iff a close token is known |

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

| Metric | Kind | Gated by |
|---|---|---|
| `thinking_format_has_open` | rate `[0,1]` | tracking on **and** an open token is known |
| `thinking_format_has_close` | rate `[0,1]` | tracking on |
| `thinking_format_correct` | rate `[0,1]` | tracking on |
| `response_length_{words,chars,tokens}` | raw count | `--log-length-metrics` |
| `thinking_length_{words,chars,tokens}` | raw count | `--log-length-metrics` |

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

| Backend | Support | `enable_thinking` |
|---|---|---|
| `vllm` (recommended) | full | forwarded always; this repo defaults it to `False` |
| `hf` | full | forwarded **only when explicitly set**; otherwise the template's default applies |
| `megatron_lm` | **unsupported** | requesting thinking metrics is a hard error |

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

The graceful launcher accepts and forwards all of these to its per-task `single` runs:

```bash
bash scripts/launch_evaluations_gracefuly.sh --task_file configs/apertus/tasks_posttrain_final.txt \
  --model /capstor/.../my-reasoner --thinking
```

Thinking runs get an isolated run name (`<model-basename>-think`, override with `--name`), so their
results and W&B run never collide with the same model's non-thinking eval — see
[Graceful / Resumable Launcher](#graceful--resumable-launcher).

### Building a thinking-only table

`make_html_table.py --thinking` renders one group per metric family and one row per task. It reuses
the suite's **existing task list** as the metric source — there is no separate config file to keep
in sync:

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

## Graceful / Resumable Launcher

`scripts/launch_evaluations_gracefuly.sh` is a **resumable, idempotent wrapper** around `launch_evaluations.sh`. Instead of launching a whole suite as one job, it inspects which tasks already have results on disk and only (re)launches the *missing* ones, then automatically aggregates everything once complete. This makes it the recommended entry point for large post-training suites where individual tasks may fail or time out and you don't want to re-run the entire suite.

```bash
bash scripts/launch_evaluations_gracefuly.sh \
  --task_file configs/apertus/tasks_posttrain_final.txt \
  --model /capstor/store/.../apertus-1.5-checkpoint \
  --table_metrics configs/apertus/tasks_posttrain_final_main_table.txt \
  --wandb_entity apertus --wandb_project apertus-1.5-post-training-v0.0 \
  --group_size 1
```

### How it works

1. **Scan**: reads `--task_file`, then scans the run's harness output directories (`<eval_prefix>/<run-name>/harness/eval_*/results_*.json`, plus the `-single` project variant) and marks each task that already has a result. The run name defaults to the model basename; thinking runs get a `-think` suffix and `--name` overrides it outright, so a reasoning run never collides with the same model's non-thinking results.
2. **Diff**: computes the set of *missing* tasks (expected − completed).
3. **Launch missing only**: groups missing tasks into batches of `--group_size` and submits each group via `launch_evaluations.sh single --task <group> --chat-template` (with `WANDB_MODE=disabled` for the per-task runs).
4. **Aggregate**: submits a follow-up job (`--dependency=afterok:<all_task_jobs>`) that re-runs this same script in `--merge_only` mode, which rebuilds the split markers and submits `aggregate_splits.sbatch` to merge all results and upload the final run to W&B.
5. If no tasks are missing on the first pass, it skips straight to marker rebuild + aggregation.

All [thinking flags](#thinking--reasoning-metrics) (`--thinking`, `--think-end-token`, …) are accepted
and forwarded verbatim to the per-task `single` runs in step 3. They are deliberately *not* forwarded to
the step-4 aggregator, which runs `--merge_only` and loads no model.

### Differences vs. `launch_evaluations.sh`

| Aspect | `launch_evaluations.sh` | `launch_evaluations_gracefuly.sh` |
|--------|-------------------------|-----------------------------------|
| Purpose | One-shot launch of a full suite (or split across nodes) | Resume/complete a partially-finished suite; only launches missing tasks |
| Suite selection | Positional `<mode>` (named suite) | Explicit `--task_file <path>` (any task list) |
| Model arg | `--model` / `--script` / default array | `--model` only |
| Granularity | One job per model (optionally `--splits K`) | One job per **task group** (`--group_size`, default 1 = per-task) |
| Idempotency | Re-runs everything every time | Skips tasks that already have results on disk |
| Aggregation | Triggered by `--splits` flow | Always; chained automatically via `afterok` + `--merge_only` |
| W&B during task runs | Uploads per job | `WANDB_MODE=disabled` per task; only the final aggregator uploads |
| SLURM placement | Defaults from sbatch script | `--account` / `--reservation` flags (cache dirs redirected to `$SCRATCH/.cache`) |
| Extra controls | judge / splits / backend / fewshot flags | `--force_tasks <substr,...>` to force re-eval, `--merge_only`, `--debug` (dry run) |

Key flags: `--task_file` and `--model` (required), `--table_metrics`, `--eval_prefix`, `--account`, `--reservation`, `--wandb_entity`, `--wandb_project`, `--group_size`, `--tokenizer`, `--name <run-name>` (override the run name — dirs + W&B run; defaults to the model basename, with a `-think` suffix for thinking runs), `--force_tasks <comma-separated substrings>` (drop matching tasks from the completed set to re-run them), `--merge_only` (skip launching, just rebuild markers + aggregate), and `--debug` (dry run — prints what would be submitted without submitting).

> [!NOTE]
> Under the hood the graceful launcher delegates each task to `launch_evaluations.sh` in `single` mode and always applies the chat template, so it is intended primarily for post-training (instruct) checkpoints.

---

## Notes

> [!NOTE]
> **vLLM vs HF inference**: Generation task results (gsm8k, squadv2) may differ slightly between backends (for instruction-tuned models).. Only compare results across models using the same backend. We recommend to perform all evaluations with the `vllm` backend (default) to ensure reproducability.
- **Megatron-LM** If you want to run Megatron-LM models natively, you need clone the [Nvdia Megatron-LM repository](https://github.com/NVIDIA/Megatron-LM) to the evals-post-train directory (or change the location with the launch script):
- **Time limits**: The default 12h SLURM limit works for most evaluations. For large suites on large models, use `--splits` to parallelize.
- **WANDB_API_KEY**: Must be available either as an environment variable or in `scripts/wandb_api_key.txt`.
- **HF_TOKEN**: Must be available either as an environment variable or in  `scripts/hf_token.txt`.
- **CSCS_SERVING_API**: Must be available either as an environment variable or in `scripts/cscs_serving_api_key.txt` to run LLM-as-a-judge evals (e.g. AlpacaEval). Key can be optained [here](https://serving.swissai.cscs.ch).

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
│   ├── launch_evaluations_gracefuly.sh # Resumable launcher (only runs missing tasks, auto-aggregates)
│   ├── launch_judge.py        # Launches judge models for LLM-as-a-judge tasks
│   ├── evaluate.sbatch        # SLURM job script for HF/vLLM model evaluation
│   ├── aggregate_splits.sbatch   # Aggregation job for split evaluations
│   └── alignment/                   # Python package for W&B upload and data handling
│       ├── wandb_alignment_utils.py # Core upload logic with stratified sample selection
│       ├── update_wandb_alignment.py       # Per-model W&B upload script
│       ├── update_wandb_all_models.py      # Batch upload for all models
│       ├── merge_split_results.py          # Merges results from split evaluation jobs
│       └── data_structures.py              # Sample, Metric, Task, ModelEvaluation classes
├── runners/              # Multi-model evaluation scripts
│   ├── hf_base_runner.sh            # Generic runner (handles split-aware job submission)
│   ├── hf_eval_multiple_other_models.sh
│   ├── hf_eval_multiple_other_base_models.sh
│   ├── hf_eval_multiple_apertus_models.sh
│   └── hf_eval_multiple_apertus_base_models.sh
├── containers/                      # Container specs (Docker, env.toml for enroot/pyxis)
│   ├── Dockerfile                   # CUDA 9.0+PTX, vLLM, FlashAttention-3
│   ├── env.toml                     # Standard container config
└── └── env_vllm.toml                # VLLM-based container config
```

---

## Parallel Task Splitting

For evaluations that would exceed the 12h SLURM time limit (or just to get results faster), the `--splits K` option distributes tasks across K parallel SLURM nodes.

### How It Works

1. The launcher submits K `sbatch` jobs, each with `NUM_SPLITS=K` and `SPLIT_INDEX=0..K-1`
2. Each job reads the task list, splits it into K chunks, and runs only its chunk
3. Each split job writes a marker file to `$HARNESS_DIR/split_markers/split_<i>.txt`
4. An aggregation job (`aggregate_splits.sbatch`) is submitted with `--dependency=afterok:<all_split_job_ids>` -- it only runs once all splits succeed
5. The aggregation job calls `merge_split_results.py` to combine `results_*.json` files and copy sample JSONL files, then uploads merged results to W&B

```
sbatch split-0  ─┐
sbatch split-1  ─┤
sbatch split-2  ─┤──> afterok ──> sbatch aggregate ──> W&B upload
sbatch split-3  ─┘
```

No manual dependency management is needed -- the launcher handles everything via `sbatch --parsable` and `--dependency`.

### Race Condition Safety

- Split jobs do **not** upload to W&B individually. Only the single aggregation job does the upload, avoiding concurrent `wandb.init(resume="allow")` conflicts.
- Output directories are unique per job ID (`eval_<timestamp>_$SLURM_JOBID`), so file writes never collide.

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

## W&B Integration

### Metrics Upload

Results are automatically uploaded to W&B after evaluation completes (or after aggregation for split jobs). Each model gets a W&B run with:

- **`main_results`** table: summary metrics specified in the `*_main_table.txt` config
- **Flat metrics**: all task metrics logged as `task_name/metric_name`
- **`eval_duration`**: wall-clock time for the evaluation

Because *every* flat metric is uploaded — not just the `*_main_table.txt` subset — the
[thinking metrics](#thinking--reasoning-metrics) reach W&B as `task_name/thinking_format_correct`
without any uploader configuration. The length families (`task_name/thinking_length_tokens`,
`task_name/response_length_tokens`, …) arrive the same way, but only once the eval was launched
with `--log-length-metrics`, which is what makes the harness aggregate them in the first place.

### Sample Upload (Stratified)

Per task, **10 example prompts** are uploaded as W&B tables at `samples/{model_name}/{task_name}`:

- **3 positive samples** (correctly answered, metric = 1.0)
- **7 negative samples** (incorrectly answered, metric = 0.0)

Samples are classified using binary metrics (`acc`, `exact_match`, `em`, `pass@1`). Each sample includes an `is_correct` field (`true`/`false`/`null`) for downstream filtering. If a task has no binary metric (e.g., perplexity), 10 random samples are uploaded instead.

If one group is underrepresented (e.g., a model gets almost everything right), the remaining slots are filled from the other group.

The stratified counts are configurable via `n_positive` and `n_negative` parameters in `create_model_evaluation_from_results()`.

### Retrieving Samples via API

Samples are stored as W&B Tables, retrievable via the W&B API:

```python
import wandb

api = wandb.Api()
run = api.run("entity/project/run_id")

# Get a specific task's samples
table = run.summary["samples/Llama-3.1-8B-Instruct/mmlu"]
```

Each row in the table is a flattened sample dict containing:

| Field | Description |
|-------|-------------|
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
| `LM_EVAL_BACKEND` | `hf` | Backend: `hf` (accelerate), `vllm`, `megatron_lm` |
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
| `NUM_SPLITS` / `SPLIT_INDEX` | `1` / `0` | Task splitting (set automatically by launcher) |
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

`hf_base_runner.sh` handles the submission loop and split-aware job orchestration. It respects `NUM_SPLITS`, `SBATCH_SCRIPT`, and `WANDB_*` environment variables from the launcher.

### Model Registry

See `configs/models.md` for the full list of available models with their HF paths, local checkpoint paths, and required special flags. Key model families:

- **Apertus** (1.0)
- **Meta Llama** (3.1, 3.3)
- **OLMo** (2-1124, 2-0325, 3)
- **Qwen** (2.5, 3)
- **Gemma** (3), **EuroLLM**, **Mistral**, **SmolLM**, **Marin**, and others

---

## Container Setup

The pipeline runs inside containers managed by enroot/pyxis on SLURM. Three container configurations are provided:

| Config | Base Image | Use Case |
|--------|-----------|----------|
| `env.toml` | Based on CSCS container image | Standard HF evals |
| `env_vllm.toml` | Based on custom VLLM 0.16 image, build from source on top of CSCS container image | Standard HF evals |

Dependencies (lm-eval-harness, vLLM, etc.) are installed at runtime inside the container via `pip install`. This ensures the latest versions but adds ~2-3 minutes of startup overhead per job.

---

## Extending the Pipeline

### Adding a New Inference Backend

The sbatch scripts support `hf`, `vllm`, and `megatron_lm` backends. To add a new one:

1. Add a new `elif` block in `evaluate.sbatch` at the `LM_EVAL_BACKEND` dispatch section (~line 181)
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
    n_positive=5,   # number of correct samples to upload (default: 3)
    n_negative=15,  # number of incorrect samples to upload (default: 7)
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

