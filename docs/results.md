# Results: W&B, exports and tables

Where results go after a run: W&B (all four frameworks share one upload path in `scripts/results/`), Every Eval Ever / Hugging Face exports, and the result tables.

## W&B Integration

### Metrics Upload

Results are automatically uploaded to W&B after evaluation completes (or after aggregation for chunked jobs). Each model gets a W&B run with:

- **`main_results`** table: summary metrics specified in the `*_main_table.txt` config
- **Flat metrics**: all task metrics logged as `task_name/metric_name`
- **`eval_duration`**: evaluation-only runtime for a direct/fail-fast evaluation; chunk aggregation sums the runtimes recorded by completed chunks (an explicit `EVAL_DURATION` overrides this for legacy recovery)

Because *every* flat metric is uploaded — not just the `*_main_table.txt` subset — the
[thinking metrics](lm-eval-harness.md#thinking--reasoning-metrics) reach W&B as `task_name/thinking_format_correct`
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
python -m scripts.results.update_wandb_alignment \
  --entity apertus --project swissai-evals \
  --name Llama-3.1-8B-Instruct \
  --logs_root /path/to/harness/eval_20250726_003542_12345 \
  --main_metrics mmlu/acc arc_challenge/acc_norm gsm8k_cot/exact_match \
  --eval_duration 3600

# Batch upload all models from a logs directory
python -m scripts.results.update_wandb_all_models \
  --entity apertus --project swissai-evals \
  --logs_root /path/to/eval-logs
```

Inspect AI, Harbor and EvalScope runs upload when their script gets `--wandb-entity <e> --wandb-project <p>`. To upload a finished run's work dir afterwards, run the framework's module from that framework's environment: `scripts.results.update_wandb_inspect`, `update_wandb_harbor` or `update_wandb_evalscope` (see [inspect-ai.md](inspect-ai.md), [harbor.md](harbor.md), [evalscope.md](evalscope.md)).

---

## Customizing Sample Upload

The stratified sample selection in `scripts/results/wandb_alignment_utils.py` can be adjusted:

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
python scripts/results/export_eval_results.py export \
  /path/to/merged_eval/results_2026-07-27T12-00-00.json \
  --output-dir eval-results/Apertus-release-2026-07 \
  --model-id swiss-ai/Apertus-8B-Instruct-2509 \
  --model-name Apertus-v1-8b \
  --include-samples
```

For a compact, reproducible datastore submission, use a per-task instance cap
and one collection directory. This preserves every aggregate score and keeps
up to 10 scorable examples from each lm-eval task, selected by a stable SHA-256
priority rather than by file order. The cap does not rescore or recompute
aggregates. Rows without a numeric per-sample value for an exported metric
(such as deferred LLM-judge payloads) are omitted instead of being reported as
zero. The manifest records capped, unscored, and missing-sample tasks; each
record retains source, scorable, and exported instance-row counts.

```bash
python scripts/results/export_eval_results.py export RESULTS.json \
  --output-dir eval-results/Apertus-release-capped \
  --model-id swiss-ai/Apertus-v1.5-8B \
  --include-samples \
  --max-samples-per-task 10 \
  --datastore-collection swissai_apertus_evals
```

The resulting EEE files can be copied directly from
`eee/data/swissai_apertus_evals/<developer>/<model>/` into the same datastore
path. Aggregate JSON filenames remain stable; only the record paths and the
optional instance companions differ from a benchmark-folder export. Use a
fresh output directory when changing sample-cap settings.

`--model-id` is optional when the result log contains an `owner/model` ID, but
it is required when the evaluation used a local checkpoint path.
EEE `model_info.id` uses this Hub ID, and `model_info.additional_details`
includes its direct Hugging Face model URL for self-deployed models. API models
do not receive a guessed Hub URL. Use `--model-name` when the release/display
name differs from the repository name; it is written to `model_info.name`
without changing the ID or URL.
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
python scripts/results/export_eval_results.py validate \
  eval-results/Apertus-release-2026-07

# Verify EEE collection names and Hugging Face benchmark task IDs remotely.
python scripts/results/export_eval_results.py check-mappings
```

The export manifest (`manifest.yaml`, containing JSON-compatible YAML) lists:

- generated EEE records and HF previews
- lm-eval tasks exported with internal names because they have no reviewed mapping
- `_self_consistency` task names resolved to a reviewed base-task mapping
- missing run-level metadata such as generation temperature or maximum tokens
- `publishing_performed: false`, confirming that the exporter only wrote local files

The YAML extension is intentional: the official EEE validator recursively
treats every `.json` file as an aggregate `EvaluationLog`. Keeping exporter
bookkeeping out of `.json` means the complete output directory can be passed
directly to `every_eval_ever validate`. The local validator still accepts
legacy exports containing `manifest.json`, and re-exporting migrates that file
to `manifest.yaml`.

Every task with numeric results is exported. Use `--strict-mappings` when every
numeric task in a run must have a reviewed mapping instead of using the fallback
described below.

By default, aggregate and non-aggregate rows are all retained. Omit
`--aggregate-only-task` when preparing a full-data submission. The option below
is available only for intentionally compact exports.

Use `--exclude-task TASK` to omit an exact result. For benchmark groups with
many language or subject results, prefer `--aggregate-only-task GROUP`: it keeps
the group's aggregate score and recursively excludes its subtasks. The exporter
uses lm-eval's `group_subtasks` metadata when available and falls back to the
generated suite's task-name prefix when lm-eval omits that metadata. Both
options are repeatable. Generated suites can be selected by either their base
group name or their full aggregate result task. For example:

```bash
python scripts/results/export_eval_results.py export RESULTS.json \
  --output-dir eval-results/Apertus-release \
  --aggregate-only-task include_base_44 \
  --aggregate-only-task include_base_new_45 \
  --aggregate-only-task global_mmlu
```

The manifest records the effective choices in `aggregate_only_tasks` and
`excluded_tasks`.

### Task mappings

Mappings live in `configs/eval_export/task_mappings.json`. Each entry maps an
exact lm-eval task name to:

- the canonical EEE datastore collection directory
- the optional composite, benchmark family, benchmark, and split used to build
  the EEE `evaluation_name`
- an optional `collection` when the datastore folder must retain a historical
  name that differs from the benchmark component in `evaluation_name`
- the source dataset ID in Hugging Face `owner/dataset` format
- optional ordered EEE metric candidates and canonical metric-ID overrides
- optional reusable `llm_scoring` metadata for judge-derived metrics, including
  judge model IDs, temperatures, exact prompt templates, pairwise baselines,
  and Arena-Hard style-control state
- optionally, a registered Hugging Face benchmark dataset, task ID, and ordered
  metric candidates
- optionally, a reviewed subtask pattern that places language/subject parts in
  the split component while removing run setup from the evaluation name

Mappings are resolvers, not an export allowlist. An exact mapping wins; reviewed
subtask patterns and the controlled `_self_consistency` suffix may resolve to a
base mapping. For unmapped tasks, the exporter detects a scored configless
aggregate prefix and uses it as the internal family/benchmark, putting the
remaining task name in the split. Truly standalone tasks retain their exact
internal identifier as family and benchmark with `overall` as the split. Logged
Hugging Face dataset IDs are preserved, including for configless aggregates
when all related children share one unambiguous ID. The manifest lists fallback
tasks in `internally_named_tasks`; `skipped_unmapped_tasks` remains an empty
compatibility field.

Mappings are resolvers, not an export allowlist. An exact mapping wins; the
controlled `_self_consistency` suffix may resolve to its reviewed base mapping.
Any other scored task is exported under its lm-eval identifier, with that
identifier used as both family and benchmark and `overall` as the split. For
example, an unmapped `aime24` result becomes `aime24.aime24.overall`. Its logged
`dataset_path` is still used as the dataset ID when present. The manifest lists
these fallbacks in `internally_named_tasks`; `skipped_unmapped_tasks` remains an
empty compatibility field.

Evaluation names use dot notation:
`{composite}.{family}.{benchmark}.{split}`. An empty composite is omitted, so a
standalone benchmark such as GSM8K exports as `gsm8k.gsm8k.overall`, without a
leading dot. The split defaults to `overall`; lm-eval filters such as
`strict-match` or `mean@32` describe scoring methodology and are retained in
metric and generation metadata rather than being treated as benchmark splits.

For Hugging Face datasets, both `source_data.dataset_name` and
`source_data.hf_repo` contain the `owner/dataset` ID. The direct dataset URL is
recorded in `source_data.additional_details.hf_dataset_url`.

Benchmark-specific metric IDs use dot namespaces (for example,
`mmlu_pro.overall` and `bbq.amb_bias_score.age`). Task hashes are scoped to the
record containing the corresponding lm-eval task instead of copying the run's
entire hash map into every record.

The repository's `_self_consistency` suffix is handled as a controlled task
variant: the exporter removes only that exact suffix and requires the remaining
base task to have a reviewed mapping. Its mapping can define
`self_consistency_metric_candidates`; `{repeats}` is expanded from the lm-eval
task config so, for example, `exact_match,mean@{repeats}` selects
`exact_match,mean@32`. The original task name, repeat count, and
`self_consistency` variant remain recorded in the exported metadata.

Do not otherwise use fuzzy matching for benchmark variants. For example,
`gsm8k_platinum` is not mapped to `gsm8k`. Until a mapping is reviewed, tasks
such as `bfcl_v3` and `swiss_ai_charter_alignment` are exported under safe
internal names rather than being omitted or assigned to a possibly incorrect
external benchmark.

The exporter preserves scores in their native lm-eval scale. It never
automatically multiplies proportions by 100.

The `polyglotoxicitypromptsllama_small.score` and
`realtoxicitypromptsllama_small.score` metrics are unsafe-continuation rates,
not continuous toxicity probabilities. The harness generates one continuation
per prompt, asks `meta-llama/Llama-Guard-4-12B` to classify the prompt and
continuation, maps `unsafe` to 1 and `safe` to 0, drops failed or unknown judge
responses, and averages the remaining labels. The multilingual Polyglot
Toxicity Prompts aggregate is the sample-size-weighted mean of its seven
language-task scores.

---

## Reporting: Building Result Tables

Both table scripts must be run manually. They read every score from **W&B run summaries** (not from disk); the metrics file
selects *which* `task/metric` keys to pull and how to group them.

### `scripts/results/make_html_table.py` — interactive HTML results table

A self-contained, collapsible HTML table: one column per model, benchmarks grouped by the `#`
headers in the metrics file.

```bash
python scripts/results/make_html_table.py \
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
  [Building a thinking-only table](lm-eval-harness.md#building-a-thinking-only-table).
- `--debug` prints the fetched W&B summary keys per run (useful when a cell is blank).

### `scripts/results/make_table.py` — hyperparameter-sweep table (PNG + CSV)

Targets sweep comparisons rather than a leaderboard: it enumerates the run sub-directories
under `--base-model-path`, parses each name for `beta` / batch-size / `lr` / length-norm, fetches
its W&B summary, and renders **delta improvements over a baseline** (`Avg_Imp`; or absolute scores
with `--absolute`). Writes `eval_table.png` **and** `eval_table.csv` into the `--output` directory.

```bash
python scripts/results/make_table.py \
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
