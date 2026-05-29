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

| Mode | Tasks | Description |
|------|-------|-------------|
| `default` | 40 tasks | Full Apertus benchmark suite with basic evaluation |
| `multi-lingual` | 10 tasks | Apertus benchmark suite with multi-lingual evaluation |
| `pretrain` | 35 tasks | Apertus pretraining benchmark suite |
| `apertus-previous` | 14 tasks | Apertus benchmark suite with multi-lingual evaluation |
| `olmo-easy` | 21 tasks | Base Easy Suite: perplexity/BPB-style evaluation (mmlu, hellaswag, arc, etc.) |
| `olmo-main` | 23 tasks | Base Main Suite: generation + MC (gsm8k_cot, humaneval, drop, etc.) |
| `olmo-heldout` | 2 tasks | Held-out Suite: mmlu_pro, bbh |
| `olmo-safety` | 4 tasks | Safety Suite: harmbench, toxigen, wmdp, bbq |
| `olmo-longcontext` | 1 task | Long-Context: RULER (8192 tokens) |
| `olmo-complete` | 30 tasks | Union of all above (excludes long-context), deduplicated |
| `single` | 1 task | One task, user-specified through `--task` |

Each mode has a corresponding task list (`configs/olmo3_<mode>.txt`) and metric config (`configs/olmo3_<mode>_main_table.txt`). Results are logged to separate W&B projects per mode (e.g., `swissai-evals-olmo3-easy`).

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

### Options

| Flag | Description |
|------|-------------|
| `--name <name>` | Override the auto-derived evaluation run name |
| `--chat-template` | Force enable chat template |
| `--no-chat-template` | Force disable chat template |
| `--tokenizer <path>` | Custom tokenizer (default: same as model) |
| `--num-fewshot N` | Override num_fewshot globally. Tasks with explicit `num_fewshot: 0` in their YAML are never overridden. OLMo3 paper uses 5-shot for most MC tasks. |
| `--backend <hf\|vllm>` | Inference backend (default: from sbatch script) |
| `--splits K` | Split task list across K parallel SLURM nodes per model |


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

# Smoke-test the standalone/sandboxed benchmark path
bash scripts/launch_evaluations.sh single \
  --task smoke_standalone --model meta-llama/Llama-3.1-8B-Instruct
```

#### Deprecated option:

| `--bos` | Prepend BOS token (deprecated: previously for Apertus models, now automatically infered from chat temlate) |

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

## Standalone Benchmarks

Some benchmarks evaluate an agent or artifact inside an external environment rather than a single model completion. They still use the normal launcher task interface: if a task is registered in `configs/standalone/benchmarks/*.toml`, the launcher routes it to `scripts/evaluate_standalone.sbatch`; otherwise it routes it to `scripts/evaluate.sbatch`.

```bash
bash scripts/launch_evaluations.sh single \
  --task smoke_standalone \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --name Llama-Standalone-Smoke
```

Standalone benchmark definitions live in `configs/standalone/benchmarks/*.toml`. They are selected through the normal task-list mechanism: pass a single registered task with `--task`, or pass a task-list file with one task name per line. Each runner emits normalized artifacts:

```
standalone/eval_<timestamp>_<jobid>/
├── run_manifest.json
├── results_<timestamp>.json
├── samples_<benchmark>_<timestamp>.jsonl
└── artifacts/
```

The `results_*.json` file intentionally mirrors the shape produced by `lm-eval`, so `scripts/alignment/update_wandb_alignment.py` can upload static and standalone benchmarks through the same path.

Standalone benchmarks can also be mixed into ordinary task lists. The launcher checks `configs/standalone/benchmarks/*.toml`, partitions registered standalone task names out of `TASKS`, and submits the right backend jobs for the same model. For example:

```text
gsm8k_cot
mmlu
swebench_verified
```

will launch `gsm8k_cot,mmlu` through `scripts/evaluate.sbatch` and `swebench_verified` through `scripts/evaluate_standalone.sbatch`.

On CSCS, standalone jobs enter containers through per-step `srun --environment="$STANDALONE_EDF"` calls. The batch script itself does not use `#SBATCH --environment`, following CE guidance to avoid nested containers and non-host execution surprises. SWE-bench is split into two phases: prediction generation runs inside the EDF, then the official sandbox harness runs on the host compute node where Podman/Docker are available. The default EDF is `./containers/env.toml`; override it with:

```bash
bash scripts/launch_evaluations.sh single \
  --task smoke_standalone \
  --standalone-edf ./containers/env.toml \
  --model my-model
```

Useful standalone options:

| Flag | Description |
|------|-------------|
| `--standalone-edf <path-or-name>` | CSCS CE Environment Definition File used by `srun --environment` |
| `--sandbox-backend <name>` | Runner hint such as `none`, `docker`, `apptainer`, `enroot`, or `remote` |
| `--container-cache-backend <name>` | Container build cache backend, currently `none` or `local_registry` |
| `--local-registry-home <path>` | Optional path to a local-registry checkout containing `env-registry`; omit when `registry` is already on `PATH` |
| `--local-registry-dir <path>` | Per-job local registry data directory |
| `PODMAN_SERVICE_PORT` | Optional fixed localhost port for the per-job Podman API service |
| `PODMAN_SERVICE_HOST` | Optional hostname/IP advertised for the per-job Podman API service |
| `PODMAN_SERVICE_USE_EXTERNAL` | Set to `true` only to reuse an existing TCP `DOCKER_HOST`; by default SWE-bench starts a fresh per-job service |
| `HOST_PYTHON` | Host-side Python used for the SWE-bench harness phase; defaults to `python` and creates a per-job virtualenv |

### SWE-bench Verified

SWE-bench Verified is wired as a standalone benchmark around lm-eval model loading and the official SWE-bench harness. By default the runner loads the requested model through lm-eval, generates an intermediate `predictions.jsonl`, then evaluates those patches in the SWE-bench runtime.

```bash
bash scripts/launch_evaluations.sh single \
  --task swebench_verified \
  --model my-model \
  --name my-model-swebench-verified \
  --sandbox-backend podman \
  --container-cache-backend local_registry
```

`SWE_PREDICTIONS_PATH` is still supported as an override when you want to evaluate pre-generated patches. The predictions file must use the official SWE-bench format:

```json
{"instance_id": "sympy__sympy-20590", "model_name_or_path": "my-model", "model_patch": "diff --git ..."}
```

Useful SWE-bench environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SWE_PREDICTIONS_PATH` | unset | Optional path to `.json`/`.jsonl` predictions, or `gold`; if unset, predictions are generated first |
| `SWE_BENCH_REFERENCE_DIR` | `swe-bench-reference` | Local SWE-bench checkout used only when `SWE_EVALUATOR=official` |
| `SWE_DATASET_NAME` | `princeton-nlp/SWE-bench_Verified` | Dataset passed to the harness |
| `SWE_SPLIT` | `test` | Dataset split |
| `SWE_EVALUATOR` | auto | Evaluator backend. The launcher uses `fast` when a `swe-bench-fast` binary or checkout is available, otherwise `official` |
| `SWE_BENCH_FAST_BIN` | auto | Binary path used when `SWE_EVALUATOR=fast`; auto-detects `dist/` and `bin/` under `swe-bench-fast-main`, `swe-bench-fast`, `external/swe-bench-fast`, then `PATH` |
| `SWE_BENCH_FAST_SOURCE_DIR` | auto | Optional `swe-bench-fast` source checkout; auto-detects `swe-bench-fast-main`, `swe-bench-fast`, then `external/swe-bench-fast` |
| `SWE_BENCH_FAST_AUTO_BUILD` | `true` | Build `swe-bench-fast` with `make build` when the source checkout exists but the binary does not |
| `SWE_BENCH_FAST_ARM64_REGISTRY` | `docker.io/greynewell/swe-bench-arm64` | ARM64 image registry for `swe-bench-fast`; tags use sanitized instance IDs such as `sympy-sympy-22005` |
| `SWE_INSTANCE_IDS` | unset | Space- or comma-separated subset for smoke runs |
| `SWE_MAX_WORKERS` | `75% of SLURM_CPUS_PER_TASK, else 4` | Harness worker count for image builds and test containers |
| `SWE_TIMEOUT` | `1800` | Per-instance timeout in seconds |
| `SWE_CACHE_LEVEL` | `env` | Official harness cache level |
| `SWE_NAMESPACE` | `none` | Image namespace. `none` builds images locally; set `swebench` only when remote prebuilt image pulls are desired and authenticated/mirrored |
| `SWE_ARCH` | host architecture | Container image architecture, normalized to `arm64` on AArch64 hosts and `x86_64` on x86 hosts |
| `SWE_RELAX_CONDA_BUILDS` | `auto` | On ARM, strip architecture-specific conda build strings from SWE-bench environment files while keeping version pins |
| `SWE_RELAX_CONDA_PACKAGE_PINS` | `setuptools pip python` | On ARM, strip version pins for selected conda packages that are commonly unavailable on `linux-aarch64` |
| `SWE_USE_PODMAN_BUILD` | `true` | Build SWE-bench images with host `podman build` instead of the Docker API build endpoint |
| `SWE_PODMAN_BUILD_STORAGE_OPTS` | `ignore_chown_errors=true` | Storage options passed to `podman build`; useful on rootless CSCS nodes without broad subuid/subgid mappings |
| `PODMAN_SERVICE_STORAGE_OPTS` | `ignore_chown_errors=true` | Storage options used when starting the per-job Podman Docker-compatible API service |
| `SWE_USE_PODMAN_CACHED` | set by local registry mode | Tell the local `swe-bench-reference` harness to use `podman-cached` for image builds |
| `LM_EVAL_BACKEND` | `vllm` | lm-eval backend used for patch generation |
| `LM_EVAL_MODEL_ARGS` | launcher-built | lm-eval model args used for patch generation |
| `BS` | `auto:20` | lm-eval batch size used for patch generation |
| `MAX_NEW_TOKENS` | `2048` | Token budget for patch generation |
| `APPLY_CHAT_TEMPLATE` | launcher-derived | Whether to apply the model chat template |

When `--container-cache-backend local_registry` is used, the standalone sbatch script starts `registry` on the allocated node, exports `LOCAL_REGISTRY`, and stops it on exit. If `registry` is not already on `PATH`, pass `--local-registry-home /path/to/local-registry` so the script can source `env-registry`. This caches image build layers for the duration of the Slurm job. The local `swe-bench-reference` checkout has been adapted to honor `SWE_USE_PODMAN_CACHED=true` in its image build function; container execution still uses the Docker-compatible API used by the official harness.

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
| `MAX_NEW_TOKENS` | `512` | Maximum generated tokens |
| `LIMIT` | (unset) | Limit number of samples per task |
| `NUM_FEWSHOT` | (unset) | Global few-shot override |
| `NUM_SPLITS` / `SPLIT_INDEX` | `1` / `0` | Task splitting (set automatically by launcher) |
| `LOGS_ROOT` | `/capstor/.../eval-logs` | Root directory for evaluation logs |
| `WANDB_ENTITY` | `apertus` | W&B entity |
| `WANDB_PROJECT` | `swissai-evals-test` | W&B project |

The script auto-detects RULER long-context tasks and adjusts `MAX_LENGTH` and `max_model_len` accordingly.

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
