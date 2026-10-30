# Inspect AI and Artificial Analysis benchmarks

Benchmarks that need their own scorer, a judge or a simulated user: Inspect AI tasks (`inspect_evals` and `scripts/aaii/`), and the standalone Artificial Analysis protocol runners in `scripts/aaii/`.

## Inspect AI evals

[Inspect AI](https://inspect.aisi.org.uk/) (UK AISI's eval framework) and its [`inspect_evals`](https://github.com/UKGovernmentBEIS/inspect_evals) collection give access to benchmarks not covered by the lm-evaluation-harness suites above (e.g. tau2-bench, an agentic tool-use benchmark). Inspect's execution and logging model (task registry, model roles, `.eval` log files) differs enough from lm-eval-harness's that these run through their own script, `scripts/inspect_ai/run_inspect_eval.sh`, instead of `launch_evaluations.sh`.

Results are uploaded to W&B whenever a W&B key is available (see [W&B upload](#wb-upload)); most examples below are smoke tests (`--limit`), so add `--no-wandb` to keep them out of the shared project.

Run it directly (e.g. on a login node) or, like the rest of this pipeline, as a SLURM job via `scripts/inspect_ai/run_inspect_eval.sbatch`, which forwards all arguments to `run_inspect_eval.sh` inside the same container as the other backends -- but requests CPU-only resources, since these benchmarks evaluate an already-running served endpoint rather than loading a model in-job:

```bash
# --reservation is a native sbatch flag (run_inspect_eval.sbatch has no --reservation of its
# own -- everything after the script path is forwarded verbatim to run_inspect_eval.sh)
sbatch --reservation=my-reservation scripts/inspect_ai/run_inspect_eval.sbatch --task tau2_retail \
  --model CSCS-Inference/swiss-ai/Apertus-v1.5-8B --api-base-url https://api.swissai.svc.cscs.ch/v1 \
  --model-role user=openai-api/swissai/CSCS-Inference/swiss-ai/Apertus-v1.5-70B \
  --task-arg message_limit=10 --limit 5
```

```bash
# run_inspect_eval.sh installs these itself at runtime (SKIP_INSTALL=1 or EVAL_PREBUILT_ENV=1 to skip);
# to install by hand for local/interactive use:
pip install "inspect-ai>=0.3.258" inspect-evals openai

# A plain benchmark against a served model (CSCS serving, vllm serve, ...)
scripts/inspect_ai/run_inspect_eval.sh --task gsm8k --model Qwen/Qwen3-8B --api-base-url http://nid001234:8000

# GPQA Diamond (inspect_evals/gpqa_diamond -- graduate-level multiple choice, no judge
# needed). Runs 4 epochs per sample by default (majority vote over repeated attempts); pass
# `-- --epochs 1` after the task flags to disable that:
scripts/inspect_ai/run_inspect_eval.sh --task gpqa_diamond \
  --model CSCS-Inference/swiss-ai/Apertus-v1.5-8B --api-base-url https://api.swissai.svc.cscs.ch/v1 \
  --limit 5

# HLE (Humanity's Last Exam, inspect_evals/hle) is graded by two judges side by side (task
# arg `graders`, default `[grader, grader_2]`), so it needs both a "grader" and a "grader_2"
# role bound, same convention as tau2's "user" role / AA-Omniscience's "grader" role above --
# otherwise it falls back to run_configs/default.yaml's OpenRouter judges and fails without
# OPENROUTER_API_KEY. The dataset is ~2,500 questions with multi-modal (image) samples
# included by default; keep --limit small for a smoke test.
#
# IMPORTANT if the grader role is a CSCS-served/vLLM model (as below): the default judge_prompt
# ("original") requires the grader to return OpenAI-style strict JSON-schema structured output
# (ResponseSchema(strict=True)). Verified against this exact command: the CSCS-served
# Apertus-v1.5-8B backend accepts that request but never returns a response -- the run hangs
# indefinitely (no error, no progress) rather than failing fast; killing and retrying with
# `--task-arg judge_prompt=grade_c_i` (the legacy plain-text GRADE:C/I judge, no structured
# output) completed the same run in ~1s. A real OpenAI grader (e.g. --model-role
# grader=openai/gpt-5.6-luna, per Artificial Analysis's own protocol) does not have this
# problem -- this only bites when grading with a served vLLM model that doesn't support
# structured outputs:
scripts/inspect_ai/run_inspect_eval.sh --task hle \
  --model CSCS-Inference/swiss-ai/Apertus-v1.5-8B --api-base-url https://api.swissai.svc.cscs.ch/v1 \
  --model-role grader=openai-api/swissai/CSCS-Inference/swiss-ai/Apertus-v1.5-8B \
  --model-role grader_2=openai-api/swissai/CSCS-Inference/swiss-ai/Apertus-v1.5-8B \
  --task-arg judge_prompt=grade_c_i --limit 5

# AAII-wrapped GPQA Diamond (scripts/aaii/gpqa_diamond.py -- a full path, not an inspect_evals name, so
# it's used as-is rather than expanded to "inspect_evals/..."), pinning Artificial Analysis's
# Intelligence Index protocol default of --epochs 5 (plain "gpqa_diamond" above keeps
# inspect_evals' own default of 4) directly in the task, not via a CLI flag -- no --model-role
# needed, same as plain gpqa_diamond:
scripts/inspect_ai/run_inspect_eval.sh --task scripts/aaii/gpqa_diamond.py \
  --model CSCS-Inference/swiss-ai/Apertus-v1.5-8B --api-base-url https://api.swissai.svc.cscs.ch/v1 \
  --limit 5

# AAII-wrapped HLE (scripts/aaii/hle.py, same "full path" convention as scripts/aaii/gpqa_diamond.py above),
# pinning the text-only subset, the official judge prompt, and a single "grader" role (AA's
# protocol) directly in the task -- unlike plain hle above, no grader_2 role is needed. Still
# needs its own "grader" role bound (same judge_prompt=original caveat re: structured-output
# support applies -- see the plain hle example above for why a served vLLM model may hang):
scripts/inspect_ai/run_inspect_eval.sh --task scripts/aaii/hle.py \
  --model CSCS-Inference/swiss-ai/Apertus-v1.5-8B --api-base-url https://api.swissai.svc.cscs.ch/v1 \
  --model-role grader=openai/gpt-5.6-luna --limit 5

# AAII-wrapped SciCode (scripts/aaii/scicode.py, same "full path" convention), pinning
# provide_scientific_background=True and --epochs 3 (AA's protocol; plain scicode below keeps
# inspect_evals' own background-free, --epochs 1 defaults) directly in the task. Needs a real
# Docker daemon for its sandboxed code execution (true on a login node or inside the sbatch
# container) -- unlike evals-svc's own AAII-wrapped scicode, which additionally pins
# --sandbox local as a deployment-specific workaround for launchers with no Docker daemon; that
# pin is NOT part of the AA protocol, so it is deliberately not baked in here:
scripts/inspect_ai/run_inspect_eval.sh --task scripts/aaii/scicode.py \
  --model CSCS-Inference/swiss-ai/Apertus-v1.5-8B --api-base-url https://api.swissai.svc.cscs.ch/v1 \
  --limit 2

# tau2-bench (no single "default" task -- it ships four domains: airline, banking, retail,
# telecom) needs a second "user"-role model to play the customer, and supports extra task
# params like message_limit or banking's retrieval_config. The "user" role doesn't need to be
# a real OpenAI model -- any served model works, so this stays entirely on the CSCS platform
# (verified working: two always-on models served there, no extra API key needed):
scripts/inspect_ai/run_inspect_eval.sh --task tau2_retail,tau2_banking \
  --model CSCS-Inference/swiss-ai/Apertus-v1.5-8B --api-base-url https://api.swissai.svc.cscs.ch/v1 \
  --model-role user=openai-api/swissai/CSCS-Inference/swiss-ai/Apertus-v1.5-70B \
  --task-arg message_limit=10 --limit 5

# A model reached through Inspect's own provider (no --api-base-url), running two tasks
# together via `inspect eval-set`, with extra flags forwarded after --
scripts/inspect_ai/run_inspect_eval.sh --task gsm8k,gaia --model anthropic/claude-3-5-sonnet-latest \
  --eval-set -- --temperature 0.5 --max-connections 10

# GDPval (inspect_evals/gdpval -- 220 economically-valuable, real-world
# deliverable tasks across 44 occupations). Runs inside a Docker sandbox whose
# image build can take up to 10 minutes on first use (see the task's own
# README for details); scoring needs a separate submission to OpenAI's
# grading form, so this is mainly useful as a smoke test that generation
# completes -- keep --limit small. Expect a harmless "Exception calling hook
# 'ConsolidateDeliverables': cannot import name 'HfFolder' from
# 'huggingface_hub'" warning on every run: inspect_evals' own gdpval/util.py
# still calls the HfFolder API huggingface_hub removed in 1.0 (confirmed live
# on their main branch, and already flagged by the maintainers themselves in
# github.com/UKGovernmentBEIS/inspect_evals/pull/1211 as unaddressed) -- the
# eval still completes and the deliverable folder is still written locally,
# only that hook's own HF-auth check fails:
scripts/inspect_ai/run_inspect_eval.sh --task gdpval \
  --model CSCS-Inference/swiss-ai/Apertus-v1.5-8B --api-base-url https://api.swissai.svc.cscs.ch/v1 \
  --limit 1

# AA-Omniscience (scripts/aaii/aa_omniscience.py, formerly custom_tasks/omniscience.py -- a full path,
# not an inspect_evals name, so it's used as-is rather than expanded to "inspect_evals/...";
# see the module docstring there for why it's a from-scratch task) needs a "grader" role for
# its judge model, same convention as tau2's "user" role above. Artificial Analysis's own
# protocol is a fixed external judge (GPT-5.6 Luna) -- the model under test grading itself
# works too (verified below) but is not AAII-comparable:
scripts/inspect_ai/run_inspect_eval.sh --task scripts/aaii/aa_omniscience.py \
  --model CSCS-Inference/swiss-ai/Apertus-v1.5-8B --api-base-url https://api.swissai.svc.cscs.ch/v1 \
  --model-role grader=openai-api/swissai/CSCS-Inference/swiss-ai/Apertus-v1.5-8B --limit 5

# AA-LCR (scripts/aaii/aa_lcr.py, formerly custom_tasks/aa_lcr.py, same "full path" / "grader" role
# convention as AA-Omniscience above). Unlike every other task in this repo, each sample is a
# ~100k-token prompt (the full Document Set) -- keep --limit at 1 for a smoke test unless you
# mean to spend real time/money on a full 100-question run:
scripts/inspect_ai/run_inspect_eval.sh --task scripts/aaii/aa_lcr.py \
  --model CSCS-Inference/swiss-ai/Apertus-v1.5-8B --api-base-url https://api.swissai.svc.cscs.ch/v1 \
  --model-role grader=openai-api/swissai/CSCS-Inference/swiss-ai/Apertus-v1.5-8B --limit 1

# CritPt (scripts/aaii/aa_critpt.py, a research-level physics reasoning benchmark -- see the module
# docstring for the two-step reasoning-then-parse protocol, and why grading happens in an
# Inspect Hook rather than inline). No --model-role grader needed, unlike AA-Omniscience/AA-LCR
# above: grading is a fixed external HTTP server (Artificial Analysis's own
# /api/v2/critpt/evaluate), not an LLM judge. Set CRITPT_API_KEY (access is granted case by
# case -- email critpt@artificialanalysis.ai) to actually submit for grading; without it the
# run still completes normally and just writes each epoch's 70-submission batch to a local
# JSON file instead (under <logs-dir>/critpt_submissions/). --epochs is a NATIVE `inspect eval`
# flag (not run_inspect_eval.sh's own), so it goes behind a literal `--`, same as GPQA Diamond
# above; the real protocol runs 5, but pass `-- --epochs 1` for a smoke test:
scripts/inspect_ai/run_inspect_eval.sh --task scripts/aaii/aa_critpt.py \
  --model CSCS-Inference/swiss-ai/Apertus-v1.5-8B --api-base-url https://api.swissai.svc.cscs.ch/v1 \
  --limit 2 -- --epochs 1
```

The model under test is either passed straight through as an Inspect-native model string, or -- when `--api-base-url` is given -- wrapped through Inspect's generic `openai-api` provider (the same OpenAI-compatible endpoints this pipeline already evaluates against with `--backend openai`). Model roles (`--model-role role=model`, repeatable) and extra task parameters (`--task-arg key=value`, repeatable) cover benchmark-specific needs like tau2's user-simulator or an LLM-as-judge grader; each role's own provider credentials (e.g. `OPENAI_API_KEY`) are your responsibility. See `scripts/inspect_ai/run_inspect_eval.sh --help` for all options.

Results are **not** viewed at inspect.aisi.org.uk — that's Inspect's documentation site. Logs land as `.eval` files under `logs/inspect/<name>/`; view them with `inspect view --log-dir logs/inspect/<name>`.

#### W&B upload

Like every framework here, it uploads whenever a W&B key is available (`WANDB_API_KEY` or `scripts/wandb_api_key.txt`), by default to `apertus/apertus-1.5-post-training-v0.1`; `--wandb-entity`/`--wandb-project` change that, and `--no-wandb` (or `WANDB_MODE=disabled`) skips the upload, e.g. for a smoke test.

```bash
scripts/inspect_ai/run_inspect_eval.sh --task tau2_retail --model Qwen/Qwen3-8B \
  --api-base-url http://nid001234:8000 --wandb-project swissai-evals-test
```

When uploading, the script waits for `inspect eval`/`eval-set` to finish, diffs `--logs-dir` against its contents from before the run to find the `.eval` log(s) this run produced, and uploads them with:

```bash
python -m scripts.results.update_wandb_inspect --entity <entity> --project <project> \
  --name <name> --eval-log <path-to-run.eval> [--eval-log <path> ...]
```

`update_wandb_inspect.py` parses the `.eval` log(s) (task scores, metrics, and a bounded per-sample summary) into the same `ModelEvaluation` structure the harness pipeline uses, and uploads through the same shared `upload_multi_model_results` — so Inspect and lm-eval-harness runs show up in W&B the same way. Auth uses `WANDB_API_KEY` (default: `scripts/wandb_api_key.txt`, same fallback as `evaluate.sbatch`).

---

## Standalone Artificial Analysis runners: tau2-bench / terminal-bench / GDPval-AA v2

Three more Artificial-Analysis-aligned benchmarks, each its own standalone script rather than an Inspect AI task — none of these three go through `run_inspect_eval.sh` or `inspect_evals` at all. This repo already documents *different*, narrower `inspect_evals` ports of tau2-bench and GDPval above (in the Inspect AI section) — those are independent implementations, not the same code as the scripts below, and don't necessarily match Artificial Analysis's real protocol values. There is no `inspect_evals` port of Terminal-Bench at all; the [Harbor runner](harbor.md) runs the same protocol too (`--benchmark terminal_bench_2_1 --attempts 3`).

Each script installs its own dependencies at runtime (tau2-bench/Harbor/Stirrup), takes `--model`/`--api-base-url` the same way `run_inspect_eval.sh` does (falling back to `scripts/cscs_serving_api_key.txt` for the serving key), bakes in Artificial Analysis's protocol defaults, and writes results to a local scratch dir instead of posting to a callback (unlike evals-svc's own copies of this same logic, which these scripts were extracted from). See each script's own `--help` for full options.

```bash
# tau2-bench (tau2-bench v1.0.1, banking_knowledge domain, bm25_grep retrieval, 5 trials, 200
# max steps -- AA's tau^3-Banking protocol). Needs a "user simulator" model too (default:
# gpt-5.4-mini, a real OpenAI model, needs OPENAI_API_KEY of your own); pass --user-llm
# openai/<served-id> instead to route it through the same gateway/key as --model, no separate
# credential needed -- smoke test with a couple of tasks/one trial:
scripts/aaii/run_tau2_bench.sh --model CSCS-Inference/swiss-ai/Apertus-v1.5-8B \
  --api-base-url https://api.swissai.svc.cscs.ch/v1 \
  --user-llm openai/CSCS-Inference/swiss-ai/Apertus-v1.5-8B --num-tasks 2 --num-trials 1

# terminal-bench (Harbor, Terminus 2 agent, terminal-bench-2-1 dataset, 3 trials). Needs a real
# Docker daemon or podman (the script falls back to podman through a docker-compatible shim,
# same as evals-svc's own runner does on Clariden) -- smoke test with one task/one trial:
scripts/aaii/run_terminal_bench.sh --model CSCS-Inference/swiss-ai/Apertus-v1.5-8B \
  --api-base-url https://api.swissai.svc.cscs.ch/v1 --num-tasks 1 --num-trials 1

# GDPval-AA v2 (Stirrup, the full 220-task openai/gdpval gold set, up to 250 turns, E2B sandbox
# by default -- needs E2B_API_KEY; pass --sandbox-backend local for the free, non-isolated
# subprocess backend instead). Grading needs at least one of
# GDPVAL_JUDGE_{OPENAI,GOOGLE,ANTHROPIC}_API_KEY set -- ungraded deliverables are still
# written. Smoke test, free sandbox, no grading:
scripts/aaii/run_gdpval.sh --model CSCS-Inference/swiss-ai/Apertus-v1.5-8B \
  --api-base-url https://api.swissai.svc.cscs.ch/v1 --sandbox-backend local --num-tasks 1
```
