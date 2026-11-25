# Custom BFCL runner

`scripts/custom/run_bfcl.sh` runs BFCL (the Berkeley Function Call Leaderboard) with its official package, [`bfcl-eval`](https://pypi.org/project/bfcl-eval/): the package's generation (the model's requests and the multi-turn loop with its simulated tool environments), its checkers and its score aggregation. No framework of this repository sits in between, so the score is the one bfcl-eval computes. The driver is `scripts/custom/bfcl.py`; the script runs it in the venv of `requirements/bfcl-runtime.lock` and `requirements/bfcl-eval.txt`: baked into the prebuilt image (`BFCL_VENV`, nothing is installed at run time), elsewhere built into a cache ([locked_venv](../scripts/common/locked_venv.sh)).

```bash
scripts/custom/run_bfcl.sh --version v3 --model <served name> --api-base-url https://api.swissai.svc.cscs.ch/v1 \
  --out-dir logs/bfcl/<name>-v3 --reasoning-effort high
```

## What runs

| `--version` | Categories | Entries |
|---|---|---|
| `v3` | non-live (simple Python/Java/JavaScript, multiple, parallel, parallel multiple, irrelevance), live (simple, multiple, parallel, parallel multiple, irrelevance, relevance), multi-turn (base, miss func, miss param, long context) | 4441 |
| `v4` | those and the memory categories (kv, vector, recursive summarization) | 4906 |

BFCL v4's web search categories need a SerpAPI key and are not run (`--test-category web_search_base` adds them). `--test-category` replaces the categories (bfcl-eval's names or groups, repeatable). The model is called in function-calling mode through bfcl-eval's OpenAI chat-completions handler, registered at run time as `<name>-FC`.

## Settings

bfcl-eval sends a temperature of 0.001 and nothing else. The sampling flags (`--temperature`, `--top-p`, `--top-k`, `--min-p`, `--repetition-penalty`, `--sampling-profile`) and the thinking flags (`--enable-thinking`, `--reasoning-effort`) are added to every request ([sampling](sampling.md)); without them the run is bfcl-eval's own default. `--num-threads` (default 32) is the number of parallel requests, `--request-timeout` (default 3600 s) the time one request may take: the OpenAI SDK's own 600 s cuts off a few long thinking answers per run. The driver patches two things in the package's handler and nothing else: it passes those settings through, and it keeps the null fields of a vLLM reply out of the assistant turns it replays (they make vLLM answer HTTP 400 and end the multi-turn conversation).

## Results

`<workdir>/summary.json` also holds `inference_errors`: how many generated entries ended in an inference error (bfcl-eval writes "Error during inference: ..." as their result and scores them wrong, inside the category's total), by cause: `context_overflow` is the conversation outgrowing the served window, a limit of the model under test; `other` (a timeout, a server error) is not necessarily the model's, and the run prints a warning when there is any, so it can be rerun. The two counts are uploaded as `n_inference_errors` and `n_context_overflows`. The GLM-4.5-Air and Nemotron 3 Super runs of 2026-11-23 had 1 (a context overflow, GLM v4) in 18,916 entries.

`<workdir>/summary.json` holds the accuracy, correct count and total of every category, the groups bfcl-eval computes (non-live, live, multi-turn, agentic: each its own aggregation, see `bfcl_eval/eval_checker/eval_runner_helper.py`) and `overall`. A group is only reported when all its categories were run, because bfcl-eval counts a category it was not asked to run as 0 in its tables: without the web search categories there is no agentic group (the memory categories are in the per-category scores). For `v3` that is the mean of the live, non-live and multi-turn groups, as the v3-era package (`bfcl-eval 3.0.0`) computed it. For `v4` there is no official overall: the package's weights (10/10/10/30/40 over non-live, live, irrelevance, multi-turn, agentic) need the agentic group, whose web search half is not run. `summary.json` therefore also holds `v4_without_web_search`: the package's own formula (BFCL's V4 changelog and `eval_runner_helper.py`: non-live and live without their irrelevance and relevance categories, irrelevance as the mean of the two irrelevance categories, multi-turn and memory as plain means) with the agentic group replaced by the memory score alone, the web search categories left out. `overall_no_web_search` is that weighted sum. It is **not the leaderboard's number**, and not comparable to it: it measures a different mix, and bfcl-eval's own overall would count the missing web search half as 0. bfcl-eval's own tables are `<workdir>/score/data_*.csv`, the model's answers `<workdir>/result/`.

**Traces.** `<workdir>/result/` holds every entry's full trace: the model's output (the multi-turn turns, tool calls and their results) and, in each entry's `inference_log`, the fully formatted input of every model call (the whole history and the tool definitions), because the run passes `include_input_log` to bfcl-eval. That makes the result files much larger than bfcl-eval's default: about ten times by an estimate from the recorded input tokens (one multi-turn file, 200 entries and 2,795 model calls: 11 MB without the input log, roughly 120 to 170 MB with it), so a run needs a few hundred MB. It has not been measured on a real run. None of it is uploaded to W&B.

Rerunning with the same `--out-dir` resumes: bfcl-eval skips the entries it already has. `--limit n` runs the first n entries of each category (a smoke test); its scores cover that subset and are not the leaderboard's.

## Samples that fail in inference

A sample whose generation fails is scored wrong and stays in the denominator (bfcl-eval records it as `inference_error`), so these lower the score and are worth counting after a run. In two full Chonk runs (v3 and v4, 2026-11-20; ~4400 samples each, made with an earlier setup) about 47 per run failed this way:

- **Prompt over the context window** (43-44): `multi_turn_long_context` prompts longer than the 32768-token window get HTTP 400. The runner cannot fix this; a longer window would.
- **Request timeouts** (3): long thinking answers outlasted the OpenAI SDK's 600 s. `--request-timeout` (default 3600 s) is the answer.
- **vLLM's tool-call parser** (2): it rejected a replayed assistant turn whose arguments the model had written as invalid JSON. That is the server's parser, not the client's history.

Those runs were not made with this runner, so the counts are what to expect, not measured here.

## In suites

`bfcl_v3` and `bfcl_v4` rows of a suite (`"framework": "bfcl"`) launch `scripts/custom/run_bfcl.sbatch`, a CPU job that calls the served model: see [suites](suites.md). `run_bfcl.sh` uploads `summary.json` to W&B (`--wandb-entity`, `--wandb-project`, `--no-wandb`): one task, `bfcl_v3` or `bfcl_v4`, with `overall` (v3), the groups, every category's accuracy and `n_samples`; `scripts/results/update_wandb_bfcl.py` does it from the venv, which has `wandb` for that.
