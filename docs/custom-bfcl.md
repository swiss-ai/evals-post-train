# Custom BFCL runner

`scripts/custom/run_bfcl.sh` runs BFCL (the Berkeley Function Call Leaderboard) with its official package, [`bfcl-eval`](https://pypi.org/project/bfcl-eval/): the package's generation (the model's requests and the multi-turn loop with its simulated tool environments), its checkers and its score aggregation. No framework of this repository sits in between (EvalScope's BFCL adapter averages the per-sample verdicts itself, which is why its score is a different number from the leaderboard's). The driver is `scripts/custom/bfcl.py`; the script runs it in the venv of `requirements/bfcl-runtime.lock` and `requirements/bfcl-eval.txt`: baked into the prebuilt image (`BFCL_VENV`, nothing is installed at run time), elsewhere built into a cache ([locked_venv](../scripts/common/locked_venv.sh)).

```bash
scripts/custom/run_bfcl.sh --version v3 --model <served name> --api-base-url https://api.swissai.svc.cscs.ch/v1 \
  --workdir logs/bfcl/<name>-v3 --reasoning-effort high
```

## What runs

| `--version` | Categories | Entries |
|---|---|---|
| `v3` | non-live (simple Python/Java/JavaScript, multiple, parallel, parallel multiple, irrelevance), live (simple, multiple, parallel, parallel multiple, irrelevance, relevance), multi-turn (base, miss func, miss param, long context) | 4441 |
| `v4` | those and the memory categories (kv, vector, recursive summarization) | 4906 |

BFCL v4's web search categories need a SerpAPI key and are not run (`--test-category web_search_base` adds them). `--test-category` replaces the categories (bfcl-eval's names or groups, repeatable). The model is called in function-calling mode through bfcl-eval's OpenAI chat-completions handler, registered at run time as `<name>-FC`.

## Settings

bfcl-eval sends a temperature of 0.001 and nothing else. The sampling flags (`--temperature`, `--top-p`, `--top-k`, `--min-p`, `--repetition-penalty`, `--sampling-profile`) and the thinking flags (`--enable-thinking`, `--reasoning-effort`) are added to every request ([sampling](sampling.md)); without them the run is bfcl-eval's own default. `--num-threads` (default 32) is the number of parallel requests, `--request-timeout` (default 3600 s) the time one request may take: the OpenAI SDK's own 600 s cut off a few long thinking answers per run in the EvalScope runs. The driver patches two things in the package's handler and nothing else: it passes those settings through, and it keeps the null fields of a vLLM reply out of the assistant turns it replays (they make vLLM answer HTTP 400 and end the multi-turn conversation).

## Results

`<workdir>/summary.json` holds the accuracy, correct count and total of every category, the groups bfcl-eval computes (non-live, live, multi-turn, agentic: each its own aggregation, see `bfcl_eval/eval_checker/eval_runner_helper.py`) and `overall`. A group is only reported when all its categories were run, because bfcl-eval counts a category it was not asked to run as 0 in its tables: without the web search categories there is no agentic group (the memory categories are in the per-category scores). For `v3` that is the mean of the live, non-live and multi-turn groups, as the v3-era package (`bfcl-eval 3.0.0`) computed it. For `v4` there is no overall: the package's weights (10/10/10/30/40 over non-live, live, irrelevance, multi-turn, agentic) need the agentic group, whose web search half is not run. bfcl-eval's own tables are `<workdir>/score/data_*.csv`, the model's answers `<workdir>/result/`.

Rerunning with the same `--workdir` resumes: bfcl-eval skips the entries it already has. `--num-tasks n` runs the first n entries of each category (a smoke test); its scores cover that subset and are not the leaderboard's.

## What the EvalScope runs showed

EvalScope's BFCL code has been removed from the repository; this is what its two Chonk runs showed. Checked against the logs of the two Chonk runs (v3 and v4, 2026-11-20): of ~4400 samples each, about 47 failed in inference and are scored wrong, as bfcl-eval does too (it keeps them in the denominator as `inference_error`). 43-44 of them are `multi_turn_long_context` prompts over the 32768 window (HTTP 400, not fixable by the runner), 3 are request timeouts (hence `--request-timeout`) and 2 are vLLM's tool-call parser rejecting a replayed assistant turn whose arguments the model wrote as invalid JSON (the server's parser, not the client's history). EvalScope's own v3 client also logged unparseable tool-call arguments, which bfcl-eval's handler does not hit.

## In suites

`bfcl_v3` and `bfcl_v4` rows of a suite (`"framework": "bfcl"`) launch `scripts/custom/run_bfcl.sbatch`, a CPU job that calls the served model: see [suites](suites.md). `run_bfcl.sh` uploads `summary.json` to W&B (`--wandb-entity`, `--wandb-project`, `--no-wandb`, as the EvalScope wrapper): one task, `bfcl_v3` or `bfcl_v4`, with `overall` (v3), the groups, every category's accuracy and `n_samples`; `scripts/results/update_wandb_bfcl.py` does it from the venv, which has `wandb` for that.
