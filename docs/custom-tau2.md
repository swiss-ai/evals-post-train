# tau2-bench, straight from upstream

`scripts/custom/run_tau2_bench.sh` runs [tau2-bench](https://github.com/sierra-research/tau2-bench) as upstream ships it, with nothing of ours in between: any tag, branch or commit of the repository (`--git-ref`, default `v2.0.0`), the dependencies upstream's own `uv.lock` pins, and every domain that ref has. At `v2.0.0` those are `mock`, `airline`, `retail`, `telecom`, `telecom-workflow`, `banking_knowledge` and `intake`. The defaults are tau2-bench's own, not a leaderboard's protocol.

```bash
scripts/custom/run_tau2_bench.sh --domain retail --model <served model> \
  --api-base-url "$EVAL_API_BASE_URL" --num-tasks 2
```

- **Install.** In the [prebuilt eval image](lm-eval-harness.md#prebuilt-evaluation-image-no-installs-at-run-time) upstream's checkout and its dependencies are baked in at the ref of `requirements/tau2-upstream.commit` (v2.0.0): nothing is installed or downloaded at run time, and `--git-ref` must name that ref or its commit (another ref, or `--repo`, is refused). Outside the image the ref is checked out into a cache keyed by its commit (`${EVAL_ENV_CACHE_ROOT:-$SCRATCH/eval-envs}/tau2-upstream`, `--cache-dir` to move it) and `uv sync --frozen --extra knowledge` installs it on Python 3.12 into the checkout's own `.venv`; that needs `git`, `uv` and network access the first time a commit is used. Our own lock does not come into it: upstream's `uv.lock` and the ref are the version. A bump of the image's tau2-bench is a change of `requirements/tau2-upstream.commit` (and so a new image).
- **Models.** The model under test is reached like `run_inspect_eval.sh` reaches its model (`--api-base-url`, key `TARGET_API_KEY`, else `CSCS_SERVING_API`). The simulated user, which also judges NL assertions, is the same model on the same endpoint unless `--user-llm` says otherwise; a bare `--user-llm` name uses litellm's own credentials. Sampling and thinking flags are the other launchers' (`--temperature`, `--top-p`, `--sampling-profile`, `--enable-thinking`, `--reasoning-effort`) and apply to the agent only; tau2 sends no sampling setting unless you pass one.
- **Trials.** `--num-trials n` runs every task n times; the summary prints `avg_reward` and pass^k for k = 1..n (all k trials succeed, not pass@k). One trial is the default, so pass^1 is the plain mean.
- **Domain options.** `--retrieval-config` is banking_knowledge's retrieval pipeline. Anything else goes after `--` to `tau2 run`, e.g. `intake` is voice-first, so a text run needs `-- --elicitation-strategy scaffolded`.
- **Results.** `<--workdir>/tau2-data/simulations/<--save-to>/results.json`: tau2 reads its data through a writable view in `--workdir` (the image is read-only), so give a `--workdir` on storage you keep when running in a job, since the container's `/tmp` is lost with it. The default run name is `<domain>-<date and time>`, a new run each time; the name of an earlier run resumes it, but only with the same domain, model and options.

`scripts/custom/` is for runners that do not go through one of the frameworks of this repository.
