# Repeated evaluation

Use `--repeat-config path.json` with the lm-eval launcher. The JSON accepts only
`repeats`, `repeat_scoring` and `task_overrides`, using the harness schema.
The harness owns generation repeats and scoring; the pipeline adds no repeat loop.

```bash
bash scripts/lm_eval_harness/launch_evaluations.sh custom \
  --task-file configs/apertus/tasks_posttrain_thinking.txt \
  --repeat-config configs/apertus/repeat_scoring.example.json \
  --backend openai --api-model-name my-model \
  --api-base-url http://localhost:8000/v1 \
  --chat-format apertus2 --tokenizer /path/to/tokenizer
```

The example policy uses mean@5, mean@10 plus pass@2/5/10 for AIME25,
pass@2/5 for MATH-500's math-verify grader, and one attempt for IFEval.
These are example budgets, not a benchmark standard. Set generation temperature,
reasoning/token budgets and other sampling settings using the existing launcher
options. Greedy decoding with repeats is still greedy. Use the appropriate chat
format for other API models.

**Deployment prerequisite:** this feature requires the harness implementation on
`feat/repeat-scoring`. The existing production harness pin does not include it.
Publish an accessible reviewed revision, build the evaluation environment with
that revision, validate a small served run, then update dependency pins. A policy
run against an old harness fails with a clear error. No service pin or production
environment is changed by the local implementation.

The controller snapshots the JSON into its run directory and includes the policy
content in the resume signature. Changing m, k or a task override invalidates old
results. Each chunk resolves the full suite through the harness task index,
rejects unmatched leaf-task overrides, and forwards only its selected overrides.
Task defaults and per-task overrides retain the harness's precedence rules.
Group/tag names can select benchmarks; override keys must be leaf task names.

Results retain the resolved policy and scalar scores per repeat. Chunk merging
replaces a rerun's entire repeat history, including when m shrinks or the task
returns to legacy scoring. HTML tables show `mean@m` or `pass@k (m=m)` when
protocol metadata is available; groups record each leaf's budget and metric
contributors. W&B stores `repeat_protocols`. Headline mappings prefer means and
never substitute derived pass@k scores. Explicit mappings can select pass@k.

Exports retain suffixed metric identities and protocol details. Repeated runs
whose configured headline metric is absent fail export instead of falling back
to an oracle score. Optional sample export emits separate attempt IDs and scores;
existing private-dataset logging restrictions continue to apply. Legacy result
files retain their existing handling. The static table resolves means, while
per-cell budget labels are available in the HTML table and exported metadata.

Offline tests cover configuration snapshots, group/chunk overrides, resumed
shorter runs, headline selection and attempt exports. The full thinking-suite
inventory resolves 39 selections to 560 declared leaf tasks. Live Apertus and a
second provider smoke test, judge/private-dataset availability and deployment
repinning remain release gates. Provider-specific seed support is recorded by
the harness and does not imply deterministic replay.
