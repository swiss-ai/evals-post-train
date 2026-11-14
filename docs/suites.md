# Suites

A suite writes down a comparison once: its benchmarks, how each is repeated and scored, and the setups a model can run under. It lives in `configs/suites/<name>.json`; `scripts/common/suites.py` reads it and knows no model.

```bash
python -m scripts.common.suites show table25
python -m scripts.common.suites plan table25 --setup apertus-1.5-70b-thinking
python -m scripts.common.suites flags table25 --setup apertus-1.5-70b-thinking --task ifbench
# --enable-thinking --chat-template --max-length 32768 --repeats 4
```

`plan` prints one row per benchmark (label, task, launcher flags, what it reports, the results key). Give `flags` output to the launcher next to `--task`.

## File

| Part | Fields |
|---|---|
| `benchmarks` (list, table order) | `label` (the table row), `task`, `framework`, `repeats {n, via}`, `metric {label, key}` |
| `setups` (by name) | `max_length` (the model's maximal window), `flags` (thinking, chat format, tokenizer) |

- `repeats.via` is `flag` when the task has no repeats of its own and the launcher gets `--repeats n`, or `task` when its yaml already repeats `n` times (AIME 8, the self-consistency tasks 4). Passing the flag there would apply the repeats twice, so nothing is passed.
- `metric.label` is what the table shows (`mean@4`); it must agree with `repeats.n`. `metric.key` is the key in a run's results that holds the value, `null` until it has been checked against a finished run.
- `--max-length` and `--repeats` come from `max_length` and `repeats`, so a setup's `flags` may not contain them. Keep the served `--max-model-len` equal to `max_length`.
- A malformed file is refused with the field named (`benchmarks[5] (aime24): repeats.via must be one of ...`).

Serving (nodes, image, partition, wall time) is not in a suite; it belongs to whoever launches the model. To add a comparison, copy `table25.json`; to add a kind of model, add a setup. The `tasks_*.txt` lists the launcher modes read are separate and unchanged.
