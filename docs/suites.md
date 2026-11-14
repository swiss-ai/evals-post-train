# Suites

A suite writes down a comparison once: its benchmarks, how each is repeated and scored, and the setups a model can run under. It lives in `configs/suites/<name>.json` (Apertus 2 suites in `configs/suites/apertus2/`, named `apertus2/<name>`); `scripts/common/suites.py` reads it and knows no model.

```bash
python -m scripts.common.suites show table25   # title and setups
python -m scripts.common.suites plan table25 --setup apertus-1.5-70b-thinking
python -m scripts.common.suites flags table25 --setup apertus-1.5-70b-thinking --task ifbench
# --enable-thinking --chat-template --max-length 32768 --repeats 4
```

`show` prints the title and the setups. `plan` prints one row per benchmark (label, task, launcher flags, what it reports, the results key). Give `flags` output to the launcher next to `--task`.

## File

| Part | Fields |
|---|---|
| `benchmarks` (list, table order) | `label` (the table row), `task`, `framework`, `repeats {n, via}`, `metric {label, key}`, optional `sampling`, optional `task_args` (inspect only) |
| `setups` (by name) | `max_length` (the model's maximal window), `sampling` (default profile), `flags` (by framework: thinking, chat format, tokenizer) |

- `repeats.via` is `flag` when the task has no repeats of its own and the launcher gets `--repeats n` (lm_eval) or `-- --epochs n` (inspect), or `task` when its yaml already repeats `n` times (AIME 8, the self-consistency tasks 4). Passing the flag there would apply the repeats twice, so nothing is passed.
- `metric.label` is what the table shows (`mean@4`); it must agree with `repeats.n`. `metric.key` is the key in a run's results that holds the value, `null` until it has been checked against a finished run.
- `sampling` names a profile of `configs/sampling_defaults.json` (`thinking`, `non_thinking`, `greedy`); a suite holds no temperature or top-p. A setup names its default and a benchmark may name another. A setup with a thinking flag (`--thinking`, `--enable-thinking`) is refused if a benchmark would sample with a temperature of 0. `plan` shows the profile per row; the launcher does not read it yet.
- `flags` is an object keyed by framework, with an entry (`[]` for none) for every framework the suite's benchmarks use: a thinking switch means `--thinking ...` to lm_eval and `--enable-thinking` to inspect. `flags_for` returns the setup's flags of the benchmark's framework, then what the suite adds: lm_eval `--max-length` and `--repeats`; inspect one `--task-arg key=value` per `task_args` entry and `-- --epochs n`. A setup's flags may not contain those. Inspect has no window flag, so `max_length` is only passed to lm_eval; keep the served `--max-model-len` equal to it.
- `task_args` are the `-T` arguments of an inspect task, as strings: `{"cot": "true"}` for `mmlu_0_shot`, `{"fewshot": "0"}` for zero-shot `gsm8k`.
- `name`, `title` and `description` are required. A malformed file is refused with the field named (`benchmarks[5] (aime24): repeats.via must be one of ...`).

Serving (nodes, image, partition, wall time) is not in a suite; it belongs to whoever launches the model. To add a comparison, copy `table25.json`; to add a kind of model, add a setup. The `tasks_*.txt` lists the launcher modes read are separate and unchanged.
