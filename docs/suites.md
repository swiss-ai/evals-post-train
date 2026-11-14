# Suites

A suite names the benchmarks of a comparison and the settings each needs, so a run is not rebuilt from memory. It lives in `configs/suites/<name>.json` and is read by `scripts/common/suites.py`.

| Part | Holds |
|---|---|
| `benchmarks` | Per task: `framework`, `repeats`, `repeats_flag`, `metric` |
| `recipes` | Per kind of setup: `max_length` and the launcher `flags` it needs (thinking, chat format, tokenizer) |

`repeats_flag` is `true` when the task has no repeats of its own and needs `--repeats N`; `false` when its yaml already repeats (AIME 8, the self-consistency tasks 4), where the flag would apply them twice. `metric` is the key of the run's results that holds the table value; `null` means it was not checked against a finished run.

```bash
python -m scripts.common.suites list
python -m scripts.common.suites flags table25 --recipe apertus-1.5-70b-thinking --task ifbench
# --enable-thinking --chat-template --max-length 32768 --repeats 4
```

Pass the printed flags to the launcher next to `--task`. Serving settings (nodes, image, `--max-model-len`, partition, wall time) are not here: they belong to whoever launches the model. Keep `--max-model-len` equal to the recipe's `max_length`.

A recipe is data, not a special case in code: add one for a new kind of setup, with the model's maximal window. `table25` is the thinking-mode comparison of the Apertus 1.5 paper; MATH is not in it yet (its task is undecided).
