"""GPQA Diamond, run via `inspect eval aaii/gpqa_diamond.py` -- Artificial
Analysis's Intelligence Index protocol, a thin defaults wrapper around
inspect_evals' own `gpqa_diamond` task (same shape as aaii/hle.py -- see that
module's docstring for why this is a wrapper rather than a from-scratch task
like aa_lcr.py/aa_omniscience.py/aa_critpt.py).

https://artificialanalysis.ai/methodology/intelligence-benchmarking:
"198 questions covering biology, physics and chemistry -- we test the GPQA
Diamond subset of the full GPQA dataset (448 questions total)... 4 option
multiple choice format... Regex-based answer extraction with pass@1
scoring", 5 repeats per question.

epochs is a real keyword parameter on inspect_evals' own gpqa_diamond
function (confirmed via `inspect.signature(gpqa_diamond)` against the
installed inspect_evals), forwarded straight into Task(epochs=...) --
so this needs no native `--epochs`-behind-a-literal-`--` CLI relocation the
way evals-svc's own defaults-injection currently does for the same setting.

epochs=5 is this wrapper's own keyword *default*, not a value hardcoded into
the call -- so `--task-arg epochs=1` still reaches inspect_evals/gpqa_diamond
for a smoke test, same as it would unwrapped. A no-argument wrapper that
calls _gpqa_diamond(epochs=5) directly, as an earlier version of this file
did, silently drops any `--task-arg epochs=...` override with a "param not
used" warning instead (confirmed against aaii/hle.py's identical mistake,
caught live against evals-svc's own self-grading path).

Everything else already matches inspect_evals' own defaults, so nothing
else is overridden here: cot=True (matches simple-evals' own CoT-eliciting
prompt, the source AA cites), the full 198-question set (no
high_level_domain/subdomain filter).

Plain `inspect_evals/gpqa_diamond` (bare `--task gpqa_diamond`) is also
selectable and runs unwrapped, at inspect_evals' own default epochs (4)
instead of AA's 5.
"""

from inspect_ai import Task, task
from inspect_evals.gpqa.gpqa import gpqa_diamond as _gpqa_diamond


@task
def gpqa_diamond(epochs: int = 5) -> Task:
    return _gpqa_diamond(epochs=epochs)
