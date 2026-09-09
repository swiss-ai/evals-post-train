"""HLE (Humanity's Last Exam), run via `inspect eval aaii/hle.py` -- Artificial
Analysis's Intelligence Index protocol, a thin defaults wrapper around
inspect_evals' own `hle` task (unlike aa_lcr.py/aa_omniscience.py/aa_critpt.py,
which are from-scratch tasks with no inspect_evals equivalent to wrap).

https://artificialanalysis.ai/methodology/intelligence-benchmarking:
text-only subset (2,158 of the 2,500-question May 2025 revision), an
equality-checker judge "adapted from the original HLE paper" using "GPT-5.6
Luna (medium)", pass@1, a single judge. Mapped onto inspect_evals/hle's own
task parameters (confirmed via `inspect.signature(hle)` against the
installed inspect_evals, all three real keyword args -- no `--task-arg`/`--`
CLI relocation needed the way evals-svc's own defaults-injection currently
does it for the same three settings):
  - include_multi_modal=False  the text-only subset
  - judge_prompt="original"    inspect_evals' verbatim official-paper judge
                                prompt -- the closest match to "adapted from
                                the original HLE paper"; NOT verified against
                                AA's own prompt text, which their methodology
                                page does not publish inline.
  - graders="grader"           AA describes one judge; inspect_evals
                                defaults to two (grader + grader_2).
  - epochs=1 is inspect_evals' own default already (pass@1), left untouched.

The grader model itself is NOT baked in here, same as aa_lcr.py/
aa_omniscience.py's own grader role: pass `--model-role
grader=openai/gpt-5.6-luna` (or any other model) to `run_inspect_eval.sh`
yourself. The dataset (`cais/hle`) is gated on Hugging Face -- your
HF_TOKEN needs to come from an account that has accepted its terms.

Plain `inspect_evals/hle` (bare `--task hle`) is also selectable and runs
completely unwrapped -- no AAII defaults, whatever inspect_evals' own
defaults are (2 graders, multi-modal included).
"""

from inspect_ai import Task, task
from inspect_evals.hle.hle import hle as _hle


@task
def hle() -> Task:
    return _hle(include_multi_modal=False, judge_prompt="original", graders="grader")
