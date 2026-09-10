"""SciCode, run via `inspect eval aaii/scicode.py` -- Artificial Analysis's
Intelligence Index protocol, a thin defaults wrapper around inspect_evals'
own `scicode` task (same shape as aaii/hle.py/aaii/gpqa_diamond.py -- see
aaii/hle.py's docstring for why this is a wrapper rather than a from-scratch
task like aa_lcr.py/aa_omniscience.py/aa_critpt.py).

https://artificialanalysis.ai/evaluations/scicode, methodology at
artificialanalysis.ai/methodology/intelligence-benchmarking:
"scientist-annotated background information included in the prompt...
sub-problem level scoring... Pass@1 evaluation criteria", 3 repeats per task.
Mapped onto:
  - provide_scientific_background=True  the background-included variant --
    inspect_evals itself defaults to the harder, background-free one. A real
    keyword parameter on inspect_evals' own scicode function (confirmed via
    `inspect.signature`).
  - epochs=3 (AA's repeats; inspect_evals' own Task default is 1, i.e.
    pass@1 with no repeats at all). Unlike provide_scientific_background,
    inspect_evals' scicode() function itself takes no epochs kwarg -- set
    directly on the returned Task object instead (Task.epochs is a plain
    instance attribute, not a frozen/dataclass field -- confirmed by reading
    inspect_ai's own Task.__init__).
  - Pass@1/sub-problem-level scoring is just how inspect_evals/scicode's own
    verify() scorer already reports -- nothing to configure for it.

Deliberately NOT pinned here, unlike evals-svc's own AAII defaults-injection
for this task: --sandbox, --timeout, --display, --log-level. Those are
deployment/infra choices (evals-svc pins `--sandbox local` because its
k8s/FirecREST launchers can't grant privileged Docker), not part of AA's
protocol -- a direct `inspect eval aaii/scicode.py` run here gets
inspect_evals' own native Docker sandbox, same as plain `scicode` and the
README's GDPval example, and needs a real Docker daemon (true on a login
node / the sbatch container, same as GDPval).

Plain `inspect_evals/scicode` (bare `--task scicode`) is also selectable and
runs completely unwrapped -- no background info, inspect_evals' own default
epochs (1).

Both provide_scientific_background=True and epochs=3 are this wrapper's own
keyword *defaults*, not values hardcoded into the call -- so
`--task-arg provide_scientific_background=False --task-arg epochs=1` still
overrides them (epochs here is this wrapper's own task-arg, not the native
`inspect eval --epochs` flag gpqa.py's plain-task history needed a `--`
relocation for -- since Task.epochs is set from this parameter inside the
function body, `-T epochs=...` reaches it directly). A no-argument wrapper
that applies these directly, as an earlier version of this file did,
silently drops such overrides instead (same mistake caught in aaii/hle.py,
confirmed live against evals-svc's own self-grading path for that one).
"""

from inspect_ai import Task, task
from inspect_evals.scicode.scicode import scicode as _scicode


@task
def scicode(provide_scientific_background: bool = True, epochs: int = 3) -> Task:
    t = _scicode(provide_scientific_background=provide_scientific_background)
    t.epochs = epochs
    return t
