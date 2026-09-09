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
"""

from inspect_ai import Task, task
from inspect_evals.scicode.scicode import scicode as _scicode


@task
def scicode() -> Task:
    t = _scicode(provide_scientific_background=True)
    t.epochs = 3
    return t
