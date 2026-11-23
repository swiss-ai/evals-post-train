"""SimpleQA Verified with a grader on the gateway, run via
`inspect eval scripts/inspect_ai/tasks/simpleqa_verified_gateway.py -T judge_model=<model>`.

This is inspect_evals' `simpleqa_verified` (same data, grader prompt and scoring) with two
changes, the same as healthbench_gateway.py:

- The grader is a task argument instead of a `--model-role grader=...`, so it can live on
  the gateway while the model under test is launched locally:
      --task-arg judge_model=openai-api/gateway/<served model name>
  `gateway` defaults to https://api.swissai.svc.cscs.ch/v1 with the job's CSCS_SERVING_API
  key; GATEWAY_BASE_URL / GATEWAY_API_KEY override them. A model on the same gateway as
  the model under test can use `openai-api/swissai/<served model name>` instead.
- Reasoning left inline in the answer (`<|inner_prefix|>...<|inner_suffix|>`, Apertus 1.5
  thinking served without a reasoning parser) is removed before grading. An answer whose
  thinking never closed is graded as empty.

`scorer` and `grade_parse` are passed through. "original" is the paper's A/B/C grader.
"""

import os
import re

from inspect_ai import Task, task
from inspect_ai.model import get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver

GATEWAY_BASE_URL = "https://api.swissai.svc.cscs.ch/v1"
THINK_START = "<|inner_prefix|>"
THINK_END = "<|inner_suffix|>"
_THINK_BLOCK = re.compile(
    re.escape(THINK_START) + r".*?" + re.escape(THINK_END), re.DOTALL
)


def strip_reasoning(text: str) -> str:
    """The answer without inline thinking; "" if the thinking never closed."""
    text = _THINK_BLOCK.sub("", text)
    if THINK_END in text:
        text = text.rsplit(THINK_END, 1)[1]
    if THINK_START in text:
        text = text.split(THINK_START, 1)[0]
    return text.strip()


@solver
def strip_inline_reasoning() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.output and state.output.choices:
            completion = state.output.completion
            stripped = strip_reasoning(completion)
            if stripped != completion:
                state.output.completion = stripped
        return state

    return solve


def _configure_gateway_provider(judge_model: str) -> None:
    if not judge_model.startswith("openai-api/gateway/"):
        return
    os.environ.setdefault("GATEWAY_BASE_URL", GATEWAY_BASE_URL)
    key = os.environ.get("GATEWAY_API_KEY") or os.environ.get("CSCS_SERVING_API")
    if not key:
        raise RuntimeError(
            "judge_model uses the `gateway` provider but neither GATEWAY_API_KEY nor "
            "CSCS_SERVING_API is set"
        )
    os.environ["GATEWAY_API_KEY"] = key


@task
def simpleqa_verified_gateway(
    judge_model: str,
    scorer: str = "original",
    grade_parse: str = "strict",
) -> Task:
    """
    Args:
        judge_model: Grader model name (required, no default).
        scorer: "original" (the paper's A/B/C grader, the default here) or "tool".
        grade_parse: "strict" or "paper"; only used by the original scorer.
    """
    from inspect_evals.simpleqa import simpleqa_verified

    _configure_gateway_provider(judge_model)
    base = simpleqa_verified(scorer=scorer, grade_parse=grade_parse)
    return Task(
        dataset=base.dataset,
        solver=[base.solver, strip_inline_reasoning()],
        scorer=base.scorer,
        model_roles={"grader": get_model(judge_model)},
        version=base.version,
        metadata=base.metadata,
    )
