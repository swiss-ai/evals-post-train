"""HealthBench with a rubric grader served on the gateway, run via
`inspect eval scripts/inspect_ai/tasks/healthbench_gateway.py -T judge_model=<model>`.

This is inspect_evals' `healthbench` (same data, prompts and scoring) with two changes:

- A run names its own grader. evals-svc refuses the bare `healthbench` task unless it can
  resolve the default OpenAI judge (`openai/gpt-4o-mini`) or the model under test is asked
  to grade itself, and neither is wanted for a launched open grader.
- Reasoning left in the answer is removed before grading. A server without a reasoning
  parser (Apertus 1.5 thinking) returns `<|inner_prefix|>...<|inner_suffix|>` inline,
  and the grader would otherwise score the thinking too. An answer still inside its
  thinking (no closing token: it ran out of tokens) is graded as empty.

Grader on the same gateway as the model under test:
    --task-arg judge_model=openai-api/swissai/<served model name>
Grader on the gateway while the model under test is launched locally (the `swissai`
provider then points at the local server):
    --task-arg judge_model=openai-api/gateway/<served model name>
`gateway` defaults to https://api.swissai.svc.cscs.ch/v1 with the job's CSCS_SERVING_API
key; GATEWAY_BASE_URL / GATEWAY_API_KEY override them.

Scores are not comparable with the published HealthBench numbers, which are graded by
GPT-4.1.
"""

import os
import re

from inspect_ai import Task, task
from inspect_ai.solver import Generate, Solver, TaskState, solver

GATEWAY_BASE_URL = "https://api.swissai.svc.cscs.ch/v1"
THINK_START = "<|inner_prefix|>"
THINK_END = "<|inner_suffix|>"
_THINK_BLOCK = re.compile(re.escape(THINK_START) + r".*?" + re.escape(THINK_END), re.S)


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
def healthbench_gateway(
    judge_model: str,
    subset: str = "full",
    **kwargs,
) -> Task:
    """
    Args:
        judge_model: Grader model name (required, no default).
        subset: "full", "hard", "consensus" or "meta_eval".
        **kwargs: Passed through to inspect_evals' `healthbench` (for example
            `bootstrap_samples`, `include_subset_scores`, `judge_max_tokens`).
    """
    from inspect_evals.healthbench import healthbench

    _configure_gateway_provider(judge_model)
    base = healthbench(subset=subset, judge_model=judge_model, **kwargs)
    return Task(
        dataset=base.dataset,
        solver=[base.solver, strip_inline_reasoning()],
        scorer=base.scorer,
        version=base.version,
        metadata=base.metadata,
    )
