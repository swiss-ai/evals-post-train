"""HealthBench with a rubric grader served on the gateway, run via
`inspect eval scripts/inspect_ai/tasks/healthbench_gateway.py -T judge_model=<model>`.

This is inspect_evals' `healthbench` unchanged (same data, prompts and scoring); it only
exists so a run can name its own grader. evals-svc refuses the bare `healthbench` task
unless it can resolve the default OpenAI judge (`openai/gpt-4o-mini`) or the model under
test is asked to grade itself, and neither is wanted for a launched open grader.

Pass the grader with `--task-arg judge_model=openai-api/swissai/<served model name>`.
Scores are not comparable with the published HealthBench numbers, which are graded by
GPT-4.1.
"""

from inspect_ai import Task, task


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

    return healthbench(subset=subset, judge_model=judge_model, **kwargs)
