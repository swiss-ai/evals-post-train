"""BrowseComp with Tavily web search and no sandbox, run via
`inspect eval scripts/inspect_ai/tasks/browsecomp_search.py -T num_samples=20`.

The dataset, query template and grader are inspect_evals' `browse_comp`. Only the browsing
solver differs. The `browse_comp` task in older inspect_evals releases browses with the
deprecated `web_browser()` tool, which needs a Docker sandbox that does not exist on the
cluster pods and does not use the Tavily key. Here a ReAct agent calls inspect's
`web_search("tavily")` tool, which reads `TAVILY_API_KEY` from the environment (evals-svc
exports it for browse_comp tasks).

Without `scorer_model`, the model under test grades its own answers.
"""

import logging
from textwrap import dedent

from inspect_ai import Task, task
from inspect_ai.agent import react, run
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import web_search

logger = logging.getLogger(__name__)


@solver
def tavily_browsing_solver() -> Solver:
    from inspect_evals.browse_comp.prompts import QUERY_TEMPLATE

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        web_surfer = react(
            name="web_surfer",
            description="Web research assistant",
            prompt=dedent("""
            You are an expert at researching the web to answer questions.
            Use the web_search tool to search for relevant information.
            Refine your searches and cross-check sources as needed to answer the question.
            Note that all the questions have a valid answer, but will likely require searching the web."""),
            tools=[web_search(providers="tavily")],
        )
        agent_state = await run(
            web_surfer, QUERY_TEMPLATE.format(Question=state.input_text)
        )
        state.output.completion = agent_state.output.completion  # type: ignore[attr-defined]
        # The run log shows no tool activity, so report it: a search that fails (bad
        # key, rate limit) otherwise looks like a model that simply got the answer wrong.
        tool_msgs = [m for m in agent_state.messages if getattr(m, "role", "") == "tool"]
        failed = [m for m in tool_msgs if getattr(m, "error", None)]
        logger.warning(
            "browsecomp_search sample=%s search_calls=%d tool_errors=%d",
            state.sample_id,
            len(tool_msgs),
            len(failed),
        )
        for m in (failed or tool_msgs)[:2]:
            logger.warning("browsecomp_search sample=%s tool_result=%.200r", state.sample_id, m.text)
        return state

    return solve


@task
def browsecomp_search(
    num_samples: int | None = None,
    scorer_model: str | None = None,
) -> Task:
    """
    Args:
        num_samples: Maximum number of questions to load.
        scorer_model: Grader model name; defaults to the model under test.
    """
    from inspect_evals.browse_comp import browse_comp

    base = browse_comp(
        with_browsing=False, num_samples=num_samples, scorer_model=scorer_model
    )
    base.solver = tavily_browsing_solver()
    base.sandbox = None
    return base
