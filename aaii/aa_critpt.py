"""CritPt (Artificial Analysis's research-level physics reasoning benchmark),
run via `inspect eval aaii/aa_critpt.py` -- no inspect_evals implementation
exists, so this is a from-scratch task, same situation as
aaii/aa_lcr.py/aa_omniscience.py.

https://artificialanalysis.ai/methodology/intelligence-benchmarking, the
paper (arxiv.org/abs/2509.26574), and the reference implementation
(github.com/CritPt-Benchmark/CritPt): AA implements the "challenge" level
components for all 70 test-set challenges (the example challenge is a
separate worked walkthrough living at data/example_challenges/ in the
reference repo -- it is not part of the public HF test set at all, so no
explicit exclusion is needed here), 5 repeats per question (--epochs 5),
pass@1. Verified directly against
huggingface.co/datasets/CritPt-Benchmark/CritPt via the datasets-server API
(https://datasets-server.huggingface.co/rows?dataset=CritPt-Benchmark/CritPt),
not just the dataset card: 70 rows, split "train", every single row's
problem_type == "main" -- so "challenge level" is this dataset's entire
content; no problem_type filter is needed or applied below.

Two-step protocol (per AA's own description): step 1 asks the model to
solve the challenge with reasoning; step 2 asks it to reformat the final
answer into the expected code format (numerical value / SymPy expression /
Python function, per each challenge's own `code_template` column) for
grading. Both prompts are copied verbatim from the reference
implementation's own templates/prompt_template_default.yaml (SYSTEM_PROMPT
rendered for PROMPT_SPECS_STYLE == "two-step" with the default
PROMPT_SPECS_PRECISION_DECIMAL of 12, and PARSE_PROMPT) -- not
reconstructed -- same "primary source, not adapted" approach as
aa_lcr.py's prompt template.

Notably, CritPt's own reference solver
(src/critpt/generation/solver.py, solve_with_parse.py) is ALREADY built on
Inspect AI (TaskState/Generate, the same solver signature this repo's own
tasks use) -- confirmed by reading it directly, not assumed. But it's
written for their general-purpose multi-step "main + sub-problem"
generation CLI (jinja-templated system prompts spread across several YAML
files, a notebook/JSON dataset reader, on-disk caching and artifacts) that
has no use here: this task only ever needs the single "main" challenge
step, never their sub-problem decomposition machinery. So the two-step
logic below is a minimal, self-contained reimplementation of just that one
step -- the same "clean reimplementation over a heavy dependency" choice
aa_lcr.py/omniscience.py already made for their own from-scratch tasks.

GRADING is the real departure from every other task in this repo. AA's own
description: "the official CritPt grading server is used to assess all
challenge responses for correctness. Grading API access is granted case by
case ... email critpt@artificialanalysis.ai." The real, production endpoint
is Artificial Analysis's own POST /api/v2/critpt/evaluate (documented at
https://artificialanalysis.ai/data-api/docs#evaluateCritPt, though that page
renders truncated -- verified instead against the actual OpenAPI spec at
https://artificialanalysis.ai/api/v2/openapi, operationId evaluateCritPt): a
wrapper in front of the same underlying CritPt evaluation server, accepting
POST {"submissions": [{problem_id, generated_code, model, generation_config,
messages}, ...], "batch_metadata": {...}} (generation_config and
batch_metadata are both required by the schema, even if empty), an
`x-api-key` header, and returning grading results whose exact shape "is
determined by the upstream grader" -- confirmed identical in shape to the
reference implementation's own client (src/critpt/evaluation/eval_client.py)
read directly earlier in this file's development. Per the OpenAPI spec:
"Rate limited to 10 requests per day per user. Only successful gradings
(200) count toward the limit -- grader error responses (400/422/503/5xx) and
grader timeouts are refunded." 400 covers "the grading server rejected the
batch (e.g. submission count mismatch)" -- confirms (from the authoritative
source, not just the reference repo's own README, which says the same
thing: "our grading server only accepts complete batches containing
responses to all 70 problems") that an incomplete batch is a real rejection
condition, not just a courtesy the client-side guard below adds. 403 means
"the account has not been granted CritPt grading access" -- exactly the
"granted case by case" state most runs of this task will be in until AA
approves it.

This is fundamentally incompatible with Inspect's per-sample Scorer API,
which scores one sample as soon as its own solver finishes -- there is no
per-sample hook point where "every other sample in this epoch is also
done" is guaranteed. So, same as inspect_evals' own gdpval.py (confirmed by
reading it directly): the @scorer below does no real grading, only
captures each sample's parsed answer (critpt_capture_scorer, an
exact()-style stub, same idea as GDPval's own scorer parameter). Real
grading happens in SubmitToCritPtGradingServer, an Inspect Hook
(on_task_end, the same mechanism inspect_evals/gdpval/hooks.py's
ConsolidateDeliverables uses, confirmed by reading that file directly) that
fires once the WHOLE run (every sample, every epoch) is actually done. It
groups completed samples by epoch (each epoch == one full 70-challenge
pass == exactly one grading-server batch, matching the "complete batch of
all 70" requirement -- 5 epochs means 5 of the account's 10 daily
submissions used by one full AAII run) and POSTs each complete batch if
CRITPT_API_KEY/CRITPT_GRADING_URL are set in the run's environment. An
incomplete batch (e.g. a caller testing with --limit) is never POSTed --
the real server would just reject it, and there is no reason to spend one
of the 10 daily submission slots on a guaranteed failure. If no credential
is set (the expected case until AA grants this account access), or a batch
is incomplete, its payload is instead written to a local JSON file next to
the eval log for manual submission later -- same fallback shape as
GDPval's own upload_to_hf=False default. There is no LLM-judge self-grading
fallback here, unlike aa_lcr.py/omniscience.py/hle.py: this is a fixed
external grading service, not a model role, so there is nothing for a
caller to substitute.

Because grading can't happen inline, this task's own `inspect eval` run
never reports a real accuracy score in Inspect's own log/metrics -- same
caveat GDPval's own README states about its scorer. The real score comes
from whatever the grading server (or a later, offline read of the written
batch files once graded) returns, outside this task entirely.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, hf_dataset
from inspect_ai.hooks import Hooks, TaskEnd, hooks
from inspect_ai.log import EvalSample, read_eval_log_async
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver

DATASET_PATH = "CritPt-Benchmark/CritPt"
# Verified via the datasets-server API (see module docstring), not assumed.
DATASET_SPLIT = "train"

# All 70 challenges must be present in one submission (see module
# docstring) -- a run with fewer samples than this can never be graded by
# the real server, so it is never worth POSTing.
CHALLENGE_COUNT = 70

GRADING_URL_ENV = "CRITPT_GRADING_URL"
GRADING_API_KEY_ENV = "CRITPT_API_KEY"
# The real, production endpoint (see module docstring) -- a hardcoded
# fallback, not just evals-svc's own settings.critpt_grading_url default, so
# this task is self-sufficient when run standalone (`inspect eval
# aaii/aa_critpt.py` directly, outside evals-svc's own env-building).
DEFAULT_GRADING_URL = "https://artificialanalysis.ai/api/v2/critpt/evaluate"

# Verbatim from the reference repo's templates/prompt_template_default.yaml,
# SYSTEM_PROMPT rendered for PROMPT_SPECS_STYLE == "two-step" with the
# default PROMPT_SPECS_PRECISION_DECIMAL (12) -- see module docstring.
SYSTEM_PROMPT_TWO_STEP = """\
You are a physics research assistant specializing in solving complex, \
research-level problems using precise, step-by-step reasoning.

**Input**

Problems will be provided in Markdown format.

**Output (Markdown format)**

1. **Step-by-Step Derivation** - Show every non-trivial step in the solution. \
Justify steps using relevant physical laws, theorems, or mathematical identities.

2. **Mathematical Typesetting** - Use LaTeX for all mathematics: \
`$...$` for inline expressions, `$$...$$` for display equations.

3. **Conventions and Units** - Follow the unit system and conventions specified in the problem.

4. **Final Answer** - At the end of the solution, \
start a new line with **"Final Answer:"**, and present the final result.

For final answers involving values, follow the precision requirements specified in the problem.

If no precision is specified:

- If an exact value is possible, provide it (e.g., $\\sqrt(2)$, $\\pi/4$).

- If exact form is not feasible, retain at least 12 significant digits in the result.

5. **Formatting Compliance** - If the user requests a specific output format (e.g., code, table), \
provide the final answer accordingly.\
"""

# Verbatim from the same file's PARSE_PROMPT, with {code_template} standing
# in for the Jinja PROMPT_SPECS_ANSWER_CODE_TEMPLATE substitution -- each
# challenge's own code_template column, populated verbatim (raw, per the
# original's "{% raw %}" wrapping), not re-templated further.
PARSE_PROMPT_TEMPLATE = """\
Populate your final answer into the code template provided below.
This step is purely for formatting/display purposes. No additional reasoning or derivation should be performed.
Do not import any modules or packages beyond what is provided in the template.

```python

{code_template}

```\
"""


def _load_dataset():
    return hf_dataset(
        path=DATASET_PATH,
        split=DATASET_SPLIT,
        sample_fields=FieldSpec(
            input="problem_description",
            # The reference implementation's own solution -- not used for
            # local scoring (see module docstring: grading is external),
            # kept only so a caller reading the log can compare by eye.
            target="answer_code",
            id="problem_id",
            metadata=["code_template", "metadata_tag", "metadata_notebook_path"],
        ),
    )


@solver
def critpt_two_step_solver() -> Solver:
    """Step 1: solve the challenge with reasoning, under CritPt's own
    two-step system prompt. Step 2: reformat the final answer into the
    challenge's own code_template, via a SEPARATE generate() call so the
    reformatting step performs no new reasoning of its own (per both the
    system prompt and the parse prompt's own text above) -- mirrors
    solve_with_parse()'s two-generate structure in the reference
    implementation, without its multi-step/caching machinery (see module
    docstring)."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.messages.insert(0, ChatMessageSystem(content=SYSTEM_PROMPT_TWO_STEP))
        await generate(state)
        # The raw reasoning+answer completion, kept separately (not
        # overwritten by the parse step) so both the grading-server
        # submission's `messages` field and a caller reading the log can
        # see the actual derivation, not just the parsed code that follows.
        state.metadata["reasoning_completion"] = state.output.completion
        state.messages.append(
            ChatMessageUser(
                content=PARSE_PROMPT_TEMPLATE.format(
                    code_template=state.metadata["code_template"]
                )
            )
        )
        await generate(state)
        return state

    return solve


@scorer(metrics=[])
def critpt_capture_scorer() -> Scorer:
    """No local grading -- see module docstring. Just records the parsed
    generated_code (the step-2 completion) as the Score's answer, so it
    ends up in the eval log where SubmitToCritPtGradingServer's hook (or a
    caller reading the log directly) can find it. metrics=[]: Score.value
    here is not a real grade, so an accuracy()-style metric over it would
    just be noise, not a number anyone should read."""

    async def score(state: TaskState, target: Target) -> Score:
        return Score(
            value="submitted",
            answer=state.output.completion,
            explanation=state.metadata.get("reasoning_completion"),
        )

    return score


def _submission_for_sample(sample: EvalSample, model_name: str) -> dict[str, Any]:
    generated_code = sample.output.completion if sample.output else ""
    reasoning = (sample.metadata or {}).get("reasoning_completion", "")
    return {
        "problem_id": sample.id,
        "generated_code": generated_code,
        "model": model_name,
        "generation_config": {},
        "messages": [
            {"role": "assistant", "content": reasoning},
            {"role": "assistant", "content": generated_code},
        ],
    }


def _group_submissions_by_epoch(
    samples: list[EvalSample], model_name: str
) -> dict[int, list[dict[str, Any]]]:
    by_epoch: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        by_epoch.setdefault(sample.epoch, []).append(
            _submission_for_sample(sample, model_name)
        )
    return by_epoch


@hooks(
    name="submit_to_critpt_grading_server",
    description=(
        "Assemble each epoch's 70 CritPt submissions into one grading-server "
        "batch and POST it if CRITPT_API_KEY/CRITPT_GRADING_URL are set, "
        "otherwise (or for an incomplete batch) write it to a local JSON "
        "file for manual submission."
    ),
)
class SubmitToCritPtGradingServer(Hooks):
    async def on_task_end(self, data: TaskEnd) -> None:
        log_location = data.log.location
        if not log_location:
            raise RuntimeError(
                "Log location is not present. CritPt submissions cannot be assembled."
            )

        full_log = await read_eval_log_async(log_location, header_only=False)
        samples = full_log.samples or []
        if not samples:
            return

        model_name = str(full_log.eval.model)
        by_epoch = _group_submissions_by_epoch(samples, model_name)

        # Falls back to the real production endpoint (see DEFAULT_GRADING_URL)
        # rather than requiring the caller to set CRITPT_GRADING_URL -- this
        # task is self-sufficient standalone, not just under evals-svc (which
        # always exports it anyway, since settings.critpt_grading_url has the
        # same default). api_key has no such fallback -- there is no public
        # default credential, and its absence is the expected state until AA
        # grants this account access (see module docstring).
        url = os.environ.get(GRADING_URL_ENV) or DEFAULT_GRADING_URL
        api_key = os.environ.get(GRADING_API_KEY_ENV)

        out_dir = Path(log_location).parent / "critpt_submissions"
        out_dir.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(
            timeout=300.0, headers={"x-api-key": api_key} if api_key else {}
        ) as client:
            for epoch, submissions in sorted(by_epoch.items()):
                payload = {
                    "submissions": submissions,
                    "batch_metadata": {
                        "model": model_name,
                        "epoch": epoch,
                        "generated_at": datetime.now(UTC).isoformat(),
                    },
                }
                batch_path = out_dir / f"epoch_{epoch}_batch.json"
                batch_path.write_text(json.dumps(payload, indent=2))

                # See module docstring / CHALLENGE_COUNT: never spend one of
                # the account's 10 daily submissions on a batch the real
                # server is guaranteed to reject with a 400 (submission
                # count mismatch).
                if not api_key or len(submissions) != CHALLENGE_COUNT:
                    continue

                # A grading failure must never fail the eval run itself --
                # every one of these is a real, documented, non-hypothetical
                # outcome (see module docstring): 403 (access not yet
                # granted -- the common case right after requesting it), 429
                # (10/day rate limit -- refunded, so safe to just retry a
                # later run), 503/504 (grading server busy/timed out). The
                # batch file above is already the durable record either way;
                # this just adds whatever the server said, for debugging.
                try:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                except httpx.HTTPError as exc:
                    status = getattr(exc, "response", None)
                    body = status.text if status is not None else str(exc)
                    (out_dir / f"epoch_{epoch}_error.json").write_text(
                        json.dumps({"error": str(exc), "response_body": body}, indent=2)
                    )
                    continue
                (out_dir / f"epoch_{epoch}_result.json").write_text(
                    json.dumps(response.json(), indent=2)
                )


@task
def critpt() -> Task:
    return Task(
        dataset=_load_dataset(),
        solver=[critpt_two_step_solver()],
        scorer=critpt_capture_scorer(),
    )
