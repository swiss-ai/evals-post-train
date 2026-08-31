"""AA-Omniscience (Artificial Analysis's knowledge/hallucination benchmark),
run via `inspect eval custom_tasks/omniscience.py` -- there is no
inspect_evals implementation to lean on (unlike hle/gpqa_diamond), so this is
a from-scratch Task rather than a thin defaults wrapper.

https://artificialanalysis.ai/methodology/intelligence-benchmarking and
https://artificialanalysis.ai/evaluations/omniscience: 6,000 short, exact-
answer factual questions (dates, names, numbers -- not multiple choice)
across 6 domains/42 topics, graded by a judge model into one of four
categories -- correct / partially correct / incorrect / not attempted -- and
combined into two Intelligence Index components: Accuracy (8% weight) and
Non-Hallucination Rate (4% weight, i.e. 1 - hallucination_rate). The judge is
GPT-5.6 Luna (medium), the same model AA uses for HLE.

Two protocol caveats that do NOT have an HLE-style "verified against the
official prompt" equivalent here:
  - Only a 600-question (10%) subset is public
    (https://huggingface.co/datasets/ArtificialAnalysis/AA-Omniscience-Public);
    AA keeps the full 6,000-question set private for their own leaderboard.
    A run against this task is therefore NEVER AAII-comparable, independent
    of how faithful the grading rubric is -- unlike HLE/GPQA Diamond, there
    is no way to close this gap from the caller's side.
  - The grading rubric below is adapted from the category descriptions in
    the paper (arxiv.org/abs/2511.13029) -- correct/partial/incorrect/
    not-attempted -- but is NOT the paper's own verbatim grading prompt
    (Appendix A.2), which was not available to reproduce exactly.

Scoring: Score.value is one of Inspect's own C/P/I/N constants (CORRECT/
PARTIAL/INCORRECT/NOANSWER from inspect_ai.scorer), which is also what
value_to_float()'s default mapping (used by accuracy()) already expects --
"not attempted" maps onto NOANSWER since both mean "no penalized guess was
made". Two custom metrics compute AA's own formulas from the raw C/P/I/N
counts directly (accuracy()'s 1/0.5/0/0 blend is the wrong shape for them):
  omniscience_index      = 100 * (correct - incorrect) / total
  hallucination_rate     = incorrect / (partial + incorrect + not_attempted)
  non_hallucination_rate = 1 - hallucination_rate
"""

import re
from collections import Counter

from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, hf_dataset
from inspect_ai.model import ChatMessageUser, get_model
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    NOANSWER,
    PARTIAL,
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    accuracy,
    frequency,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate

DATASET_PATH = "ArtificialAnalysis/AA-Omniscience-Public"

# The greedy ".*" binds to the LAST "GRADE: X" in the completion, the same
# defense inspect_ai's own model_graded_qa uses: an off-menu or early mention
# of a letter earlier in the grader's chain-of-thought can't get picked up
# over its actual final verdict.
_GRADE_RE = re.compile(r"(?is).*grade\s*:\s*([cpin])\b")
_LETTER_TO_VALUE = {"C": CORRECT, "P": PARTIAL, "I": INCORRECT, "N": NOANSWER}

_GRADING_TEMPLATE = """\
You are grading a submitted answer to a factual question against a reference \
answer.

[Question]: {question}
[Reference answer]: {criterion}
[Submitted answer]: {answer}

Classify the submitted answer into exactly one of these four categories:

CORRECT: The submitted answer fully contains, or is equivalent in meaning \
to, the reference answer. Differences in formatting, phrasing, or a level \
of precision that doesn't change the substance of the answer are fine.
PARTIAL: The submitted answer is accurate and on the right track, but is \
missing part of what the reference answer requires, or is less precise or \
complete than it.
INCORRECT: The submitted answer contradicts the reference answer, or gives \
a materially different answer.
NOT_ATTEMPTED: No answer was given, or the response explicitly states it \
does not know or cannot answer, without guessing.

First give a brief explanation of your reasoning. Then finish your \
response with exactly one line of the form:

GRADE: <letter>

where <letter> is one of C (correct), P (partial), I (incorrect), or N \
(not attempted).
"""


@metric(scores="unreduced")
def omniscience_index() -> Metric:
    """100 * (correct - incorrect) / total -- AA's own bounded (-100..100) index.

    scores="unreduced" is required, not cosmetic: by default (scores="auto")
    a custom metric receives each Score.value already collapsed through
    value_to_float() -- C/P/I/N turned into 1.0/0.5/0/0 -- before this
    function ever sees it, so counts.get(CORRECT, ...) etc. would silently
    match nothing and this would always compute 0.0 (this happened -- see
    tests/test_omniscience.py's IntegrationTests). frequency() below
    declares the same, for the same reason: it is the one built-in metric
    that also needs the raw categorical labels rather than a float blend.
    """

    def compute(scores: list[SampleScore]) -> float:
        if not scores:
            return 0.0
        counts = Counter(s.score.value for s in scores)
        correct, incorrect = counts.get(CORRECT, 0), counts.get(INCORRECT, 0)
        total = len(scores)
        return 100.0 * (correct - incorrect) / total

    return compute


@metric(scores="unreduced")
def hallucination_rate() -> Metric:
    """incorrect / (partial + incorrect + not_attempted) -- excludes correct
    answers from the denominator by design: this measures how often the model
    guesses wrong INSTEAD of abstaining, among the answers it didn't get
    right, not overall error rate. See omniscience_index() for why
    scores="unreduced" is required here."""

    def compute(scores: list[SampleScore]) -> float:
        counts = Counter(s.score.value for s in scores)
        incorrect = counts.get(INCORRECT, 0)
        denom = incorrect + counts.get(PARTIAL, 0) + counts.get(NOANSWER, 0)
        return 0.0 if denom == 0 else incorrect / denom

    return compute


@metric(scores="unreduced")
def non_hallucination_rate() -> Metric:
    """1 - hallucination_rate -- the actual Intelligence Index component
    (4% weight). See omniscience_index() for why scores="unreduced" is
    required here."""

    def compute(scores: list[SampleScore]) -> float:
        counts = Counter(s.score.value for s in scores)
        incorrect = counts.get(INCORRECT, 0)
        denom = incorrect + counts.get(PARTIAL, 0) + counts.get(NOANSWER, 0)
        return 1.0 if denom == 0 else 1.0 - incorrect / denom

    return compute


@scorer(
    metrics=[
        accuracy(),
        stderr(),
        frequency(categories=[CORRECT, PARTIAL, INCORRECT, NOANSWER]),
        omniscience_index(),
        hallucination_rate(),
        non_hallucination_rate(),
    ]
)
def omniscience_scorer(model_role: str = "grader") -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        # required=True: AA's protocol is a fixed external judge (GPT-5.6
        # Luna), not the model under test grading itself -- unlike
        # get_model()'s own default fallback (silently use the evaluated
        # model when no role is bound), which would turn every run without
        # an explicit --model-role grader=... into an unflagged, non-AAII-
        # comparable self-grade. A caller who genuinely wants that can still
        # get it, explicitly, by passing --model-role grader=<same model>.
        grader = get_model(role=model_role, required=True)
        prompt = _GRADING_TEMPLATE.format(
            question=state.input_text,
            criterion=target.text,
            answer=state.output.completion,
        )
        result = await grader.generate([ChatMessageUser(content=prompt)])
        match = _GRADE_RE.search(result.completion)
        # An unparseable grade is scored as NOANSWER (not silently dropped,
        # and not counted as a penalized guess) rather than raising -- a
        # single malformed judge response shouldn't fail an entire run.
        value = _LETTER_TO_VALUE[match.group(1).upper()] if match else NOANSWER
        return Score(
            value=value,
            answer=state.output.completion,
            explanation=result.completion,
        )

    return score


@task
def omniscience(model_role: str = "grader") -> Task:
    return Task(
        dataset=hf_dataset(
            path=DATASET_PATH,
            split="train",
            sample_fields=FieldSpec(
                input="question",
                target="answer",
                id="question_id",
                metadata=["domain", "topic", "subtopic"],
            ),
        ),
        solver=[generate()],
        scorer=omniscience_scorer(model_role=model_role),
    )
