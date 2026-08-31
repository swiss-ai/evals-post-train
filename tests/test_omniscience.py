"""AA-Omniscience task (custom_tasks/omniscience.py): the two custom metrics
(omniscience_index, hallucination_rate/non_hallucination_rate) implement AA's
own formulas from raw C/P/I/N counts, and the grade-parsing regex has to
survive an early, off-menu mention of a grade letter inside the judge's own
chain-of-thought reasoning -- both are exactly the kind of thing that's easy
to get an off-by-one or precedence bug into.

IntegrationTests below is not redundant with the unit tests above it: a
first version of these metrics passed every one of those (calling
omniscience_index()/hallucination_rate() directly with literal
SampleScore(score=Score(value=CORRECT)) objects) while being silently wrong
end-to-end. The reason: inspect_ai only hands a custom @metric raw
categorical Score.value by default when it is a dict-returning metric like
frequency() (scores="unreduced"); a scalar-returning custom metric with the
default scores="auto" instead receives every value pre-collapsed through
value_to_float() (C/P/I/N -> 1.0/0.5/0/0) before ever reaching the function
-- so counts.get(CORRECT, ...) et al matched nothing and both metrics always
computed 0.0, invisibly, in a real `inspect eval` run. Constructing
SampleScore by hand in a unit test can never reproduce that, since it skips
the reduction pipeline entirely -- only running the actual scorer through
inspect_ai's own eval() (here with the mockllm provider, so no network
call/API key is needed) exercises it."""

import unittest

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, PARTIAL, SampleScore, Score
from inspect_ai.solver import generate

from custom_tasks.omniscience import (
    _GRADE_RE,
    hallucination_rate,
    non_hallucination_rate,
    omniscience_index,
    omniscience_scorer,
)


def _scores(*values: str) -> list[SampleScore]:
    return [SampleScore(score=Score(value=v)) for v in values]


class OmniscienceIndexTests(unittest.TestCase):
    def test_all_correct_is_100(self):
        scores = _scores(CORRECT, CORRECT, CORRECT)
        self.assertEqual(omniscience_index()(scores), 100.0)

    def test_all_incorrect_is_minus_100(self):
        scores = _scores(INCORRECT, INCORRECT)
        self.assertEqual(omniscience_index()(scores), -100.0)

    def test_partial_and_noanswer_count_toward_total_but_not_the_numerator(self):
        # 2 correct, 1 incorrect, 1 partial, 1 not-attempted -> (2-1)/5 * 100
        scores = _scores(CORRECT, CORRECT, INCORRECT, PARTIAL, NOANSWER)
        self.assertEqual(omniscience_index()(scores), 20.0)

    def test_empty_scores_is_zero_not_a_divide_by_zero(self):
        self.assertEqual(omniscience_index()([]), 0.0)


class HallucinationRateTests(unittest.TestCase):
    def test_excludes_correct_from_the_denominator(self):
        # Correct answers don't belong in "how often did it guess wrong
        # instead of abstaining" -- only wrong/partial/abstained do.
        scores = _scores(CORRECT, CORRECT, CORRECT, INCORRECT)
        self.assertEqual(hallucination_rate()(scores), 1.0)

    def test_abstaining_lowers_the_rate(self):
        scores = _scores(INCORRECT, NOANSWER, NOANSWER, NOANSWER)
        self.assertEqual(hallucination_rate()(scores), 0.25)

    def test_all_correct_is_zero_not_a_divide_by_zero(self):
        scores = _scores(CORRECT, CORRECT)
        self.assertEqual(hallucination_rate()(scores), 0.0)

    def test_non_hallucination_rate_is_the_complement(self):
        scores = _scores(INCORRECT, NOANSWER, NOANSWER, NOANSWER)
        self.assertEqual(
            hallucination_rate()(scores) + non_hallucination_rate()(scores), 1.0
        )

    def test_non_hallucination_rate_with_no_denominator_is_one(self):
        # No incorrect/partial/not-attempted answers at all -> nothing to
        # hallucinate on -- the complement of hallucination_rate()'s own 0.0
        # in that case, not an independent divide-by-zero guard.
        scores = _scores(CORRECT, CORRECT)
        self.assertEqual(non_hallucination_rate()(scores), 1.0)


class GradeParsingTests(unittest.TestCase):
    def test_parses_each_letter_case_insensitively(self):
        for letter in "CPIN":
            self.assertEqual(
                _GRADE_RE.search(f"some reasoning.\nGRADE: {letter}").group(1).upper(),
                letter,
            )
            self.assertEqual(
                _GRADE_RE.search(f"some reasoning.\ngrade: {letter.lower()}")
                .group(1)
                .upper(),
                letter,
            )

    def test_binds_to_the_last_grade_mention_not_the_first(self):
        # A grader reasoning out loud ("this looks like GRADE: C at first,
        # but...") before its real final verdict must not have the earlier
        # mention picked up over the actual answer.
        text = "This looks like GRADE: C at first, but on reflection GRADE: I"
        self.assertEqual(_GRADE_RE.search(text).group(1).upper(), "I")

    def test_no_grade_present_does_not_match(self):
        self.assertIsNone(_GRADE_RE.search("The answer is correct."))


def _run_with_fixed_grades(grades: list[str]) -> dict[str, float]:
    """Runs omniscience_scorer() through a real inspect_ai eval(), with the
    grader role bound to a mockllm model that returns each of `grades` in
    turn (one per sample) -- no network call, no API key, but the actual
    scores="auto"/"unreduced" reduction pipeline runs for real. Returns
    {metric_name: value}."""
    n = len(grades)
    dataset = [Sample(input=f"question {i}", target="the answer", id=i) for i in range(n)]
    target_model = get_model(
        "mockllm/target", custom_outputs=[ModelOutput.from_content("mockllm", "an answer")] * n
    )
    grade_iter = iter(grades)
    grader_model = get_model(
        "mockllm/grader",
        custom_outputs=lambda *a, **k: ModelOutput.from_content(
            "mockllm", f"reasoning...\nGRADE: {next(grade_iter)}"
        ),
    )
    task = Task(dataset=dataset, solver=[generate()], scorer=omniscience_scorer())
    logs = inspect_eval(
        task,
        model=target_model,
        model_roles={"grader": grader_model},
        display="none",
        log_dir="/tmp/test_omniscience_integration_logs",
    )
    metrics: dict[str, float] = {}
    for score in logs[0].results.scores:
        metrics.update({name: m.value for name, m in score.metrics.items()})
    return metrics


class IntegrationTests(unittest.TestCase):
    """Runs the real scorer/metrics through inspect_ai's own eval() -- see
    the module docstring for why the unit tests above cannot substitute for
    this."""

    def test_metrics_match_the_formulas_on_an_asymmetric_distribution(self):
        # 3 correct, 0 partial, 1 incorrect, 1 not-attempted (5 total) --
        # deliberately NOT correct==incorrect, so a metric that silently
        # computes 0.0 for everything (this task's actual past bug) cannot
        # be mistaken for a coincidentally-correct answer.
        metrics = _run_with_fixed_grades(["C", "C", "C", "I", "N"])
        self.assertEqual(metrics["accuracy"], 0.6)
        self.assertEqual(metrics["omniscience_index"], 40.0)  # 100*(3-1)/5
        self.assertEqual(metrics["hallucination_rate"], 0.5)  # 1/(0+1+1)
        self.assertEqual(metrics["non_hallucination_rate"], 0.5)

    def test_metrics_match_the_formulas_on_all_correct(self):
        metrics = _run_with_fixed_grades(["C", "C", "C"])
        self.assertEqual(metrics["omniscience_index"], 100.0)
        self.assertEqual(metrics["hallucination_rate"], 0.0)
        self.assertEqual(metrics["non_hallucination_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
