"""AA-Omniscience task (custom_tasks/omniscience.py): the two custom metrics
(omniscience_index, hallucination_rate/non_hallucination_rate) implement AA's
own formulas from raw C/P/I/N counts, and the grade-parsing regex has to
survive an early, off-menu mention of a grade letter inside the judge's own
chain-of-thought reasoning -- both are exactly the kind of thing that's easy
to get an off-by-one or precedence bug into, and neither is covered by
actually running the task (which only exercises the happy path)."""

import unittest

from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, PARTIAL, SampleScore, Score

from custom_tasks.omniscience import (
    _GRADE_RE,
    hallucination_rate,
    non_hallucination_rate,
    omniscience_index,
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


if __name__ == "__main__":
    unittest.main()
