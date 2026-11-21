from copy import deepcopy
import unittest

from scripts.helpfulness import arena_alignment as alignment
from scripts.helpfulness.teacher_smoke import FAILURE_FLAGS


class ArenaAlignmentTests(unittest.TestCase):
    def fixture(self):
        cases = [{"case": {"id": str(i), "category": "hard_prompt", "question": "q", "reference_answer": "r"},
                  "answer": "a", "split": split}
                 for i, split in enumerate(["pilot_review"] * 20 + ["fresh_constructed_controls"] * 20)]
        oracle = {"cases": [{"case_id": str(i), "allowed_margins": [1, 2], "basis": "Constructed factual error"}
                            for i in range(40)]}
        responses = [{"choices": [{"finish_reason": "stop", "message": {
            "content": "The candidate is correct. " + ("[[B>A]]" if i % 2 == 0 else "[[A>B]]")}}]}
            for i in range(80)]
        return cases, oracle, responses

    def test_gate_uses_both_orders_and_separate_twenty_case_denominators(self):
        cases, oracle, responses = self.fixture()
        rows, summary = alignment.results(cases, oracle, responses)
        self.assertTrue(summary["alignment_gate_passed"])
        self.assertFalse(summary["automatic_scoring_ready"])
        for index in (0, 2, 4, 40, 42, 44):
            responses[index]["choices"][0]["message"]["content"] = "Both answers are equivalent. [[A=B]]"
        rows, summary = alignment.results(cases, oracle, responses)
        self.assertTrue(summary["alignment_gate_passed"])
        self.assertEqual(summary["split_results"]["pilot_review"]["complete_case_matches"], 17)
        responses[6]["choices"][0]["message"]["content"] = "Both answers are equivalent. [[A=B]]"
        self.assertFalse(alignment.results(cases, oracle, responses)[1]["alignment_gate_passed"])

    def test_invalid_output_blocks_gate_even_above_count_threshold(self):
        cases, oracle, responses = self.fixture()
        responses[0]["choices"][0]["finish_reason"] = "length"
        _, summary = alignment.results(cases, oracle, responses)
        self.assertEqual(summary["valid_orders"], 79)
        self.assertEqual(summary["split_results"]["pilot_review"]["complete_case_matches"], 19)
        self.assertFalse(summary["alignment_gate_passed"])
        with self.assertRaises(ValueError):
            alignment.results(cases, oracle, responses[:-1])

    def test_model_failure_has_no_judge_orders_and_remains_in_twenty_case_denominator(self):
        cases, oracle, responses = self.fixture()
        cases[0].update(status="model_no_final_reply", answer="")
        oracle["cases"][0]["allowed_margins"] = []
        rows, summary = alignment.results(cases, oracle, responses[2:])
        self.assertEqual(rows[0]["orders"], [])
        self.assertEqual(rows[0]["scores"], {})
        self.assertEqual(summary["expected_judge_orders"], 78)
        self.assertEqual(summary["split_results"]["pilot_review"]["selected_cases"], 20)
        self.assertEqual(summary["split_results"]["pilot_review"]["complete_case_matches"], 19)
        self.assertFalse(summary["checkpoint_reply_coverage_complete"])
        self.assertTrue(summary["alignment_gate_passed"])
