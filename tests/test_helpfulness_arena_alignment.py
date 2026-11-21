from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.helpfulness import arena_alignment as alignment
from scripts.helpfulness.teacher_smoke import FAILURE_FLAGS


class ArenaAlignmentTests(unittest.TestCase):
    def test_structured_strength_requires_a_demonstrated_major_gap(self):
        value = {
            "a_assessment": "Correct computation with a minor omission.",
            "b_assessment": "Same correct computation.",
            "a_status": "no_major_error", "b_status": "no_major_error", "preference": "A",
        }
        self.assertEqual(alignment.structured_preference(json.dumps(value), "B"), -1)
        value["b_status"] = "major_error"
        self.assertEqual(alignment.structured_preference(json.dumps(value), "B"), -2)
        self.assertEqual(alignment.structured_preference(json.dumps(value), "A"), 2)
        value["a_status"] = "uncertain"
        self.assertEqual(alignment.structured_preference(json.dumps(value), "B"), -1)
        value["preference"] = "tie"
        self.assertEqual(alignment.structured_preference(json.dumps(value), "B"), 0)
        value["extra"] = True
        with self.assertRaises(ValueError):
            alignment.structured_preference(json.dumps(value), "B")

    def test_compact_rubric_is_hash_bound_without_changing_native_messages(self):
        cases, _, _ = self.fixture()
        original = [{"candidate_position": "B", "messages": [
            {"role": "system", "content": "native rubric"},
            {"role": "user", "content": "exact question and answers"},
        ]}]
        with TemporaryDirectory() as directory:
            rubric = Path(directory) / "rubric.txt"
            rubric.write_text("Compact blinded criteria", encoding="utf-8")
            args = SimpleNamespace(compact_rubric=rubric, compact_rubric_sha256=hashlib.sha256(rubric.read_bytes()).hexdigest())
            text, protocol, limit = alignment.comparison_profile(args)
            with patch.object(alignment.open_benchmarks, "arena_messages", side_effect=lambda *a: deepcopy(original)):
                native = alignment.plans(cases[:1], {})
                compact = alignment.plans(cases[:1], {}, text)
            self.assertEqual(native[0]["messages"], original[0]["messages"])
            self.assertEqual(compact[0]["messages"][1], original[0]["messages"][1])
            self.assertNotEqual(compact[0]["messages"][0], original[0]["messages"][0])
            self.assertEqual(protocol, alignment.COMPACT_PROTOCOL)
            self.assertEqual(alignment.request(compact[0], "open-judge", limit)["max_tokens"], 512)
            rubric.write_text("Changed rubric", encoding="utf-8")
            with self.assertRaises(ValueError):
                alignment.comparison_profile(args)

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
