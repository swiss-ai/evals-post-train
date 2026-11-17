from copy import deepcopy
import json
import unittest

from scripts.helpfulness import abstention_calibration as calibration


class EvidenceCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.sample = {
            "case": {
                "id": "controlled:01", "question": "The stored count is 4.", "reference_answers": ["4"],
                "should_abstain": False,
            },
            "answer": "The stored count is **9**.", "status": "generated",
        }
        self.value = {
            "unresolved_interpretations": [],
            "abstention": {"verdict": "no", "evidence_ids": [], "reason": "Direct answer."},
            "reference_agreement": {"verdict": "contradicts", "reason": "Nine differs from four."},
            "whole_answer_factuality": {
                "verdict": "contains_error", "basis": "provided_material", "reason": "The count differs.",
                "errors": [{
                    "reply_evidence_ids": ["a1"], "source_evidence_ids": ["s1"],
                    "basis": "provided_material", "explanation": "The supplied stored count is four.",
                }],
            },
            "flags": {
                "question_ambiguous": False, "references_ambiguous": False,
                "unsupported_assumptions": False, "reason": "",
            },
        }

    def parse(self):
        return calibration.parse_annotation(json.dumps(self.value), self.sample)

    def response(self):
        return {"model": "open-fixture", "choices": [{
            "finish_reason": "stop", "message": {"content": json.dumps(self.value)},
        }]}

    def test_evidence_ids_resolve_markdown_without_rewriting_model_quotes(self):
        self.assertEqual(self.parse(), self.value)
        units = calibration.evidence(self.sample)
        self.assertEqual(units["assistant_reply"]["a1"], "The stored count is **9**.")
        payload = json.loads(calibration.messages(self.sample)[1]["content"])
        self.assertNotIn("id", payload)
        self.assertNotIn("should_abstain", payload)
        self.assertNotIn("expected", payload)

    def test_unknown_wrong_source_and_duplicate_ids_reject(self):
        for ids in (["a999"], ["s1"], ["a1", "a1"], []):
            with self.subTest(ids=ids):
                self.value["whole_answer_factuality"]["errors"][0]["reply_evidence_ids"] = ids
                with self.assertRaisesRegex(ValueError, "evidence IDs"):
                    self.parse()

    def test_scope_review_must_match_flag_and_use_real_source_ids(self):
        self.value["unresolved_interpretations"] = [{
            "reading": "First context", "answer_under_reading": "Yes", "source_evidence_ids": ["s1"],
        }]
        with self.assertRaisesRegex(ValueError, "zero or two"):
            self.parse()
        self.value["unresolved_interpretations"].append({
            "reading": "Second context", "answer_under_reading": "No", "source_evidence_ids": ["s1"],
        })
        with self.assertRaisesRegex(ValueError, "Ambiguity flag"):
            self.parse()
        self.value["flags"].update(question_ambiguous=True, reason="Two contexts.")
        self.parse()
        self.value["unresolved_interpretations"][1]["source_evidence_ids"] = ["a1"]
        with self.assertRaisesRegex(ValueError, "evidence IDs"):
            self.parse()

    def test_request_schema_binds_ids_to_each_actual_reply(self):
        first = calibration.request(calibration.messages(self.sample), "open-fixture")
        self.sample["answer"] += "\nSecond line."
        second = calibration.request(calibration.messages(self.sample), "open-fixture")
        first_ids = first["response_format"]["json_schema"]["schema"]["properties"]["abstention"]["properties"]["evidence_ids"]["items"]["enum"]
        second_ids = second["response_format"]["json_schema"]["schema"]["properties"]["abstention"]["properties"]["evidence_ids"]["items"]["enum"]
        self.assertEqual(first_ids, ["a1"])
        self.assertEqual(second_ids, ["a1", "a2"])

    def test_long_units_keep_exact_text_and_literal_unicode_separators(self):
        text = ("Long **formatted** paragraph. " * 120) + "\u2028still the same physical line"
        units = calibration._units(text, "s")
        self.assertEqual("".join(units.values()), text)
        self.assertTrue(all(len(value) <= 1200 for value in units.values()))
        self.assertIn("\u2028", list(units.values())[-1])

    def test_replay_preserves_separate_control_denominator_and_rejects_tampering(self):
        response = self.response()
        row = calibration.outcome(self.sample, response)
        calls = [
            {"phase": "requested", "request": calibration.request(calibration.messages(self.sample), "open-fixture")},
            {"phase": "returned", "response": response},
        ]
        summary = calibration.replay([self.sample], [row], calls, "open-fixture")
        self.assertEqual(summary["controlled_checks"]["annotated_cases"], 1)
        self.assertEqual(summary["original_reply_diagnostics"]["annotated_cases"], 0)
        changed = deepcopy(row)
        changed["annotation"]["reference_agreement"]["verdict"] = "agrees"
        with self.assertRaisesRegex(ValueError, "differs"):
            calibration.replay([self.sample], [changed], calls, "open-fixture")

    def test_control_gate_uses_frozen_labels_not_candidate_consensus(self):
        row = calibration.outcome(self.sample, self.response())
        expected = {
            "case_id": "controlled:01", "family": "synthetic",
            "abstention": ["no"], "reference_agreement": ["contradicts"],
            "whole_answer_factuality": ["contains_error"], "flags": self.value["flags"].copy(),
        }
        expected["flags"].pop("reason")
        oracle = {"kind": "synthetic_not_human_gold", "gate": "all", "limitation": "controlled only", "cases": [expected]}
        self.assertTrue(calibration.score_controls([row], oracle)["gate_passed"])
        expected["reference_agreement"] = ["agrees"]
        result = calibration.score_controls([row], oracle)
        self.assertFalse(result["gate_passed"])
        self.assertIsNone(result["general_judge_accuracy"])
        self.assertFalse(result["human_gold"])


if __name__ == "__main__":
    unittest.main()
