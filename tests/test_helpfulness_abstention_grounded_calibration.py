from copy import deepcopy
import json
import unittest

from scripts.helpfulness import abstention_grounded_calibration as judge


class GroundedCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.sample = {
            "case": {"id": "fresh:01", "question": "The recorded total is 5.", "reference_answers": ["5"]},
            "answer": "The total is **8**.", "status": "generated",
        }
        self.audit = {"scope": "resolved", "readings": [], "reference_agreement": "contradicts",
                      "reference_issue": "clean", "reason": "Eight differs from five."}
        self.assessment = {
            "abstention": "no", "abstention_reply_id": "none", "factuality": "contradicted",
            "assumptions": False, "errors": [{"reply_id": "a1", "source_id": "s1", "reason": "Wrong total."}],
            "reason": "The recorded total disproves eight.",
        }

    def response(self, value, finish="stop"):
        return {"model": "open-fixture", "choices": [{
            "finish_reason": finish, "message": {"content": json.dumps(value)},
        }]}

    def oracle(self):
        return {"fresh_cases": [{
            "case_id": "fresh:01", "abstention": ["no"], "reference_agreement": ["contradicts"],
            "whole_answer_factuality": ["contains_error"],
            "flags": {"question_ambiguous": False, "references_ambiguous": False, "unsupported_assumptions": False},
        }], "gate": "all", "limitation": "constructive fixture"}

    def test_source_scope_and_reference_are_not_merged_with_factuality(self):
        self.audit.update(scope="multiple_readings", readings=[
            {"interpretation": "First reading", "source_id": "s1"},
            {"interpretation": "Second reading", "source_id": "s1"},
        ])
        self.assessment.update(factuality="unsupported", errors=[])
        a = judge.parse("audit", self.sample, self.response(self.audit))
        b = judge.parse("assessment", self.sample, self.response(self.assessment))
        combined = judge.annotation(a, b)
        self.assertTrue(combined["flags"]["question_ambiguous"])
        self.assertEqual(combined["reference_agreement"]["verdict"], "contradicts")
        self.assertEqual(combined["whole_answer_factuality"]["verdict"], "insufficient_evidence")

    def test_empty_reason_bad_ids_and_inconsistent_findings_fail(self):
        for update in ({"reason": "   "}, {"errors": []}, {"abstention_reply_id": "a1"}):
            value = {**self.assessment, **update}
            with self.subTest(update=update), self.assertRaises(ValueError):
                judge.parse("assessment", self.sample, self.response(value))
        bad = deepcopy(self.assessment)
        bad["errors"][0]["source_id"] = "a1"
        from jsonschema.exceptions import ValidationError
        with self.assertRaises(ValidationError):
            judge.parse("assessment", self.sample, self.response(bad))

    def test_failed_audit_preserved_and_never_credited_as_calibration_pass(self):
        responses = {"audit": self.response(self.audit, "length"),
                     "assessment": self.response(self.assessment)}
        row = judge.row_from_responses(self.sample, responses)
        self.assertEqual(row["status"], "annotation_failure")
        self.assertIn("audit", row["stage_errors"])
        self.assertIsNone(row["annotation"])
        self.assertFalse(judge.summarize([row], self.oracle())["gate_passed"])

    def test_two_stage_replay_binds_actual_audit_into_second_request(self):
        responses = {"audit": self.response(self.audit), "assessment": self.response(self.assessment)}
        row = judge.row_from_responses(self.sample, responses)
        calls = []
        for stage in judge.LIMITS:
            calls.extend([
                {"phase": "requested", "request": judge.request(
                    stage, self.sample, "open-fixture", self.audit if stage == "assessment" else None,
                )},
                {"phase": "returned", "response": responses[stage]},
            ])
        result = judge.replay([self.sample], [row], calls, "open-fixture", self.oracle())
        self.assertEqual(result["whole_case_passed"], 1)
        altered = deepcopy(calls)
        body = json.loads(altered[2]["request"]["messages"][1]["content"])
        body["scope_audit"]["scope"] = "missing_information"
        altered[2]["request"]["messages"][1]["content"] = json.dumps(body)
        with self.assertRaisesRegex(ValueError, "request differs"):
            judge.replay([self.sample], [row], altered, "open-fixture", self.oracle())

    def test_absent_references_have_explicit_nonapplicability_without_defaults(self):
        self.sample["case"]["reference_answers"] = None
        spec = judge.schema("audit", self.sample)["properties"]
        self.assertEqual(spec["reference_agreement"]["enum"], ["not_applicable"])
        self.assertEqual(spec["reference_issue"]["enum"], ["no_references"])

    def test_markdown_sources_and_short_output_budgets_are_preserved(self):
        request = judge.request("assessment", self.sample, "open-fixture", self.audit)
        body = json.loads(request["messages"][1]["content"])
        self.assertEqual(body["assistant_reply"]["a1"], "The total is **8**.")
        self.assertNotIn("case_id", body)
        self.assertNotIn("should_abstain", body)
        self.assertEqual(request["max_tokens"], 384)
        self.assertNotIn("basis", request["response_format"]["json_schema"]["schema"]["properties"])


if __name__ == "__main__":
    unittest.main()
