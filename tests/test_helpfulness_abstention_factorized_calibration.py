from copy import deepcopy
import json
import unittest

from scripts.helpfulness import abstention_factorized_calibration as judge


class FactorizedCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.sample = {"case": {"id": "fresh:01", "question": "The recorded total is 5.", "reference_answers": ["5"]},
                       "answer": "The total is 8.", "status": "generated"}
        self.values = {
            "audit": {"scope": "resolved", "readings": [], "reference_agreement": "contradicts",
                      "reference_issue": "clean", "reason": "Eight differs from five."},
            "assessment": {"reason": "Eight contradicts the supplied five.", "abstention": "no",
                           "abstention_reply_id": "none", "errors": [
                               {"reply_id": "a1", "source_id": "s1", "reason": "Wrong recorded total."},
                           ], "factuality": "contradicted"},
            "assumptions": {"reason": "The source supplies the quantity, so this is not a filled missing premise.",
                            "premise": "none", "reply_id": "none"},
        }

    def response(self, value, finish="stop"):
        return {"model": "open-fixture", "choices": [{"finish_reason": finish,
                                                     "message": {"content": json.dumps(value)}}]}

    def test_premise_decision_is_separate_from_factual_error_and_all_stages_replay(self):
        responses = {stage: self.response(value) for stage, value in self.values.items()}
        row = judge.row_from_responses(self.sample, responses)
        self.assertEqual(row["annotation"]["whole_answer_factuality"]["verdict"], "contains_error")
        self.assertFalse(row["annotation"]["flags"]["unsupported_assumptions"])
        calls = []
        for stage in judge.LIMITS:
            calls.extend([{"phase": "requested", "request": judge.request(
                stage, self.sample, "open-fixture", self.values["audit"] if stage != "audit" else None,
            )}, {"phase": "returned", "response": responses[stage]}])
        oracle = {"regression_cases": [{"case_id": "fresh:01", "abstention": ["no"],
                   "reference_agreement": ["contradicts"], "whole_answer_factuality": ["contains_error"],
                   "flags": {"question_ambiguous": False, "references_ambiguous": False, "unsupported_assumptions": False}}],
                  "fresh_cases": [], "minimum_passes": 17, "gate": "unit fixture", "limitation": "not acceptance"}
        self.assertEqual(judge.replay([self.sample], [row], calls, "open-fixture", oracle)["regression_whole_case_passed"], 1)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            judge.replay([self.sample], [row], calls[:-2], "open-fixture", oracle)

    def test_failed_third_stage_cannot_be_credited_or_inferred_from_other_stages(self):
        responses = {stage: self.response(value) for stage, value in self.values.items()}
        responses["assumptions"] = self.response(self.values["assumptions"], "length")
        row = judge.row_from_responses(self.sample, responses)
        self.assertEqual(row["status"], "annotation_failure")
        self.assertIsNone(row["annotation"])
        self.assertIn("assumptions", row["stage_errors"])
        value = deepcopy(self.values["assumptions"])
        value.update(premise="introduced_missing", reply_id="a1")
        parsed = judge.parse("assumptions", self.sample, self.response(value))
        self.assertTrue(judge.combine({**self.values, "assumptions": parsed})["flags"]["unsupported_assumptions"])
        for update in ({"reply_id": "unknown"}, {"premise": "introduced_missing"}, {"reason": " "}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                judge.parse("assumptions", self.sample, self.response({**self.values["assumptions"], **update}))

    def test_factuality_justification_precedes_verdict_and_premise_prompt_is_blinded(self):
        self.assertEqual(next(iter(judge.schema("assessment", self.sample)["properties"])), "reason")
        self.assertNotIn("assumptions", judge.schema("assessment", self.sample)["properties"])
        body = json.loads(judge.messages("assumptions", self.sample, self.values["audit"])[1]["content"])
        self.assertEqual(set(body), {"question_and_supplied_context", "assistant_reply"})
        self.assertNotIn("scope_audit", body)
        self.assertNotIn("fresh:01", json.dumps(body))


if __name__ == "__main__":
    unittest.main()
