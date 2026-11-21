from copy import deepcopy
import json
import unittest

from scripts.helpfulness import abstention_contrastive_calibration as judge
from scripts.helpfulness import abstention_grounded_calibration as previous


class ContrastiveCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.sample = {"case": {"id": "transfer:01", "question": "The total is 5.", "reference_answers": ["5"]},
                       "answer": "The total is 5.", "status": "generated"}
        self.audit = {"scope": "resolved", "readings": [], "reference_agreement": "agrees",
                      "reference_issue": "clean", "reason": "The answer matches five."}
        self.assessment = {"abstention": "no", "abstention_reply_id": "none", "factuality": "supported",
                           "assumptions": "absent", "errors": [], "reason": "The source states five."}

    def response(self, value):
        return {"model": "open-fixture", "choices": [{"finish_reason": "stop",
                                                     "message": {"content": json.dumps(value)}}]}

    def expected(self, cid):
        return {"case_id": cid, "abstention": ["no"], "reference_agreement": ["agrees"],
                "whole_answer_factuality": ["no_identified_error"],
                "flags": {"question_ambiguous": False, "references_ambiguous": False, "unsupported_assumptions": False}}

    def test_categorical_alternatives_map_exactly_and_do_not_modify_v4_schema(self):
        for label, expected in (("absent", False), ("present", True)):
            with self.subTest(label=label):
                assessment = judge.parse("assessment", self.sample, self.response({**self.assessment, "assumptions": label}))
                self.assertEqual(judge.annotation(self.audit, assessment)["flags"]["unsupported_assumptions"], expected)
        for invalid in (True, False, "false", "true", "unknown"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                judge.parse("assessment", self.sample, self.response({**self.assessment, "assumptions": invalid}))
        self.assertEqual(previous.schema("assessment", self.sample)["properties"]["assumptions"], {"type": "boolean"})

    def test_exact_seventeen_case_threshold_requires_both_splits_and_valid_outputs(self):
        rows, oracle = [], {"regression_cases": [], "fresh_cases": [], "minimum_passes": 17,
                            "gate": "seventeen in each", "limitation": "unit fixture"}
        for split, prefix in (("regression_cases", "fresh"), ("fresh_cases", "transfer")):
            for i in range(1, 21):
                sample = deepcopy(self.sample)
                sample["case"]["id"] = f"{prefix}:{i:02d}"
                rows.append(judge.row_from_responses(sample, {
                    "audit": self.response(self.audit), "assessment": self.response(self.assessment),
                }))
                oracle[split].append(self.expected(sample["case"]["id"]))
        for offset in (0, 20):
            for i in (17, 18, 19):
                rows[offset+i]["annotation"]["flags"]["unsupported_assumptions"] = True
        result = judge.summarize(rows, oracle)
        self.assertEqual((result["regression_whole_case_passed"], result["whole_case_passed"]), (17, 17))
        self.assertTrue(result["gate_passed"])
        rows[36]["annotation"]["flags"]["unsupported_assumptions"] = True
        self.assertFalse(judge.summarize(rows, oracle)["gate_passed"])
        rows[36]["annotation"]["flags"]["unsupported_assumptions"] = False
        rows[0]["status"], rows[0]["annotation"] = "annotation_failure", None
        self.assertFalse(judge.summarize(rows, oracle)["gate_passed"])
        altered = {**oracle, "minimum_passes": 16}
        with self.assertRaises(ValueError):
            judge.summarize(rows, altered)

    def test_shared_replay_keeps_actual_categorical_output_and_audit_binding(self):
        responses = {"audit": self.response(self.audit), "assessment": self.response(self.assessment)}
        row = judge.row_from_responses(self.sample, responses)
        calls = []
        for stage in judge.LIMITS:
            calls.extend([{"phase": "requested", "request": judge.request(
                stage, self.sample, "open-fixture", self.audit if stage == "assessment" else None,
            )}, {"phase": "returned", "response": responses[stage]}])
        oracle = {"regression_cases": [], "fresh_cases": [self.expected("transfer:01")],
                  "minimum_passes": 17, "gate": "unit fixture", "limitation": "not acceptance"}
        result = judge.replay([self.sample], [row], calls, "open-fixture", oracle)
        self.assertEqual(result["whole_case_passed"], 1)
        self.assertEqual(row["stages"]["assessment"]["assumptions"], "absent")
        self.assertEqual(calls[2]["request"]["messages"][0]["content"], judge.RUBRICS["assessment"])
        altered = deepcopy(calls)
        altered[2]["request"] = previous.request("assessment", self.sample, "open-fixture", self.audit)
        with self.assertRaisesRegex(ValueError, "request differs"):
            judge.replay([self.sample], [row], altered, "open-fixture", oracle)


if __name__ == "__main__":
    unittest.main()
