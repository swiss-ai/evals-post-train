from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.helpfulness import abstention_annotation as annotation
from scripts.helpfulness.teacher_smoke import FAILURE_FLAGS


class AbstentionAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.sample = {
            "case": {
                "id": "synthetic", "question": "The supplied material says Yes. Is it true?",
                "reference_answers": ["Yes"], "should_abstain": False,
            },
            "answer": "No. The event occurred in 1900.", "status": "generated",
        }
        self.value = {
            "abstention": {"verdict": "no", "quote": "", "reason": "Direct answer."},
            "reference_agreement": {"verdict": "contradicts", "reason": "No conflicts with Yes."},
            "whole_answer_factuality": {
                "verdict": "insufficient_evidence", "basis": "provided_material",
                "errors": [], "reason": "The date cannot be verified from the supplied material.",
            },
            "flags": {
                "question_ambiguous": False, "references_ambiguous": False,
                "unsupported_assumptions": False, "reason": "",
            },
        }

    def parse(self):
        return annotation.parse_annotation(json.dumps(self.value), self.sample)

    def response(self, value=None, finish="stop"):
        return {"model": "open-fixture", "choices": [{
            "finish_reason": finish,
            "message": {"content": json.dumps(self.value if value is None else value)},
        }]}

    def test_reference_conflict_and_unverified_factuality_remain_separate(self):
        value = self.parse()
        self.assertEqual(value["reference_agreement"]["verdict"], "contradicts")
        self.assertEqual(value["whole_answer_factuality"]["verdict"], "insufficient_evidence")
        payload = json.loads(annotation.messages(self.sample)[1]["content"])
        self.assertNotIn("should_abstain", payload)
        self.assertNotIn("id", payload)

    def test_correct_option_can_have_a_quoted_factual_error(self):
        self.value["reference_agreement"]["verdict"] = "agrees"
        self.value["whole_answer_factuality"] = {
            "verdict": "contains_error", "basis": "model_knowledge",
            "errors": [{
                "quote": "The event occurred in 1900.", "explanation": "Synthetic counterexample.",
                "basis": "model_knowledge", "source_quote": "",
            }], "reason": "The option and explanation are separate.",
        }
        self.assertEqual(self.parse()["whole_answer_factuality"]["verdict"], "contains_error")

    def test_supplied_source_and_answer_quotes_must_be_exact(self):
        self.value["whole_answer_factuality"] = {
            "verdict": "contains_error", "basis": "provided_material",
            "errors": [{
                "quote": "No.", "explanation": "Contradicts the supplied claim.",
                "basis": "provided_material", "source_quote": "The supplied material says Yes.",
            }], "reason": "Evidence-backed synthetic fixture.",
        }
        self.parse()
        self.value["whole_answer_factuality"]["errors"][0]["source_quote"] = "Fabricated source"
        with self.assertRaisesRegex(ValueError, "source quote"):
            self.parse()
        self.value["whole_answer_factuality"]["errors"][0]["quote"] = "Fabricated reply"
        with self.assertRaisesRegex(ValueError, "reply evidence"):
            self.parse()

    def test_absent_references_are_not_scored_as_agreement(self):
        self.sample["case"]["reference_answers"] = None
        with self.assertRaisesRegex(ValueError, "absent references"):
            self.parse()
        self.value["reference_agreement"]["verdict"] = "not_applicable"
        self.parse()

    def test_hypothetical_assumptions_do_not_force_factual_error(self):
        self.value["flags"].update(unsupported_assumptions=True, reason="Arbitrary hypothetical.")
        self.value["whole_answer_factuality"]["verdict"] = "no_identified_error"
        self.parse()

    def test_missing_fields_nonboolean_flags_and_unquoted_abstention_fail(self):
        original = deepcopy(self.value)
        del self.value["flags"]["question_ambiguous"]
        with self.assertRaisesRegex(ValueError, "fields"):
            self.parse()
        self.value = deepcopy(original)
        self.value["flags"]["question_ambiguous"] = 1
        with self.assertRaisesRegex(ValueError, "booleans"):
            self.parse()
        self.value = deepcopy(original)
        self.value["abstention"]["verdict"] = "borderline"
        with self.assertRaisesRegex(ValueError, "bounded text"):
            self.parse()

    def test_invalid_and_truncated_annotation_are_explicit_failures(self):
        self.assertEqual(annotation.outcome(self.sample, self.response(finish="length"))["status"], "annotation_failure")
        broken = deepcopy(self.value)
        broken["whole_answer_factuality"]["verdict"] = "verified_correct"
        self.assertEqual(annotation.outcome(self.sample, self.response(broken))["status"], "annotation_failure")

    def test_model_failure_never_gets_a_judge_request_or_annotation(self):
        failed = deepcopy(self.sample)
        failed.update(answer="", status="model_no_final_reply", generation_info={key: 0 for key in FAILURE_FLAGS})
        failed["generation_info"].update(
            generation_truncated=1, generation_think_truncated=1, apertus2_output="Synthetic repetitive thinking",
        )
        self.assertEqual(annotation.judge_messages([failed]), [])
        row = annotation.outcome(failed, None)
        self.assertEqual(row["status"], "model_failure")
        self.assertIsNone(row["annotation"])

    def test_raw_replay_rejects_tampering_and_keeps_failures_visible(self):
        samples = [self.sample, deepcopy(self.sample)]
        samples[1]["case"]["id"] = "second-synthetic"
        responses = [self.response(), self.response(finish="length")]
        calls = []
        rows = []
        for sample, response in zip(samples, responses):
            calls.extend([
                {"phase": "requested", "request": annotation.request(annotation.messages(sample), "open-fixture")},
                {"phase": "returned", "response": response},
            ])
            rows.append(annotation.outcome(sample, response))
        result = annotation.replay(samples, rows, calls, "open-fixture")
        self.assertEqual(result["annotated_cases"], 1)
        self.assertEqual(result["annotation_failure_case_ids"], ["second-synthetic"])
        self.assertFalse(result["annotation_coverage_complete"])
        altered = deepcopy(rows)
        altered[0]["annotation"]["reference_agreement"]["verdict"] = "agrees"
        with self.assertRaisesRegex(ValueError, "differs"):
            annotation.replay(samples, altered, calls, "open-fixture")
        calls[0]["request"]["max_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "preflight"):
            annotation.replay(samples, rows, calls, "open-fixture")

    def test_preflight_binds_rubric_samples_model_and_endpoint(self):
        args = SimpleNamespace(
            model="open-fixture", base_url="http://127.0.0.1:8000/v1",
            rubric_sha256=annotation.RUBRIC_SHA256, samples_sha256="frozen-samples", identity="fixture",
            selection_sha256="frozen-selection",
        )
        identity = {
            "public_model": "open-fixture", "served_model": args.model, "base_url": args.base_url,
            "annotation_rubric_sha256": annotation.RUBRIC_SHA256,
            "source_samples_sha256": args.samples_sha256,
            "license": "synthetic-test-license", "public_repository_revision": "frozen-test-revision",
        }
        with (
            patch.object(annotation.abstention_pilot, "load_selection", return_value=[self.sample["case"]]),
            patch.object(annotation.abstention_pilot, "load_samples", return_value=[self.sample]),
            patch.object(annotation, "read_json", return_value=identity),
        ):
            _, checked = annotation.preflight(args)
            self.assertEqual(len(checked["judge_messages"]), 1)
            for key in ("served_model", "base_url", "source_samples_sha256", "annotation_rubric_sha256"):
                previous = identity[key]
                identity[key] = "altered"
                with self.assertRaisesRegex(ValueError, "hashes differ"):
                    annotation.preflight(args)
                identity[key] = previous


if __name__ == "__main__":
    unittest.main()
