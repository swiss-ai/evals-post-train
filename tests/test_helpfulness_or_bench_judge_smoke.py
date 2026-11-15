import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.helpfulness import or_bench_judge_smoke as smoke
from scripts.helpfulness.teacher_smoke import FAILURE_FLAGS


class SemanticJudgeSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        info = {flag: 0 for flag in FAILURE_FLAGS}
        info["apertus2_output"] = "Native fixture, not checkpoint output"
        self.samples = self.root / "samples.jsonl"
        rows = [
            {"case": {"id": str(i), "question": "Fixture request"},
             "answer": "I decline that part. Here is the requested remainder.",
             "generation_info": info, "scores": {"refused": i == 0}, "status": "succeeded"}
            for i in range(3)
        ]
        self.samples.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        self.identity = self.root / "identity.json"
        self.identity.write_text(json.dumps({
            "served_model": "gemma-fixture", "base_url": "http://127.0.0.1:8000/v1",
            "public_model": "google/gemma-4-31B-it", "license": "Apache-2.0",
            "public_repository_revision": "fixture-not-real-provenance",
        }), encoding="utf-8")
        self.args = SimpleNamespace(
            samples=self.samples, samples_sha256=hashlib.sha256(self.samples.read_bytes()).hexdigest(),
            identity=self.identity, model="gemma-fixture", base_url="http://127.0.0.1:8000/v1",
            output=self.root / "run", ready_timeout=1, expected_cases=3, rubric_version="v1",
        )

    def verdict(self, label="partial_refusal", quote="I decline that part."):
        return json.dumps({"label": label, "reason": "Declines only the first part.",
                           "evidence_quote": quote})

    def test_partial_refusal_remains_distinct_and_requires_verbatim_evidence(self):
        value = smoke.parse_verdict(self.verdict(), "I decline that part. Here is the rest.")
        self.assertEqual(value["label"], "partial_refusal")
        with self.assertRaisesRegex(ValueError, "exact nonempty"):
            smoke.parse_verdict(self.verdict(quote="I cannot help."), "I decline that part.")
        with self.assertRaisesRegex(ValueError, "unknown"):
            smoke.parse_verdict(self.verdict(label="safe"), "I decline that part.")
        with self.assertRaisesRegex(ValueError, "exactly"):
            smoke.parse_verdict('{"label":"refusal"}', "I decline that part.")

    def test_changed_source_is_rejected_before_judging(self):
        self.samples.write_text(self.samples.read_text() + "\n", encoding="utf-8")
        with patch.object(smoke, "ChatClient") as client:
            with self.assertRaisesRegex(ValueError, "checksum"):
                smoke.run(self.args)
            client.assert_not_called()
        self.assertEqual(json.loads((self.args.output / "report.json").read_text())["status"], "failed")

    def test_fenced_json_is_accepted_without_accepting_surrounding_prose(self):
        answer = "I decline that part."
        for fence in ("```json", "```"):
            with self.subTest(fence=fence):
                value = smoke.parse_verdict(fence + "\n" + self.verdict() + "\n```", answer)
                self.assertEqual(value["label"], "partial_refusal")
        with self.assertRaises(json.JSONDecodeError):
            smoke.parse_verdict("Here is my verdict:\n```json\n" + self.verdict() + "\n```", answer)
        with self.assertRaisesRegex(ValueError, "exact nonempty"):
            smoke.parse_verdict("```json\n" + self.verdict(quote="Invented quote") + "\n```", answer)

    def test_invalid_verdict_preserves_raw_output_and_fails_without_counting_it(self):
        with patch.object(smoke, "wait_for_server"), patch.object(smoke, "ChatClient") as client:
            client.return_value.judge.return_value = self.verdict(quote="Invented quote")
            with self.assertRaisesRegex(ValueError, "exact nonempty"):
                smoke.run(self.args)
        calls = (self.args.output / "judge-calls.jsonl").read_text()
        self.assertIn("Invented quote", calls)
        report = json.loads((self.args.output / "report.json").read_text())
        self.assertEqual((report["status"], report["judged_cases"]), ("failed", 0))
        self.assertFalse((self.args.output / "verdicts.jsonl").exists())

    def test_unclear_and_partial_outcomes_are_not_silently_folded_into_compliance(self):
        with patch.object(smoke, "wait_for_server"), patch.object(smoke, "ChatClient") as client:
            client.return_value.judge.side_effect = [
                self.verdict(label=label) for label in ("refusal", "partial_refusal", "unclear")
            ]
            report = smoke.run(self.args)
        self.assertEqual(report["label_counts"], {
            "refusal": 1, "partial_refusal": 1, "compliance": 0, "unclear": 1,
        })
        self.assertEqual(report["semantic_full_refusal_rate"]["value"], 1 / 3)
        self.assertEqual(report["semantic_any_refusal_rate"]["value"], 2 / 3)
        self.assertEqual(report["original_prefix_refusal_rate"]["value"], 1 / 3)
        self.assertFalse(report["target_regenerated"])
        self.assertFalse(report["published_protocol_equivalence"])
        self.assertEqual(report["calibration"], "not_performed")
        self.assertEqual(report["protocol"], "adapted-or-bench-semantic-refusal-v1")
        self.assertEqual(report["rubric_sha256"], smoke.digest(smoke.RUBRIC))
        client.return_value.judge.assert_called_with(
            smoke.RUBRIC,
            {"question": "Fixture request",
             "assistant_answer": "I decline that part. Here is the requested remainder."},
            max_tokens=2048,
        )

    def test_revised_rubric_is_explicit_and_recorded_without_claiming_validation(self):
        self.args.rubric_version = "v2"
        with patch.object(smoke, "wait_for_server"), patch.object(smoke, "ChatClient") as client:
            client.return_value.judge.return_value = self.verdict()
            report = smoke.run(self.args)
        self.assertEqual(report["protocol"], "adapted-or-bench-semantic-refusal-v2")
        self.assertEqual(report["rubric_sha256"], smoke.digest(smoke.RUBRIC_V2))
        self.assertEqual(report["calibration"], "rubric_revised_not_validated")
        self.assertFalse(report["independent_judge_validation"])
        self.assertFalse(report["target_regenerated"])
        self.assertEqual(report["selected_cases"], 3)
        rubric = json.loads((self.args.output / "rubric.json").read_text())
        self.assertEqual(rubric["instruction"], smoke.RUBRIC_V2)
        calls = [json.loads(line) for line in
                 (self.args.output / "judge-calls.jsonl").read_text().splitlines()]
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(call["instruction"] == smoke.RUBRIC_V2 for call in calls))
        client.return_value.judge.assert_called_with(
            smoke.RUBRIC_V2,
            {"question": "Fixture request",
             "assistant_answer": "I decline that part. Here is the requested remainder."},
            max_tokens=2048,
        )

    def test_preflight_binds_explicit_revised_rubric_without_inference(self):
        output = self.root / "preflight.json"
        with patch.object(smoke, "ChatClient") as client:
            smoke.main([
                "preflight", "--samples", str(self.samples),
                "--samples-sha256", self.args.samples_sha256,
                "--identity", str(self.identity), "--model", self.args.model,
                "--rubric-version", "v2", "--output", str(output),
            ])
        client.assert_not_called()
        report = json.loads(output.read_text())
        self.assertEqual(report["rubric_version"], "v2")
        self.assertEqual(report["rubric_sha256"], smoke.digest(smoke.RUBRIC_V2))
        self.assertFalse(report["inference_performed"])

    def test_model_identity_mismatch_is_rejected(self):
        self.args.model = "different-model"
        with self.assertRaisesRegex(ValueError, "not bound"):
            smoke.preflight(self.args)

    def structured(self, label="partial_refusal", fulfillment="substantive", withholding=True):
        return {
            "requested_task": "Fixture request", "constraints": ["Fixture request"],
            "withholding": {"requested_part": "Declined part", "quote": "I decline that part."}
            if withholding else None,
            "fulfillment": fulfillment,
            "attempt": {"requested_part": "Requested remainder", "quote": "Here is the requested remainder."}
            if fulfillment == "substantive" else None,
            "label": label, "reason": "Evidence-linked classification.",
            "evidence_quote": "I decline that part.",
        }

    def test_structured_labels_must_follow_grounded_component_table(self):
        answer = "I decline that part. Here is the requested remainder."
        for label, fulfillment, withholding in (
                ("refusal", "none", True), ("partial_refusal", "substantive", True),
                ("compliance", "substantive", False), ("unclear", "none", False),
                ("unclear", "ambiguous", True), ("unclear", "ambiguous", False)):
            with self.subTest(label=label, fulfillment=fulfillment, withholding=withholding):
                value = self.structured(label, fulfillment, withholding)
                self.assertEqual(smoke.parse_verdict(
                    json.dumps(value), answer, rubric_version="v3", question="Fixture request"
                ), value)
        value = self.structured(label="refusal", fulfillment="substantive", withholding=False)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            smoke.parse_verdict(json.dumps(value), answer, rubric_version="v3", question="Fixture request")

    def test_structured_quotes_are_bound_to_the_correct_source_and_presence(self):
        answer = "I decline that part. Here is the requested remainder."
        for mutation, error in (
                ({"constraints": ["Invented constraint"]}, "question excerpts"),
                ({"constraints": "Fixture request"}, "question excerpts"),
                ({"withholding": {"requested_part": "Part", "quote": "Not in answer"}}, "exact answer quote"),
                ({"attempt": {"requested_part": "", "quote": "Here is the requested remainder."}}, "requested part"),
                ({"attempt": None}, "required exactly"),
                ({"fulfillment": "none"}, "required exactly"),
                ({"requested_task": ""}, "requested task")):
            with self.subTest(mutation=mutation):
                with self.assertRaisesRegex(ValueError, error):
                    smoke.parse_verdict(json.dumps({**self.structured(), **mutation}), answer,
                                        rubric_version="v3", question="Fixture request")

    def test_structured_run_preserves_both_evidence_passages_and_separate_protocol(self):
        self.args.rubric_version = "v3"
        value = self.structured()
        with patch.object(smoke, "wait_for_server"), patch.object(smoke, "ChatClient") as client:
            client.return_value.judge.return_value = json.dumps(value)
            report = smoke.run(self.args)
        self.assertEqual(report["protocol"], "adapted-or-bench-semantic-refusal-v3")
        self.assertEqual(report["label_counts"]["partial_refusal"], 3)
        self.assertEqual(report["outcomes"][0]["withholding"], value["withholding"])
        self.assertEqual(report["outcomes"][0]["attempt"], value["attempt"])
        self.assertFalse(report["independent_judge_validation"])
        self.assertEqual(report["rubric_sha256"], smoke.digest(smoke.RUBRIC_V3))
        client.return_value.judge.assert_called_with(
            smoke.RUBRIC_V3, {"question": "Fixture request",
                             "assistant_answer": "I decline that part. Here is the requested remainder."},
            max_tokens=2048,
        )

    def test_inconsistent_structured_output_fails_without_silently_relabeling(self):
        self.args.rubric_version = "v3"
        value = self.structured(label="refusal")
        with patch.object(smoke, "wait_for_server"), patch.object(smoke, "ChatClient") as client:
            client.return_value.judge.return_value = json.dumps(value)
            with self.assertRaisesRegex(ValueError, "conflicts"):
                smoke.run(self.args)
        self.assertIn('"label": "refusal"', json.loads(
            (self.args.output / "judge-calls.jsonl").read_text().splitlines()[0])["raw_judge_output"])
        self.assertFalse((self.args.output / "verdicts.jsonl").exists())
        report = json.loads((self.args.output / "report.json").read_text())
        self.assertEqual((report["status"], report["judged_cases"]), ("failed", 0))

    def line_verdict(self):
        return {
            "requested_task": "Fixture request", "constraint_lines": [1],
            "withholding": {"requested_part": "Declined part", "start_line": 1, "end_line": 1},
            "fulfillment": "substantive",
            "attempt": {"requested_part": "Requested remainder", "start_line": 1, "end_line": 1},
            "label": "partial_refusal", "reason": "Both withholding and requested help.",
            "evidence_lines": {"start_line": 1, "end_line": 1},
        }

    def test_line_selection_recovers_exact_contiguous_source_without_generated_quotes(self):
        answer = "I decline that part.\r\nHere is the requested remainder.\r\nThird line."
        value = self.line_verdict()
        value["attempt"].update(start_line=2, end_line=3)
        parsed = smoke.parse_verdict(json.dumps(value), answer, rubric_version="v3-lines",
                                     question="Fixture request")
        self.assertEqual(parsed["withholding"]["quote"], "I decline that part.")
        self.assertEqual(parsed["attempt"]["quote"], "Here is the requested remainder.\r\nThird line.")
        self.assertEqual(parsed["constraints"], ["Fixture request"])
        self.assertEqual(parsed["source_line_spans"]["attempt"], {"start_line": 2, "end_line": 3})
        self.assertEqual(smoke.judge_evidence("Fixture request", "First\n\nThird", "v3-lines"),
                         {"question": "Q1: Fixture request", "assistant_answer": "A1: First\nA2: \nA3: Third"})

    def test_invalid_line_selection_and_label_conflicts_are_not_repaired(self):
        answer = "I decline that part. Here is the requested remainder."
        for start, end in ((0, 1), (1, 2), (True, 1), ("1", 1), (2, 1)):
            with self.subTest(start=start, end=end):
                value = self.line_verdict()
                value["evidence_lines"] = {"start_line": start, "end_line": end}
                with self.assertRaisesRegex(ValueError, "valid contiguous integer"):
                    smoke.parse_verdict(json.dumps(value), answer, rubric_version="v3-lines",
                                        question="Fixture request")
        value = self.line_verdict()
        value["label"] = "refusal"
        with self.assertRaisesRegex(ValueError, "conflicts"):
            smoke.parse_verdict(json.dumps(value), answer, rubric_version="v3-lines", question="Fixture request")

    def test_line_selected_run_records_original_raw_selection_and_resolved_source_quotes(self):
        self.args.rubric_version = "v3-lines"
        raw = json.dumps(self.line_verdict())
        with patch.object(smoke, "wait_for_server"), patch.object(smoke, "ChatClient") as client:
            client.return_value.judge.return_value = raw
            report = smoke.run(self.args)
        self.assertEqual(report["protocol"], "adapted-or-bench-semantic-refusal-v3-lines")
        self.assertEqual(report["label_counts"]["partial_refusal"], 3)
        self.assertEqual(report["outcomes"][0]["source_line_spans"]["constraints"], [1])
        self.assertEqual(report["outcomes"][0]["attempt"]["quote"],
                         "I decline that part. Here is the requested remainder.")
        calls = [json.loads(line) for line in (self.args.output / "judge-calls.jsonl").read_text().splitlines()]
        self.assertEqual(calls[0]["raw_judge_output"], raw)
        self.assertEqual(calls[0]["evidence"]["question"], "Q1: Fixture request")
        self.assertTrue(calls[0]["evidence"]["assistant_answer"].startswith("A1: "))
        self.assertFalse(report["independent_judge_validation"])

    def test_twenty_case_pilot_uses_explicit_count_and_correct_denominators(self):
        sample = json.loads(self.samples.read_text(encoding="utf-8").splitlines()[0])
        rows = [{**sample, "case": {**sample["case"], "id": str(i)},
                 "scores": {"refused": i < 4}} for i in range(20)]
        self.samples.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        self.args.samples_sha256 = hashlib.sha256(self.samples.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "exactly 3"):
            smoke.preflight(self.args)
        self.args.expected_cases = 20
        with patch.object(smoke, "wait_for_server"), patch.object(smoke, "ChatClient") as client:
            client.return_value.judge.side_effect = [
                self.verdict(label="refusal" if i < 5 else "compliance") for i in range(20)
            ]
            report = smoke.run(self.args)
        self.assertEqual(report["judged_cases"], 20)
        self.assertEqual(report["semantic_full_refusal_rate"], {"value": .25, "denominator": 20})
        self.assertEqual(report["original_prefix_refusal_rate"], {"value": .2, "denominator": 20})
        self.assertEqual(len(report["outcomes"]), 20)

    def test_duplicate_or_incomplete_frozen_samples_cannot_satisfy_requested_count(self):
        rows = [json.loads(line) for line in self.samples.read_text(encoding="utf-8").splitlines()]
        rows[1]["case"]["id"] = rows[0]["case"]["id"]
        self.samples.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        expected = hashlib.sha256(self.samples.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "distinct"):
            smoke.load_samples(self.samples, expected, 3)
        with self.assertRaisesRegex(ValueError, "exactly 20"):
            smoke.load_samples(self.samples, expected, 20)


if __name__ == "__main__":
    unittest.main()
