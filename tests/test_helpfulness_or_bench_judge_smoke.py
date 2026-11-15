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
