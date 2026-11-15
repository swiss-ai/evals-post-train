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
            output=self.root / "run", ready_timeout=1,
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

    def test_model_identity_mismatch_is_rejected(self):
        self.args.model = "different-model"
        with self.assertRaisesRegex(ValueError, "not bound"):
            smoke.preflight(self.args)


if __name__ == "__main__":
    unittest.main()
