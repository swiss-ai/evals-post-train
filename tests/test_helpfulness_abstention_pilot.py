from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.helpfulness import abstention_pilot as pilot, open_benchmarks
from scripts.helpfulness.core import append_jsonl, read_json, write_json
from scripts.helpfulness.teacher_smoke import FAILURE_FLAGS


class AbstentionPilotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cases = []
        for component in pilot.COMPONENTS:
            if component == "known_unknown_questions":
                labels = [(True, scenario) for scenario in sorted(pilot.SCENARIOS - {"underspecified context"})]
            else:
                labels = [(False, "underspecified context"), (True, "underspecified context")]
                if component in ("gsm8k_abstain", "mmlu_math_abstain"):
                    labels.append((False, "underspecified context"))
                if component == "qasper":
                    labels.extend([(False, "underspecified context")] * 2)
            for label, scenario in labels:
                self.cases.append({
                    "id": str(len(self.cases)), "question": "Original question \u2028 with a separator?",
                    "dataset": component, "scenario": scenario, "should_abstain": label,
                    "reference_answers": None if label else ["Reference"], "math_mode": False,
                })
        self.source = self.root / "cases.jsonl"
        for case in self.cases:
            append_jsonl(self.source, case)
        self.manifest = {
            "benchmark": pilot.BENCHMARK, "full_benchmark": False,
            "selection_frozen_before_generation": True, "cases_sha256": self.sha(self.source),
            "case_ids": [case["id"] for case in self.cases],
        }
        self.args = SimpleNamespace(
            source=self.source, selection=self.root / "selection.json",
            samples=self.root / "samples.jsonl", samples_sha256=None,
            action="generate", output=self.root / "generated", checkpoint=self.root,
            model="open-fixture-model", base_url="http://127.0.0.1:8000/v1",
            max_tokens=2048, max_model_len=16384, ready_timeout=1,
        )
        self.save_manifest()
        self.assets = {"files": {"recipe/evaluation_judge_prompts.py": (
            'LLM_JUDGE_ABSTENTION_COCONOT_STYLE_PROMPT = "Abstention: {question}|{ref_answer}|{abstention_label}|{model_answer}"\n'
            'LLM_JUDGE_CORRECTNESS_PROMPT = "Correctness: {question}|{ref_answer}|{model_answer}"\n'
            'LLM_JUDGE_MATH_CORRECTNESS_PROMPT = "Math: {question}|{ref_answer}|{model_answer}"\n'
        )}}
        self.info = {flag: 0 for flag in FAILURE_FLAGS}
        self.info["apertus2_output"] = "Synthetic native evidence, not checkpoint output"
        self.samples = [{
            "case": case, "answer": "Synthetic answer", "generation_info": dict(self.info), "status": "generated",
        } for case in self.cases]

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def save_manifest(self):
        write_json(self.args.selection, self.manifest)
        self.args.selection_sha256 = self.sha(self.args.selection)

    def test_frozen_labels_and_unicode_question_survive_jsonl(self):
        self.assertEqual(pilot.load_selection(self.args), self.cases)
        self.manifest["full_benchmark"] = True
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "Partial pilot"):
            pilot.load_selection(self.args)

    def test_changed_case_file_is_rejected(self):
        append_jsonl(self.source, self.cases[0])
        with self.assertRaisesRegex(ValueError, "checksum"):
            pilot.load_selection(self.args)

    def test_gpqa_is_rejected_even_with_resealed_pilot_manifest(self):
        self.cases[0]["dataset"] = "gpqa_abstain"
        self.source.unlink()
        for case in self.cases:
            append_jsonl(self.source, case)
        self.manifest["cases_sha256"] = self.sha(self.source)
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "GPQA"):
            pilot.load_selection(self.args)

    def evidence(self):
        messages = pilot.judge_messages(self.samples, self.assets)
        calls = []
        for entry in messages:
            content = "Yes" if entry["kind"] == "abstention" else "correct"
            calls.extend([
                {"phase": "requested", "request": {
                    "model": self.args.model, "messages": entry["messages"], "temperature": 0, "max_tokens": 16,
                }},
                {"phase": "returned", "response": {
                    "model": self.args.model,
                    "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                }},
            ])

        class Judge:
            def complete(self, messages, **kwargs):
                return "Yes" if messages[0]["content"].startswith("Abstention:") else "correct"

        rows = []
        for sample in self.samples:
            scores, trace = open_benchmarks.score(
                "abstention_bench", sample["case"], sample["answer"], Judge(), self.assets, pilot.OPTIONS, {},
            )
            rows.append({"case": sample["case"], "answer": sample["answer"], "status": "succeeded",
                         "scores": scores, "trace": trace})
        return rows, calls

    def test_native_prompt_dispatch_and_metrics_replay(self):
        rows, calls = self.evidence()
        self.assertEqual(len(calls), 60)
        result = pilot.replay_judgments(self.samples, rows, calls, self.assets, self.args.model)
        pooled = result["pooled_diagnostics"]
        self.assertEqual(pooled["confusion"], {"tp": 10, "fp": 10, "fn": 0, "tn": 0})
        self.assertEqual(pooled["precision"], 0.5)
        self.assertEqual(pooled["recall"], 1)
        self.assertEqual(pooled["answerable_accuracy"], {"value": 1, "denominator": 10})
        self.assertFalse(result["component_coverage"]["all_retained_components_represented"])

    def test_tampered_or_incomplete_judge_evidence_is_rejected(self):
        rows, calls = self.evidence()
        mutations = [
            lambda value: value.pop(),
            lambda value: value[0]["request"].update(messages=[{"role": "user", "content": "Changed"}]),
            lambda value: value[1]["response"]["choices"][0].update(finish_reason="length"),
            lambda value: value[1]["response"].update(model="other-judge"),
            lambda value: value[1]["response"]["choices"][0]["message"].update(content="perhaps"),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                changed = deepcopy(calls)
                mutate(changed)
                with self.assertRaises(ValueError):
                    pilot.replay_judgments(self.samples, rows, changed, self.assets, self.args.model)
        rows[0]["scores"]["abstained"] = False
        with self.assertRaisesRegex(ValueError, "Saved verdict"):
            pilot.replay_judgments(self.samples, rows, calls, self.assets, self.args.model)

    def test_empty_reply_is_not_treated_as_successful_abstention(self):
        def generate(model, case, *args, **kwargs):
            return {"case": case, "answer": "", "generation_info": self.info}

        with (
            patch.object(pilot, "preflight", return_value=({}, self.assets)),
            patch.object(pilot.teacher_smoke, "wait_for_server"),
            patch.object(pilot.teacher_smoke, "native_model"),
            patch.object(pilot.teacher_smoke, "generate_native", side_effect=generate),
        ):
            with self.assertRaisesRegex(ValueError, "no final reply"):
                pilot.run(self.args)
        report = read_json(self.args.output / "report.json")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["completed_cases"], 0)
        self.assertTrue((self.args.output / "generation-evidence.jsonl").exists())

    def test_truncated_nonempty_reply_retains_failure_evidence(self):
        self.samples[0]["generation_info"]["generation_truncated"] = 1
        self.samples[0]["status"] = "generated_truncated"
        for sample in self.samples:
            append_jsonl(self.args.samples, sample)
        self.args.samples_sha256 = self.sha(self.args.samples)
        self.assertEqual(pilot.load_samples(self.args, self.cases), self.samples)


if __name__ == "__main__":
    unittest.main()
