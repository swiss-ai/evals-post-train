from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.helpfulness import abstention_pilot as pilot, open_benchmarks
from scripts.helpfulness.core import append_jsonl, read_json, read_jsonl, write_json
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
            if sample["status"] == "model_no_final_reply":
                rows.append(pilot.failed_judgment(sample))
                continue
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

    def budget_failure(self, index):
        sample = deepcopy(self.samples[index])
        sample["answer"] = ""
        sample["generation_info"].update(generation_truncated=1, generation_think_truncated=1)
        sample["status"] = "model_no_final_reply"
        return sample

    def generate_with(self, function):
        with (
            patch.object(pilot, "preflight", return_value=({}, self.assets)),
            patch.object(pilot.teacher_smoke, "wait_for_server"),
            patch.object(pilot.teacher_smoke, "native_model"),
            patch.object(pilot.teacher_smoke, "generate_native", side_effect=function),
        ):
            return pilot.run(self.args)

    def test_budget_failure_is_recorded_and_later_cases_continue(self):
        self.args.continue_model_failures = True
        attempted = []

        def generate(model, case, history, task, index, decoding):
            attempted.append(index)
            return self.budget_failure(index) if index == 0 else deepcopy(self.samples[index])

        report = self.generate_with(generate)
        self.assertEqual(attempted, list(range(20)))
        self.assertEqual(report["status"], "succeeded_with_model_failures")
        self.assertEqual(report["recorded_cases"], 20)
        self.assertEqual(report["completed_cases"], 19)
        self.assertEqual(report["model_failure_cases"], 1)
        self.assertEqual(read_jsonl(self.args.output / "samples.jsonl")[0]["status"], "model_no_final_reply")

    def test_transport_failure_still_stops_even_when_continuation_enabled(self):
        self.args.continue_model_failures = True
        bad = self.budget_failure(0)
        bad["generation_info"]["generation_request_failed"] = 1
        with self.assertRaises(ValueError):
            self.generate_with(lambda *args: bad)
        self.assertEqual(read_json(self.args.output / "report.json")["status"], "failed")

    def prepare_resume(self):
        self.args.continue_model_failures = True
        for name in ("samples", "evidence", "report"):
            setattr(self.args, "resume_" + name, self.root / ("resume-" + name + ".jsonl"))
        for sample in self.samples[:12]:
            append_jsonl(self.args.resume_samples, sample)
        for sample in [*self.samples[:12], self.budget_failure(12)]:
            append_jsonl(self.args.resume_evidence, {**sample, "status": "generated_unvalidated"})
        write_json(self.args.resume_report, {
            "status": "failed", "benchmark": pilot.BENCHMARK,
            "model": self.args.model, "base_url": self.args.base_url,
            "selection_sha256": self.args.selection_sha256, "target_max_tokens": 2048,
            "preflight": {}, "completed_cases": 12,
        })
        for name in ("samples", "evidence", "report"):
            setattr(self.args, "resume_" + name + "_sha256", self.sha(getattr(self.args, "resume_" + name)))

    def test_resume_preserves_bytes_and_never_reruns_attempted_cases(self):
        self.prepare_resume()
        attempted = []

        def generate(model, case, history, task, index, decoding):
            attempted.append(index)
            return deepcopy(self.samples[index])

        report = self.generate_with(generate)
        self.assertEqual(attempted, list(range(13, 20)))
        self.assertEqual(report["resumed_cases"], 13)
        self.assertEqual(report["new_generation_attempts"], 7)
        self.assertEqual(report["completed_cases"], 19)
        self.assertEqual(report["model_failure_cases"], 1)
        for source, output in (
            (self.args.resume_samples, self.args.output / "samples.jsonl"),
            (self.args.resume_evidence, self.args.output / "generation-evidence.jsonl"),
        ):
            self.assertTrue(output.read_bytes().startswith(source.read_bytes()))

    def test_resume_rejects_resealed_raw_response_that_disagrees_with_samples(self):
        self.prepare_resume()
        rows = read_jsonl(self.args.resume_evidence)
        rows[0]["answer"] = "Tampered answer"
        self.args.resume_evidence.unlink()
        for row in rows:
            append_jsonl(self.args.resume_evidence, row)
        self.args.resume_evidence_sha256 = self.sha(self.args.resume_evidence)
        with self.assertRaisesRegex(ValueError, "raw evidence differs"):
            pilot.resume_data(self.args, self.cases, {})

    def test_failed_answerable_case_gets_no_abstention_credit_and_stays_in_failure_adjusted_accuracy(self):
        self.samples[0] = self.budget_failure(0)
        rows, calls = self.evidence()
        result = pilot.replay_judgments(self.samples, rows, calls, self.assets, self.args.model)
        self.assertEqual(len(calls), 56)
        self.assertEqual(rows[0]["scores"], {})
        self.assertEqual(result["scored_cases"], 19)
        self.assertEqual(result["failed_cases"], 1)
        self.assertEqual(result["pooled_diagnostics"]["abstention_denominator"], 19)
        self.assertEqual(result["pooled_diagnostics"]["answerable_accuracy"], {"value": 1, "denominator": 9})
        self.assertEqual(result["pooled_diagnostics"]["selected_answerable_accuracy_failures_incorrect"],
                         {"value": 0.9, "denominator": 10})
        rows[0] = {**rows[0], "status": "succeeded", "scores": {"abstained": True, "correct": False}}
        with self.assertRaisesRegex(ValueError, "credited as an abstention"):
            pilot.replay_judgments(self.samples, rows, calls, self.assets, self.args.model)


if __name__ == "__main__":
    unittest.main()
