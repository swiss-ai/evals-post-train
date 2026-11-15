from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.helpfulness import benchmark_pilot as pilot
from scripts.helpfulness.core import digest, write_json
from scripts.helpfulness.teacher_smoke import FAILURE_FLAGS


class BenchmarkPilotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        docs = [
            {"instruction": f"Fixture prompt {i}", "output": f"Released reference {i}", "dataset": "fixture"}
            for i in range(25)
        ]
        source = self.root / "source.json"
        write_json(source, docs)
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        patcher = patch.object(pilot, "ALPACA_SOURCE_SHA256", source_hash)
        patcher.start()
        self.addCleanup(patcher.stop)
        ids = sorted(
            (digest(doc) for doc in docs),
            key=lambda key: hashlib.sha256((pilot.SELECTION_SEED + key).encode()).hexdigest(),
        )[:20]
        selection = self.root / "selection.json"
        write_json(selection, {
            "benchmark": "alpaca_eval_2_lc", "source_sha256": source_hash, "case_ids": ids,
        })
        self.args = SimpleNamespace(
            action="generate", source=source, selection=selection,
            selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(),
            expected_cases=20, output=self.root / "generation", checkpoint=self.root,
            model="chonk-fixture", base_url="http://127.0.0.1:8000/v1",
            ready_timeout=1, max_tokens=128, max_model_len=16384,
            allow_truncated=False, resume_samples=None, resume_samples_sha256=None,
            resume_report=None, resume_report_sha256=None,
        )
        self.info = {flag: 0 for flag in FAILURE_FLAGS}
        self.info["apertus2_output"] = "Native fixture, not real checkpoint inference"

    def record(self, model, case, messages, task, index, decoding):
        return {
            "case": case, "answer": f"Candidate answer {index}",
            "generation_info": dict(self.info), "status": "generated_unvalidated",
        }

    def generate(self, side_effect=None):
        with (
            patch.object(pilot.teacher_smoke, "preflight", return_value={"fixture": True}),
            patch.object(pilot.teacher_smoke, "wait_for_server"),
            patch.object(pilot.teacher_smoke, "native_model"),
            patch.object(pilot.teacher_smoke, "generate_native", side_effect=side_effect or self.record) as generate,
        ):
            result = pilot.run(self.args)
        return result, generate

    def judge_args(self):
        source = self.args.output / "samples.jsonl"
        self.args.action = "judge"
        self.args.samples = source
        self.args.samples_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        self.args.output = self.root / "judgment"
        self.args.scorer_assets = self.root / "assets"
        self.args.identity = self.root / "identity.json"
        self.args.model = "gemma-fixture"
        write_json(self.args.identity, {
            "served_model": self.args.model, "base_url": self.args.base_url,
            "public_model": "google/gemma-4-31B-it", "license": "Apache-2.0",
            "public_repository_revision": "fixture-only",
        })
        self.args.scorer_assets.mkdir()
        features = self.args.scorer_assets / "df_gamed.csv"
        features.write_text("fixture,not_real_features\n", encoding="utf-8")
        patcher = patch.object(pilot, "ALPACA_FEATURES_SHA256", hashlib.sha256(features.read_bytes()).hexdigest())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_selection_rejects_reordering_or_changed_originals_before_inference(self):
        cases = pilot.load_selection(self.args)
        self.assertEqual(len(cases), 20)
        value = json.loads(self.args.selection.read_text())
        value["case_ids"].reverse()
        write_json(self.args.selection, value)
        self.args.selection_sha256 = hashlib.sha256(self.args.selection.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "deterministic"):
            pilot.load_selection(self.args)
        self.args.source.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            pilot.load_selection(self.args)

    def test_generation_preserves_references_and_native_evidence_without_fake_scores(self):
        result, generate = self.generate()
        rows = pilot.read_jsonl(self.args.output / "samples.jsonl")
        self.assertEqual(result["completed_cases"], 20)
        self.assertEqual([row["case"] for row in rows], pilot.load_selection(self.args))
        self.assertTrue(all(row["status"] == "generated" and "scores" not in row for row in rows))
        self.assertTrue(all(row["generation_info"] == self.info for row in rows))
        call = generate.call_args
        self.assertEqual(call.args[2], [{"role": "user", "content": rows[-1]["case"]["question"]}])
        self.assertEqual(call.args[-1], {"temperature": 0, "max_gen_toks": 128, "until": []})
        self.assertFalse(result["published_protocol_equivalence"])

    def test_native_failure_is_retained_and_cannot_become_a_successful_sample(self):
        def faulty(*args):
            row = self.record(*args)
            if args[4] == 1:
                row["generation_info"]["generation_truncated"] = 1
            return row

        with self.assertRaisesRegex(ValueError, "generation_truncated"):
            self.generate(faulty)
        self.assertEqual(len(pilot.read_jsonl(self.args.output / "generation-evidence.jsonl")), 2)
        self.assertEqual(len(pilot.read_jsonl(self.args.output / "samples.jsonl")), 1)
        self.assertEqual(pilot.read_json(self.args.output / "report.json")["status"], "failed")

    def test_judge_preflight_checks_all_frozen_answers_and_assets_without_inference(self):
        self.generate()
        self.judge_args()
        with (
            patch.object(pilot.native, "require_harness"),
            patch.object(pilot.importlib, "import_module"),
        ):
            samples, identity = pilot.judge_preflight(self.args, pilot.load_selection(self.args))
        self.assertEqual(len(samples), 20)
        self.assertEqual(identity["served_model"], self.args.model)
        rows = list(reversed(pilot.read_jsonl(self.args.samples)))
        self.args.samples.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        self.args.samples_sha256 = hashlib.sha256(self.args.samples.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "every frozen case"):
            pilot.load_samples(self.args, pilot.load_selection(self.args))

    def test_explicit_truncation_policy_preserves_flags_and_finishes_other_cases(self):
        self.args.allow_truncated = True

        def truncated(*args):
            row = self.record(*args)
            if args[4] == 1:
                row["generation_info"]["generation_truncated"] = 1
            return row

        result, generate = self.generate(truncated)
        self.assertEqual(generate.call_count, 20)
        self.assertEqual(result["truncated_cases"], 1)
        self.args.samples = self.args.output / "samples.jsonl"
        self.args.samples_sha256 = hashlib.sha256(self.args.samples.read_bytes()).hexdigest()
        rows = pilot.load_samples(self.args, pilot.load_selection(self.args), allow_truncated=True)
        self.assertEqual(rows[1]["status"], "generated_truncated")
        self.assertEqual(rows[1]["generation_info"]["generation_truncated"], 1)
        with self.assertRaisesRegex(ValueError, "generation_truncated"):
            pilot.load_samples(self.args, pilot.load_selection(self.args))
        self.judge_args()
        with (
            patch.object(pilot.native, "require_harness"),
            patch.object(pilot.importlib, "import_module"),
        ):
            judged, _ = pilot.judge_preflight(self.args, pilot.load_selection(self.args))
        self.assertEqual(len(judged), 20)
        self.assertEqual(judged[1]["status"], "generated_truncated")

    def test_allowing_truncation_does_not_hide_parse_or_transport_failures(self):
        self.args.allow_truncated = True

        def invalid(*args):
            row = self.record(*args)
            row["generation_info"].update(generation_truncated=1, generation_parse_failed=1)
            return row

        with self.assertRaisesRegex(ValueError, "generation_parse_failed"):
            self.generate(invalid)
        self.assertEqual(len(pilot.read_jsonl(self.args.output / "generation-evidence.jsonl")), 1)
        self.assertFalse((self.args.output / "samples.jsonl").exists())

    def prepare_resume(self):
        def faulty(*args):
            row = self.record(*args)
            if args[4] == 9:
                row["generation_info"]["generation_truncated"] = 1
            return row

        with self.assertRaisesRegex(ValueError, "generation_truncated"):
            self.generate(faulty)
        previous = self.args.output
        self.args.resume_samples = previous / "samples.jsonl"
        self.args.resume_samples_sha256 = hashlib.sha256(self.args.resume_samples.read_bytes()).hexdigest()
        self.args.resume_report = previous / "report.json"
        self.args.resume_report_sha256 = hashlib.sha256(self.args.resume_report.read_bytes()).hexdigest()
        self.args.output = self.root / "resumed-generation"
        self.args.max_tokens = 64
        self.args.allow_truncated = True
        return pilot.read_jsonl(self.args.resume_samples)

    def test_resume_reuses_nine_answers_and_only_generates_the_remaining_eleven(self):
        saved = self.prepare_resume()
        result, generate = self.generate()
        self.args.samples = self.args.output / "samples.jsonl"
        self.args.samples_sha256 = hashlib.sha256(self.args.samples.read_bytes()).hexdigest()
        rows = pilot.load_samples(self.args, pilot.load_selection(self.args), allow_truncated=True)
        self.assertEqual(rows[:9], saved)
        self.assertEqual(generate.call_count, 11)
        self.assertEqual(generate.call_args_list[0].args[4], 9)
        self.assertEqual(result["resumed_cases"], 9)
        self.assertEqual(result["new_generations"], 11)
        self.assertEqual([row["decoding"]["max_gen_toks"] for row in result["decoding_segments"]], [128, 64])
        self.assertEqual(result["resume_source"]["samples_sha256"], self.args.resume_samples_sha256)
        self.assertEqual(len(pilot.read_jsonl(self.args.output / "generation-evidence.jsonl")), 11)

    def test_resume_rejects_changed_checkpoint_metadata_and_changed_sample_content(self):
        self.prepare_resume()
        preflight = {"fixture": "different checkpoint"}
        with self.assertRaisesRegex(ValueError, "checkpoint and frozen"):
            pilot.resume_samples(self.args, pilot.load_selection(self.args), preflight)
        self.args.resume_samples.write_text("[]\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            pilot.resume_samples(self.args, pilot.load_selection(self.args), {"fixture": True})

    def test_native_lc_scoring_reuses_saved_answers_and_preserves_a_failed_scorer(self):
        self.generate()
        self.judge_args()
        seen = []

        def score(items, config, output):
            seen.extend(items)
            write_json(output / "validated_metrics.json", {
                "raw_win_rate": 0.4, "length_controlled_winrate": 0.35, "judged_cases": 20,
            })

        with (
            patch.object(pilot.native, "require_harness"),
            patch.object(pilot.importlib, "import_module"),
            patch.object(pilot.teacher_smoke, "wait_for_server"),
            patch.object(pilot.native, "offline_alpaca_assets", return_value=nullcontext()),
            patch.object(pilot, "record_sdk_calls", return_value=nullcontext()),
            patch.object(pilot, "judge_messages", return_value=[]),
            patch.object(pilot.native, "alpaca_aggregation", side_effect=score),
            patch.object(pilot.teacher_smoke, "native_model") as target,
        ):
            result = pilot.run(self.args)
        target.assert_not_called()
        self.assertEqual(result["metrics"]["length_controlled_winrate"], 0.35)
        self.assertFalse(result["target_regenerated"])
        self.assertFalse(result["reasoning_available"])
        self.assertEqual(seen[0]["reference_output"], pilot.load_selection(self.args)[0]["reference_answer"])
        self.assertEqual([item["index"] for item in seen],
                         [case["source_index"] for case in pilot.load_selection(self.args)])
        self.args.output = self.root / "failed-judge"
        with (
            patch.object(pilot, "judge_preflight", side_effect=ValueError("Invalid native scorer")),
            self.assertRaisesRegex(ValueError, "Invalid native scorer"),
        ):
            pilot.run(self.args)
        failed = pilot.read_json(self.args.output / "report.json")
        self.assertEqual(failed["status"], "failed")
        self.assertNotIn("metrics", failed)

    def test_sdk_recorder_blocks_undeclared_models_and_preserves_raw_logprobs(self):
        calls = []

        class Resource:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(model_dump=lambda **kw: {"logprobs": "fixture"})

        modules = {
            "openai.resources.chat.completions": SimpleNamespace(Completions=Resource),
            "openai.resources.completions": SimpleNamespace(Completions=type("OtherResource", (), {"create": Resource.create})),
        }
        resource = Resource()
        resource._client = SimpleNamespace(base_url=self.args.base_url + "/")
        path = self.root / "calls.jsonl"
        messages = [{"role": "user", "content": "Frozen native prompt fixture"}]
        with (
            patch.dict("sys.modules", modules),
            pilot.record_sdk_calls(path, self.args.base_url, "gemma-fixture", [{"messages": messages}]),
        ):
            resource.create(model="gemma-fixture", logprobs=True, messages=messages)
            with self.assertRaisesRegex(ValueError, "undeclared"):
                resource.create(model="unselected-model")
            with self.assertRaisesRegex(ValueError, "tokenized preflight"):
                resource.create(model="gemma-fixture", messages=[])
            resource._client.base_url = "https://example.invalid/v1"
            with self.assertRaisesRegex(ValueError, "undeclared"):
                resource.create(model="gemma-fixture")
        self.assertEqual(len(calls), 1)
        rows = pilot.read_jsonl(path)
        self.assertEqual([row["phase"] for row in rows], ["requested", "returned"])
        self.assertEqual(rows[1]["response"]["logprobs"], "fixture")


if __name__ == "__main__":
    unittest.main()
