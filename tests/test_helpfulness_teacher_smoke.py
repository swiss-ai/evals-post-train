import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from scripts.helpfulness import teacher_smoke


class TeacherSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def checkpoint(self, shard="model-00001.safetensors"):
        files = {
            "config.json": {"architectures": ["Apertus2KDAForCausalLM"]},
            "model.safetensors.index.json": {"weight_map": {"layer.weight": shard}},
            "tokenizer_config.json": {},
            "tokenizer.json": {},
            "generation_config.json": {"eos_token_id": [12, 2]},
        }
        for name, value in files.items():
            (self.root / name).write_text(json.dumps(value), encoding="utf-8")
        (self.root / "configuration_apertus2.py").write_text("# fixture\n", encoding="utf-8")
        return self.root

    def test_inventory_does_not_claim_weight_integrity_or_model_loading(self):
        directory = self.checkpoint()
        (directory / "model-00001.safetensors").write_bytes(b"fixture-not-real-weights")
        result = teacher_smoke.checkpoint_inventory(directory)
        self.assertEqual(result["shards"], 1)
        self.assertFalse(result["weight_contents_hashed"])
        self.assertFalse(result["chat_template_present"])
        self.assertNotIn("weights_loaded", result)

    def test_missing_and_unsafe_shards_fail(self):
        directory = self.checkpoint()
        with self.assertRaisesRegex(ValueError, "Missing or empty"):
            teacher_smoke.checkpoint_inventory(directory)
        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"layer": "../outside.safetensors"}}), encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "unsafe shard"):
            teacher_smoke.checkpoint_inventory(directory)

    def test_parse_failure_or_truncation_cannot_pass_even_with_nonempty_reply(self):
        info = {flag: 0 for flag in teacher_smoke.FAILURE_FLAGS}
        info["apertus2_output"] = "native fixture"
        teacher_smoke.validate_generation("Reply", info)
        for flag in ("generation_parse_failed", "generation_truncated", "generation_prompt_mismatch"):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, flag):
                teacher_smoke.validate_generation("Reply", {**info, flag: 1})
        with self.assertRaisesRegex(ValueError, "required failure evidence"):
            teacher_smoke.validate_generation("Reply", {})

    def test_no_remote_inference_endpoint_is_accepted(self):
        with patch.object(teacher_smoke, "urlopen") as request:
            with self.assertRaisesRegex(ValueError, "loopback"):
                teacher_smoke.wait_for_server("https://example.invalid/v1", "teacher", 1)
            request.assert_not_called()

    def test_wrong_advertised_model_fails_before_generation(self):
        with patch.object(teacher_smoke, "urlopen") as request:
            request.return_value.__enter__.return_value.read.return_value = b'{"data":[{"id":"wrong"}]}'
            with self.assertRaisesRegex(ValueError, "does not advertise"):
                teacher_smoke.wait_for_server("http://127.0.0.1:8000/v1", "teacher", 1)

    def test_smoke_uses_explicit_or_bench_options_through_generation_and_scoring(self):
        source = self.root / "hard.csv"
        source.write_text("prompt,category\nExplain encryption.,privacy\n", encoding="utf-8")
        info = {flag: 0 for flag in teacher_smoke.FAILURE_FLAGS}
        info["apertus2_output"] = "native fixture, not checkpoint output"
        model = MagicMock()

        def generate(requests, **kwargs):
            requests[0].length_info = [info]
            return ["Encryption protects data."]

        model.generate_until.side_effect = generate
        args = SimpleNamespace(
            output=self.root / "run", checkpoint=self.root, source=source,
            upstream=self.root, limit=1, model="chonk-sft-fixture",
            base_url="http://127.0.0.1:8000/v1", max_tokens=128,
            max_model_len=16384, ready_timeout=1,
        )
        modules = {
            "lm_eval.api.instance": SimpleNamespace(Instance=SimpleNamespace),
            "lm_eval.models.apertus2": SimpleNamespace(Apertus2Completions=lambda **kwargs: model),
        }
        with (
            patch.object(teacher_smoke, "preflight", return_value={"weights_loaded": False}),
            patch.object(teacher_smoke.open_benchmarks, "staged_assets", return_value={"files": {}}),
            patch.object(teacher_smoke, "wait_for_server") as ready,
            patch.dict("sys.modules", modules),
        ):
            report = teacher_smoke.run(args)
        ready.assert_called_once()
        self.assertEqual(report["status"], "succeeded")
        self.assertEqual(report["scored_cases"], 1)
        self.assertEqual(report["benchmark_options"], {
            "keyword_profile": "opensource", "dataset_config": "or-bench-hard-1k",
        })
        self.assertEqual(report["benchmark"]["metrics"]["refused"], {"value": 0.0, "denominator": 1})

    def test_preflight_validates_benchmark_inputs_without_starting_inference(self):
        source = self.root / "hard.csv"
        source.write_text("prompt,category\nExplain encryption.,privacy\n", encoding="utf-8")
        output = self.root / "preflight.json"
        with (
            patch.object(teacher_smoke, "preflight", return_value={"weights_loaded": False}),
            patch.object(teacher_smoke.open_benchmarks, "staged_assets", return_value={"files": {}}),
            patch.object(teacher_smoke, "wait_for_server") as ready,
        ):
            code = teacher_smoke.main([
                "preflight", "--checkpoint", str(self.root), "--source", str(source),
                "--upstream", str(self.root), "--limit", "1", "--output", str(output),
            ])
        self.assertEqual(code, 0)
        ready.assert_not_called()
        self.assertEqual(json.loads(output.read_text())["or_bench"]["source_cases"], 1)

    def test_preflight_accepts_explicit_twenty_case_pilot(self):
        source = self.root / "balanced.csv"
        source.write_text(
            "prompt,category\n" + "".join(f"Fixture request {i},privacy\n" for i in range(20)),
            encoding="utf-8",
        )
        output = self.root / "twenty-preflight.json"
        with (
            patch.object(teacher_smoke, "preflight", return_value={"weights_loaded": False}),
            patch.object(teacher_smoke.open_benchmarks, "staged_assets", return_value={"files": {}}),
        ):
            code = teacher_smoke.main([
                "preflight", "--checkpoint", str(self.root), "--source", str(source),
                "--upstream", str(self.root), "--limit", "20", "--output", str(output),
            ])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.read_text())["or_bench"]["selected_cases"], 20)


if __name__ == "__main__":
    unittest.main()
