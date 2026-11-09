import json
import tempfile
import unittest
from pathlib import Path

from scripts.lm_eval_harness.bos_default import needs_bos


class BosDefaultTests(unittest.TestCase):
    def test_gemma4_repo_ids(self):
        for model in (
            "google/gemma-4-31B-it",
            "google/gemma-4-E2B-it",
            "google/gemma-4-26B-A4B-it",
            "CSCS-Inference/google/gemma-4-31B-it",
            "some-org/gemma4",
        ):
            self.assertTrue(needs_bos(model), model)

    def test_other_models_keep_the_tokenizer_default(self):
        for model in (
            "google/gemma-3-27b-it",
            "Qwen/Qwen3-32B",
            "openai/gpt-oss-120b",
            "swiss-ai/Apertus-8B-Instruct-2509",
            "allenai/Olmo-3.1-32B-Instruct",
            "google/medgemma-4b-it",
            "RCP-AIaaS/google/medgemma-27b-text-it",
            "google/gemma-3-4b-it",
        ):
            self.assertFalse(needs_bos(model), model)

    def test_local_directory_is_read_by_model_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "checkpoint-1000"
            ckpt.mkdir()
            (ckpt / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
            self.assertTrue(needs_bos(str(ckpt)))
            (ckpt / "config.json").write_text(json.dumps({"model_type": "gemma3"}))
            self.assertFalse(needs_bos(str(ckpt)))

    def test_local_model_type_wins_over_the_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "gemma-4-distilled-into-llama"
            ckpt.mkdir()
            (ckpt / "config.json").write_text(json.dumps({"model_type": "llama"}))
            self.assertFalse(needs_bos(str(ckpt)))

    def test_local_directory_without_config_falls_back_to_its_name(self):
        self.assertTrue(needs_bos("/nonexistent/models/gemma-4-31B-it"))
        self.assertFalse(needs_bos("/nonexistent/gemma4/Qwen3-8B"))


if __name__ == "__main__":
    unittest.main()
