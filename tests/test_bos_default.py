import ast
import json
import tempfile
import unittest
from pathlib import Path

from scripts.lm_eval_harness.bos_default import needs_bos, prepends_bos


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

    def _tokenizer_dir(self, tmp, config):
        tok = Path(tmp) / "tok"
        tok.mkdir()
        (tok / "tokenizer_config.json").write_text(json.dumps(config))
        return str(tok)

    def test_without_a_chat_template_a_bos_token_the_tokenizer_does_not_add_is_prepended(self):
        with tempfile.TemporaryDirectory() as tmp:
            # the Apertus 2 instruct tokenizer: BOS defined, add_bos_token false
            tok = self._tokenizer_dir(tmp, {"add_bos_token": False, "bos_token": "<s>"})
            self.assertTrue(prepends_bos(tok))
            self.assertTrue(needs_bos(tok, chat_template=False))

    def test_a_bos_token_given_as_a_dict_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok = self._tokenizer_dir(tmp, {"add_bos_token": False, "bos_token": {"content": "<s>"}})
            self.assertTrue(needs_bos(tok, chat_template=False))

    def test_a_tokenizer_that_adds_bos_itself_is_left_alone(self):
        # --bos there reaches a server that adds a BOS too: the harness decodes its own BOS into
        # log-likelihood prompts, so the server sees two (the Chonk eval tokenizer, Llama 3, Mistral)
        with tempfile.TemporaryDirectory() as tmp:
            tok = self._tokenizer_dir(tmp, {"add_bos_token": True, "bos_token": "<s>"})
            self.assertFalse(needs_bos(tok, chat_template=False))

    def test_a_tokenizer_that_says_nothing_is_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok = self._tokenizer_dir(tmp, {"bos_token": "<s>"})
            self.assertFalse(needs_bos(tok, chat_template=False))

    def test_with_a_chat_template_the_template_writes_the_bos(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok = self._tokenizer_dir(tmp, {"add_bos_token": False, "bos_token": "<s>"})
            self.assertFalse(needs_bos(tok))
            self.assertFalse(needs_bos(tok, chat_template=True))

    def test_no_bos_token_means_nothing_is_prepended(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok = self._tokenizer_dir(tmp, {"add_bos_token": False, "bos_token": None})
            self.assertFalse(needs_bos(tok, chat_template=False))

    def test_unreadable_tokenizer_config_means_nothing_is_prepended(self):
        self.assertFalse(prepends_bos("/nonexistent/tokenizer"))
        with tempfile.TemporaryDirectory() as tmp:
            tok = Path(tmp) / "tok"
            tok.mkdir()
            (tok / "tokenizer_config.json").write_text("not json")
            self.assertFalse(prepends_bos(str(tok)))

    def test_gemma4_still_wins_without_a_chat_template(self):
        self.assertTrue(needs_bos("/nonexistent/models/gemma-4-31B-it", chat_template=False))

    def test_the_helper_runs_on_the_python_3_6_of_the_cluster_nodes(self):
        # run_lm_eval.sbatch calls it with the node's bare python3, which is 3.6 on Clariden: no future-annotations
        # import (3.7), nothing a 3.6 grammar rejects; the helper then answered "false" silently
        source = (Path(__file__).resolve().parent.parent / "scripts/lm_eval_harness/bos_default.py").read_text()
        ast.parse(source, feature_version=(3, 6))
        self.assertNotIn("from __future__ import annotations", source)


if __name__ == "__main__":
    unittest.main()
