"""Offline checks of scripts/custom/run_bfcl.sh's argument handling and of scripts/custom/bfcl.py's helpers
(no network, model or bfcl-eval: the driver imports the package only inside the functions that need it)."""

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.custom import bfcl

REPO = Path(__file__).resolve().parent.parent


def run(*args):
    return subprocess.run(
        ["bash", "scripts/custom/run_bfcl.sh", *args],
        cwd=REPO,
        env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
        capture_output=True,
        text=True,
        timeout=60,
    )


class ScriptTest(unittest.TestCase):
    def test_help_documents_the_options(self):
        result = run("--help")
        self.assertEqual(result.returncode, 0)
        for flag in ("--version", "--api-base-url", "--workdir", "--num-tasks", "--test-category", "--reasoning-effort",
                     "--request-timeout"):
            self.assertIn(flag, result.stdout)

    def test_version_model_and_endpoint_are_required(self):
        self.assertIn("--version must be v3 or v4", run("--model", "m", "--api-base-url", "http://x").stderr)
        self.assertIn("--version must be v3 or v4", run("--version", "v5", "--model", "m").stderr)
        self.assertIn("Missing --model", run("--version", "v3").stderr)
        self.assertIn("Missing --api-base-url", run("--version", "v3", "--model", "m").stderr)

    def test_unknown_argument_is_refused(self):
        result = run("--version", "v3", "--bogus")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown argument", result.stderr)


class CategoriesTest(unittest.TestCase):
    def test_v4_is_v3_and_the_memory_categories(self):
        self.assertEqual(len(bfcl.CATEGORIES["v3"]), 17)
        self.assertEqual(bfcl.CATEGORIES["v4"][:17], bfcl.CATEGORIES["v3"])
        self.assertEqual(bfcl.CATEGORIES["v4"][17:], ["memory_kv", "memory_vector", "memory_rec_sum"])
        self.assertFalse([c for c in bfcl.CATEGORIES["v4"] if c.startswith("web_search")])


class RequestTest(unittest.TestCase):
    def test_temperature_goes_to_the_handler_and_the_rest_to_the_request(self):
        self.assertEqual(bfcl.split_llm_args({}), (bfcl.DEFAULT_TEMPERATURE, {}))
        args = {"temperature": 0.6, "top_p": 0.95, "reasoning_effort": "high", "extra_body": {"top_k": 20}}
        self.assertEqual(
            bfcl.split_llm_args(args),
            (0.6, {"top_p": 0.95, "reasoning_effort": "high", "extra_body": {"top_k": 20}}),
        )

    def test_registry_names_have_no_slash(self):
        self.assertEqual(bfcl.registry_name("CSCS-Inference/swiss-ai/Apertus-v1.5-8B"), "CSCS-Inference_swiss-ai_Apertus-v1.5-8B-FC")

    def test_clean_message_drops_nulls_and_keeps_content_for_text_turns(self):
        self.assertEqual(
            bfcl.clean_message({"role": "assistant", "content": None, "tool_calls": None, "refusal": None}),
            {"role": "assistant", "content": ""},
        )
        calls = [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        self.assertEqual(
            bfcl.clean_message({"role": "assistant", "content": None, "tool_calls": calls}),
            {"role": "assistant", "tool_calls": calls},
        )


class SummaryTest(unittest.TestCase):
    def write(self, root: Path, registry: str):
        for group, category, correct, total in (
            ("non_live", "simple_python", 9, 10), ("live", "live_simple", 6, 10), ("multi_turn", "multi_turn_base", 2, 10),
        ):
            directory = root / registry / group
            directory.mkdir(parents=True)
            (directory / f"BFCL_v4_{category}_score.json").write_text(
                json.dumps({"accuracy": correct / total, "correct_count": correct, "total_count": total}) + "\n{}\n"
            )
        for name, column, value in (
            ("data_non_live.csv", "Non-Live Overall Acc", "90.00%"),
            ("data_live.csv", "Live Overall Acc", "60.00%"),
            ("data_multi_turn.csv", "Multi Turn Overall Acc", "20.00%"),
            ("data_agentic.csv", "Agentic Overall Acc", "N/A"),
        ):
            with open(root / name, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Rank", "Model", column])
                writer.writerow(["N/A", "m (FC)", value])

    def test_v3_overall_is_the_mean_of_the_three_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write(Path(tmp), "m-FC")
            summary = bfcl.summarize("v3", "m-FC", Path(tmp))
        self.assertAlmostEqual(summary["overall"], (0.9 + 0.6 + 0.2) / 3)
        self.assertEqual(summary["groups"], {"non_live": 0.9, "live": 0.6, "multi_turn": 0.2, "agentic": None})
        self.assertEqual(summary["categories"]["simple_python"], {"accuracy": 0.9, "correct": 9, "total": 10})

    def test_v4_has_no_overall_without_all_its_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write(Path(tmp), "m-FC")
            self.assertIsNone(bfcl.summarize("v4", "m-FC", Path(tmp))["overall"])


if __name__ == "__main__":
    unittest.main()
