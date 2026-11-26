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
        for flag in ("--version", "--api-base-url", "--out-dir", "--limit", "--test-category", "--reasoning-effort",
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

    def test_registry_names_survive_bfcl_evals_directory_round_trip(self):
        # The evaluator turns the result directory's "_" back into "/": no key may hold either character.
        name = bfcl.registry_name("CSCS-Inference/swiss-ai/Apertus-v1.5-8B")
        self.assertEqual(name, "custom-CSCS-Inference-swiss-ai-Apertus-v1.5-8B-FC")
        self.assertEqual(name.replace("/", "_").replace("_", "/"), name)
        weird = bfcl.registry_name("Qwen/Qwen3_8B (x)")
        self.assertNotIn("/", weird)
        self.assertNotIn("_", weird)
        self.assertTrue(weird.startswith("custom-"))

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
    CATEGORIES = {
        "non_live": ["simple_python", "simple_java", "simple_javascript", "multiple", "parallel", "parallel_multiple", "irrelevance"],
        "live": ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple", "live_irrelevance", "live_relevance"],
        "multi_turn": ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"],
        "agentic": ["memory_kv", "memory_vector", "memory_rec_sum"],
    }

    def write(self, root: Path, registry: str, groups=("non_live", "live", "multi_turn", "agentic")):
        for group in groups:
            for category in self.CATEGORIES[group]:
                # bfcl-eval puts the memory categories one level deeper: agentic/memory/<kv|vector|rec_sum>/
                directory = root / registry / group / ("memory/" + category.split("_", 1)[1] if category.startswith("memory") else "")
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f"BFCL_v4_{category}_score.json").write_text(
                    json.dumps({"accuracy": 0.5, "correct_count": 5, "total_count": 10}) + "\n{}\n"
                )
        for name, column, value in (
            ("data_non_live.csv", "Non-Live Overall Acc", "90.00%"),
            ("data_live.csv", "Live Overall Acc", "60.00%"),
            ("data_multi_turn.csv", "Multi Turn Overall Acc", "20.00%"),
            ("data_agentic.csv", "Agentic Overall Acc", "0.00%"),  # bfcl-eval counts a missing category as 0
        ):
            with open(root / name, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Rank", "Model", column])
                writer.writerow(["N/A", "m (FC)", value])

    def test_num_tasks_keeps_the_prerequisite_conversations_of_the_selected_memory_scenarios(self):
        ids = ["memory_kv_prereq_0-customer-0", "memory_kv_prereq_1-customer-1", "memory_kv_prereq_22-student-0",
               "memory_kv_0-customer-0", "memory_kv_1-customer-1", "memory_kv_30-student-0"]
        self.assertEqual(bfcl.select_ids(ids, 2), ["memory_kv_prereq_0-customer-0", "memory_kv_prereq_1-customer-1",
                                                   "memory_kv_0-customer-0", "memory_kv_1-customer-1"])
        self.assertEqual(bfcl.select_ids(ids, 3)[-3:], ["memory_kv_0-customer-0", "memory_kv_1-customer-1", "memory_kv_30-student-0"])
        self.assertIn("memory_kv_prereq_22-student-0", bfcl.select_ids(ids, 3))

    def test_num_tasks_on_other_categories_is_the_first_n(self):
        self.assertEqual(bfcl.select_ids(["simple_python_0", "simple_python_1", "simple_python_2"], 2),
                         ["simple_python_0", "simple_python_1"])

    def test_v3_overall_is_the_mean_of_the_three_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write(Path(tmp), "m-FC", groups=("non_live", "live", "multi_turn"))
            summary = bfcl.summarize("v3", "m-FC", Path(tmp))
        self.assertAlmostEqual(summary["overall"], (0.9 + 0.6 + 0.2) / 3)
        self.assertEqual(summary["groups"], {"non_live": 0.9, "live": 0.6, "multi_turn": 0.2})  # no agentic: not run
        self.assertEqual(summary["categories"]["simple_python"], {"accuracy": 0.5, "correct": 5, "total": 10})

    def test_the_agentic_group_needs_web_search_and_so_v4_has_no_overall(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write(Path(tmp), "m-FC")  # memory categories only
            summary = bfcl.summarize("v4", "m-FC", Path(tmp))
        self.assertNotIn("agentic", summary["groups"])
        self.assertIsNone(summary["overall"])
        self.assertIn("memory_kv", summary["categories"])
        self.assertIn("v4_without_web_search", summary)  # every category it needs was run

    @staticmethod
    def accuracies(value, **given):
        categories = {name: {"accuracy": value, "correct": 0, "total": 10}
                      for name in bfcl.CATEGORIES["v4"]}
        for name, accuracy in given.items():
            categories[name] = {**categories[name], "accuracy": accuracy}
        return categories

    def test_v4_without_web_search_follows_the_packages_weights(self):
        # Every category equally right: any weighted mean is that accuracy.
        self.assertAlmostEqual(bfcl.v4_without_web_search(self.accuracies(0.5))["overall_no_web_search"], 0.5)
        zero = {name: 0.0 for name in bfcl.CATEGORIES["v4"]}
        for given, weight in (
            ({"memory_kv": 1, "memory_vector": 1, "memory_rec_sum": 1}, 0.40),  # agentic: memory alone
            ({n: 1 for n in bfcl.V3_CATEGORIES if n.startswith("multi_turn")}, 0.30),
            ({"irrelevance": 1, "live_irrelevance": 1}, 0.10),
            ({"live_simple": 1, "live_multiple": 1, "live_parallel": 1, "live_parallel_multiple": 1}, 0.10),
            ({"simple_python": 1, "simple_java": 1, "simple_javascript": 1, "multiple": 1, "parallel": 1,
              "parallel_multiple": 1}, 0.10),
        ):
            scores = bfcl.v4_without_web_search(self.accuracies(0.0, **given))
            self.assertAlmostEqual(scores["overall_no_web_search"], weight, msg=str(given))
        self.assertEqual(set(zero), set(bfcl.CATEGORIES["v4"]))

    def test_v4_without_web_search_averages_irrelevance_and_the_simple_languages(self):
        scores = bfcl.v4_without_web_search(self.accuracies(0.0, irrelevance=1.0))
        self.assertAlmostEqual(scores["irrelevance"], 0.5)
        scores = bfcl.v4_without_web_search(self.accuracies(0.0, simple_python=1.0))
        self.assertAlmostEqual(scores["non_live"], (1 / 3) / 4)  # simple is a third of its quarter

    def test_the_live_group_weights_its_categories_by_size(self):
        categories = self.accuracies(0.0, live_simple=1.0)
        categories["live_simple"]["total"] = 30
        scores = bfcl.v4_without_web_search(categories)
        self.assertAlmostEqual(scores["live"], 30 / (30 + 10 * 3))

    def test_v4_without_web_search_needs_every_category_it_uses(self):
        categories = self.accuracies(0.5)
        del categories["memory_vector"]
        self.assertIsNone(bfcl.v4_without_web_search(categories))


class InferenceErrorsTest(unittest.TestCase):
    def _result_dir(self, tmp, lines_by_file):
        root = Path(tmp)
        for relative, lines in lines_by_file.items():
            path = root / "custom-m-FC" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
        return root

    def test_entries_that_ended_in_an_inference_error_are_counted_by_cause(self):
        overflow = "Error during inference: Error code: 400 - This model's maximum context length is 32768 tokens"
        timeout = "Error during inference: Request timed out."
        with tempfile.TemporaryDirectory() as tmp:
            root = self._result_dir(tmp, {
                "BFCL_v4_simple_result.json": [{"id": "a", "result": "[f(x=1)]"}, {"id": "b", "result": timeout}],
                "multi_turn/BFCL_v4_multi_turn_long_context_result.json": [
                    {"id": "c", "result": overflow}, {"id": "d", "result": [["call"], ["call"]]}],
                "memory/BFCL_v4_memory_kv_result.json": [{"id": "e", "result": "Error occurred during inference: " + overflow}],
            })
            found = bfcl.inference_errors(root, "custom-m-FC")
        self.assertEqual(found, {"n_entries": 5, "n_errors": 3, "by_cause": {"context_overflow": 2, "other": 1}})

    def test_a_run_without_errors_or_results_reports_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._result_dir(tmp, {"BFCL_v3_simple_result.json": [{"id": "a", "result": "[f()]"}]})
            self.assertEqual(bfcl.inference_errors(root, "custom-m-FC")["n_errors"], 0)
            self.assertEqual(bfcl.inference_errors(root, "another-model"), {"n_entries": 0, "n_errors": 0, "by_cause": {"context_overflow": 0, "other": 0}})


class GenerationArgsTests(unittest.TestCase):
    def test_the_input_of_every_call_is_kept_in_the_trace(self):
        args = bfcl.generation_args("custom-m-FC", ["multi_turn_base"], 0.6, 8, 0)
        self.assertTrue(args.include_input_log)
        self.assertFalse(args.exclude_state_log)
        self.assertEqual((args.model, args.test_category, args.temperature, args.num_threads), (["custom-m-FC"], ["multi_turn_base"], 0.6, 8))

    def test_only_a_smoke_run_selects_entry_ids(self):
        self.assertFalse(bfcl.generation_args("m", [], 0.0, 1, 0).run_ids)
        self.assertTrue(bfcl.generation_args("m", [], 0.0, 1, 5).run_ids)


if __name__ == "__main__":
    unittest.main()
