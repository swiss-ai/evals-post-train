import json
import tempfile
import unittest
from pathlib import Path

from scripts.results.bfcl_wandb_utils import create_model_evaluation_from_bfcl_run


class BfclWandbTests(unittest.TestCase):
    def test_summary_becomes_one_task_with_groups_categories_and_size(self):
        summary = {
            "version": "v3", "overall": 0.5, "duration_s": 42,
            "groups": {"non_live": 0.9, "live": 0.8, "multi_turn": 0.1, "agentic": None},
            "categories": {"simple_python": {"accuracy": 0.9, "correct": 9, "total": 10},
                           "multi_turn_base": {"accuracy": 0.1, "correct": 1, "total": 10}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "summary.json").write_text(json.dumps(summary))
            evaluation, duration = create_model_evaluation_from_bfcl_run("m", Path(tmp))
        self.assertEqual(duration, 42)
        self.assertEqual(evaluation.get_flattened_metrics(), {
            "bfcl_v3/overall": 0.5, "bfcl_v3/non_live": 0.9, "bfcl_v3/live": 0.8, "bfcl_v3/multi_turn": 0.1,
            "bfcl_v3/simple_python": 0.9, "bfcl_v3/multi_turn_base": 0.1, "bfcl_v3/n_samples": 20,
        })

    def test_v4_has_no_overall(self):
        summary = {"version": "v4", "overall": None, "groups": {"live": 0.7}, "categories": {}}
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "summary.json").write_text(json.dumps(summary))
            evaluation, duration = create_model_evaluation_from_bfcl_run("m", Path(tmp))
        self.assertNotIn("bfcl_v4/overall", evaluation.get_flattened_metrics())
        self.assertEqual(duration, 0)


if __name__ == "__main__":
    unittest.main()
