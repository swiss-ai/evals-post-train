import json
import tempfile
import unittest
from pathlib import Path

from scripts.alignment.wandb_alignment_utils import (
    create_model_evaluation_from_results,
)


class WandbAlignmentUtilsTests(unittest.TestCase):
    def test_lm_eval_task_metadata_is_not_treated_as_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            eval_dir = Path(temporary)
            result = {
                "results": {
                    "bfcl_v3_simple": {
                        "name": "bfcl_v3_simple",
                        "alias": "bfcl_v3_simple",
                        "sample_len": 400,
                        "sample_count": 400,
                        "samples": ["task metadata"],
                        "acc,none": 0.8975,
                        "acc_stderr,none": "N/A",
                        "acc_lenient,none": 0.8975,
                        "acc_lenient_stderr,none": "N/A",
                    }
                }
            }
            (eval_dir / "results_2026-08-15T21-40-36.909277.json").write_text(
                json.dumps(result), encoding="utf-8"
            )

            evaluation = create_model_evaluation_from_results("test-model", eval_dir)

            self.assertEqual(evaluation.model_name, "test-model")
            self.assertEqual(len(evaluation.tasks), 1)
            self.assertEqual(
                [(metric.name, metric.score) for metric in evaluation.tasks[0].metrics],
                [("acc", 0.8975), ("acc_lenient", 0.8975)],
            )

    def test_unknown_non_numeric_result_field_is_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            eval_dir = Path(temporary)
            result = {
                "results": {
                    "example": {
                        "future_metadata": {"schema": 2},
                        "acc,none": "0.5",
                    }
                }
            }
            (eval_dir / "results_2026-08-15T21-40-36.json").write_text(
                json.dumps(result), encoding="utf-8"
            )

            with self.assertLogs(
                "scripts.alignment.wandb_alignment_utils", level="WARNING"
            ) as logs:
                evaluation = create_model_evaluation_from_results(
                    "test-model", eval_dir
                )

            self.assertIn("future_metadata", "\n".join(logs.output))
            self.assertEqual(
                [(metric.name, metric.score) for metric in evaluation.tasks[0].metrics],
                [("acc", 0.5)],
            )


if __name__ == "__main__":
    unittest.main()
