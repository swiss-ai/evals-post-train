import json
import tempfile
import unittest
from pathlib import Path

from scripts.results.evalscope_wandb_utils import create_model_evaluation_from_evalscope_run


def _review(instance, acc, report, completed=True, start=1000.0, end=1060.0):
    return {
        "agent_trace": {
            "total_usage": {"input_tokens": 100, "output_tokens": 10},
            "events": [{"timestamp": start}, {"timestamp": end}],
        },
        "sample_score": {
            "sample_id": 0,
            "sample_metadata": {"instance_id": instance, "repo": "org/repo"},
            "score": {
                "value": {"acc": acc},
                "extracted_prediction": "diff --git a/x b/x",
                "metadata": {"completed": completed, "resolved": acc == 1.0, "report": report},
            },
        },
    }


def _report(dataset, score, num):
    return {
        "path": f"reports/m/{dataset}.json",
        "dataset_name": dataset,
        "model_name": "m",
        "score": None,
        "metrics": [{"identity": {"name": "accuracy"}, "num": num, "score": score}],
    }


class EvalScopeWandbTests(unittest.TestCase):
    def test_model_evaluation_from_evalscope_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary)
            summary = {
                "benchmarks": {"swe_bench": "swe_bench_verified_agentic", "swe_bench_pro": "swe_bench_pro"},
                "reports": [_report("swe_bench_pro", 0.25, 4), _report("swe_bench_verified_agentic", 1.0, 1)],
                "evaluation_errors": {"m/swe_bench_pro_default": {"count": 1, "instances": ["p4"]}},
            }
            (work_dir / "summary.json").write_text(json.dumps(summary))
            reviews = work_dir / "reviews" / "m"
            reviews.mkdir(parents=True)
            pro = [
                # passed_count counts every passing test, not only the required ones
                _review("p1", 1.0, {"f2p": ["A"], "p2p": [], "tests": [{"name": "A", "status": "PASSED"},
                                                                         {"name": "Z", "status": "PASSED"}],
                                    "passed_count": 2, "required_count": 1}),
                # a required test that did not run at all also failed
                _review("p2", 0.0, {"f2p": ["A", "B"], "p2p": ["C"], "tests": [{"name": "A", "status": "PASSED"},
                                                                                {"name": "C", "status": "FAILED"}]}),
                _review("p3", 0.0, {"f2p": ["A"], "p2p": [], "tests": []}, end=1500.0),
                _review("p4", 0.0, {"error": "sandbox exec lost"}, completed=False),
            ]
            (reviews / "swe_bench_pro_default.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pro))
            verified = [_review("v1", 1.0, {"v1": {"resolved": True, "tests_status": {
                "FAIL_TO_PASS": {"success": ["t1"], "failure": []},
                "PASS_TO_PASS": {"success": ["t2", "t3"], "failure": []}}}}, start=900.0)]
            (reviews / "swe_bench_verified_agentic_default.jsonl").write_text(json.dumps(verified[0]) + "\n")

            model_eval, duration = create_model_evaluation_from_evalscope_run("m-run", work_dir)

        tasks = {task.task_name: task for task in model_eval.tasks}
        self.assertEqual(set(tasks), {"swe_bench", "swe_bench_pro"})
        self.assertEqual(
            {m.name: m.score for m in tasks["swe_bench_pro"].metrics},
            {"accuracy": 0.25, "n_samples": 4, "n_errors": 1},
        )
        self.assertEqual({m.name: m.score for m in tasks["swe_bench"].metrics}["n_errors"], 0)
        self.assertEqual(duration, 600)

        samples = {s.sample_data["instance_id"]: s.sample_data for s in tasks["swe_bench_pro"].samples}
        self.assertEqual(set(samples), {"p1", "p2", "p3", "p4"})
        self.assertEqual((samples["p1"]["tests_passed"], samples["p1"]["tests_required"]), (1, 1))
        self.assertEqual((samples["p2"]["tests_passed"], samples["p2"]["failed_tests"]), (1, ["B", "C"]))
        self.assertTrue(samples["p1"]["is_correct"])
        self.assertFalse(samples["p3"]["is_correct"])
        # An evaluation error is not the model's failure: neither passed nor failed.
        self.assertIsNone(samples["p4"]["is_correct"])
        self.assertNotIn("acc", samples["p4"])
        self.assertEqual(samples["p4"]["error"], "sandbox exec lost")

        (v1,) = [s.sample_data for s in tasks["swe_bench"].samples]
        self.assertEqual((v1["tests_passed"], v1["tests_required"], v1["failed_tests"]), (3, 3, []))


if __name__ == "__main__":
    unittest.main()
