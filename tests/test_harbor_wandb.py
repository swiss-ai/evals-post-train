import json
import tempfile
import unittest
from pathlib import Path

from scripts.alignment.harbor_wandb_utils import create_model_evaluation_from_harbor_run


def _trial(task, reward=None, error=None, start="2026-10-27T09:00:00Z", end="2026-10-27T09:10:00Z"):
    return {
        "task_name": task,
        "trial_name": f"{task}__abc",
        "verifier_result": {"rewards": {"reward": reward}} if reward is not None else None,
        "exception_info": {"exception_type": error, "exception_message": "boom"} if error else None,
        "agent_result": {"n_input_tokens": 100, "n_output_tokens": 10},
        "started_at": start,
        "finished_at": end,
    }


class HarborWandbTests(unittest.TestCase):
    def test_model_evaluation_from_harbor_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary)
            summary = {
                "benchmarks": {
                    "swe_bench": {
                        "metrics": {"reward": 0.5},
                        "n_trials": 3,
                        "n_scored": 2,
                        "evaluation_errors": [{"task": "c", "trial": "c__abc", "error": "X"}],
                    },
                    "slop_code_bench": {"metrics": {"strict_pass_rate": 0.25}, "n_trials": 1, "n_scored": 1,
                                        "evaluation_errors": []},
                    "aider_polyglot": {"mean_reward": 1.0, "n_trials": 2, "n_scored": 2, "pass_at_k": {"2": 1.0}},
                }
            }
            (work_dir / "summary.json").write_text(json.dumps(summary))
            trials = {
                "swe_bench": [_trial("a", 1.0), _trial("b", 0.0, end="2026-10-27T09:30:00Z"), _trial("c", error="X")],
                "slop_code_bench": [_trial("d", 0.25)],
            }
            for bench, items in trials.items():
                for i, trial in enumerate(items):
                    path = work_dir / "jobs" / bench / f"t{i}" / "result.json"
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps(trial))

            evaluation, duration = create_model_evaluation_from_harbor_run("m", work_dir)

        self.assertEqual(duration, 30 * 60)
        flat = evaluation.get_flattened_metrics()
        self.assertEqual(flat["swe_bench/reward"], 0.5)
        self.assertEqual(flat["swe_bench/n_errors"], 1)
        self.assertEqual(flat["slop_code_bench/strict_pass_rate"], 0.25)
        # An older summary.json (mean_reward only), with pass@k from --attempts.
        self.assertEqual(flat["aider_polyglot/reward"], 1.0)
        self.assertEqual(flat["aider_polyglot/pass@2"], 1.0)
        swe = evaluation.tasks[0]
        self.assertEqual(len(swe.samples), 3)
        by_task = {s.sample_data["task"]: s.sample_data for s in swe.samples}
        self.assertTrue(by_task["a"]["is_correct"])
        self.assertFalse(by_task["b"]["is_correct"])
        self.assertIsNone(by_task["c"]["is_correct"])
        self.assertTrue(by_task["c"]["error"].startswith("X: "))
        # A non-binary reward is not classified as pass/fail.
        self.assertNotIn("pass@1", evaluation.tasks[1].samples[0].sample_data)


if __name__ == "__main__":
    unittest.main()
