import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.alignment.merge_split_results import merge_split_results
from scripts.eval_state import read_tasks, scan_results


class EvalStateTests(unittest.TestCase):
    def test_read_tasks_supports_comments_and_comma_expressions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_file = Path(tmp) / "tasks.txt"
            task_file.write_text(
                "# heading\nalpha\nbeta  # inline\n\n",
                encoding="utf-8",
            )
            self.assertEqual(read_tasks(str(task_file)), ["alpha", "beta"])
        self.assertEqual(read_tasks("alpha,beta,gamma"), ["alpha", "beta", "gamma"])

    def test_scan_maps_nested_group_keys_and_unique_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness_dir = Path(tmp) / "harness"
            first = harness_dir / "eval_001"
            second = harness_dir / "eval_002"
            merged = harness_dir / "eval_merged_003"
            first.mkdir(parents=True)
            second.mkdir()
            merged.mkdir()
            (first / "results_a.json").write_text(
                json.dumps({"results": {"alpha": {}, "group": {"beta": {}}}}),
                encoding="utf-8",
            )
            (second / "results_b.json").write_text(
                json.dumps({"results": {"gamma": {}}}), encoding="utf-8"
            )
            (merged / "results_c.json").write_text(
                json.dumps({"results": {"missing": {}}}), encoding="utf-8"
            )

            completed, missing, result_dirs = scan_results(
                ["alpha", "beta", "gamma", "missing"], harness_dir, []
            )

            self.assertEqual(set(completed), {"alpha", "beta", "gamma"})
            self.assertEqual(missing, ["missing"])
            self.assertEqual(result_dirs, [first, second])

    def test_force_patterns_make_matching_tasks_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "harness" / "eval_001"
            result_dir.mkdir(parents=True)
            (result_dir / "results_test.json").write_text(
                json.dumps({"results": {"alpha": {}, "beta": {}}}),
                encoding="utf-8",
            )
            completed, missing, _ = scan_results(
                ["alpha", "beta"], result_dir.parent, ["alp"]
            )
            self.assertEqual(set(completed), {"beta"})
            self.assertEqual(missing, ["alpha"])

    def test_forced_task_requires_a_result_from_the_current_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = Path(tmp) / "harness"
            for directory_name in ("eval_20260101_000000_1", "eval_20270101_000000_2"):
                result_dir = harness / directory_name
                result_dir.mkdir(parents=True)
                (result_dir / "results_test.json").write_text(
                    json.dumps({"results": {"alpha": {}}}), encoding="utf-8"
                )

            completed, missing, result_dirs = scan_results(
                ["alpha"],
                harness,
                ["alp"],
                force_after="eval_20260731_000000",
            )
            self.assertEqual(set(completed), {"alpha"})
            self.assertEqual(missing, [])
            self.assertEqual(result_dirs, [harness / "eval_20270101_000000_2"])

    def test_orchestrator_dry_run_builds_bounded_array(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            task_file = Path(tmp) / "tasks.txt"
            task_file.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
            env = os.environ | {
                "TASKS": str(task_file),
                "TABLE_METRICS": str(task_file),
                "LOGS_ROOT": str(Path(tmp) / "logs"),
                "WANDB_ENTITY": "test",
                "WANDB_PROJECT": "test",
                "EVAL_DRY_RUN": "true",
                "EVAL_FAILURE_POLICY": "resume",
                "EVAL_CHUNK_SIZE": "2",
                "EVAL_MAX_PARALLEL": "2",
                "EVAL_MAX_RETRIES": "1",
                "EVAL_FORCE_TASKS": "",
                "SBATCH_SCRIPT": "scripts/evaluate.sbatch",
                "LM_EVAL_BACKEND": "vllm",
            }
            completed = subprocess.run(
                [
                    "bash",
                    "-c",
                    "set -euo pipefail; source scripts/evaluation_orchestrator.sh; "
                    "submit_evaluation test/model test-model",
                ],
                cwd=repo_root,
                env=env,
                check=True,
                text=True,
                capture_output=True,
            )
            output = completed.stdout + completed.stderr
            self.assertIn("--array=0-2%2", output)
            self.assertIn("scripts/prepare_eval_env.sbatch", output)
            self.assertIn("afterany:dry-array-0", output)
            self.assertIn("--job-name=eval-ctrl-test-model-a0", output)

    def test_merge_prefers_newer_metrics_and_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            older = root / "eval_001"
            newer = root / "eval_002"
            output = root / "merged"
            older.mkdir()
            newer.mkdir()
            output.mkdir()
            for directory, score, sample in (
                (older, 1, "old"),
                (newer, 2, "new"),
            ):
                (directory / "results_test.json").write_text(
                    json.dumps(
                        {
                            "results": {"alpha": {"score": score}},
                            "configs": {"alpha": {}},
                        }
                    ),
                    encoding="utf-8",
                )
                (directory / "samples_alpha_test.jsonl").write_text(
                    sample, encoding="utf-8"
                )

            merge_split_results([older, newer], output)
            merged = json.loads(
                next(output.glob("results_*.json")).read_text(encoding="utf-8")
            )
            self.assertEqual(merged["results"]["alpha"]["score"], 2)
            self.assertEqual(
                (output / "samples_alpha_test.jsonl").read_text(encoding="utf-8"),
                "new",
            )

            partial = root / "partial"
            partial.mkdir()
            merge_split_results([older, newer], partial, {"alpha"})
            partial_results = json.loads(
                next(partial.glob("results_*.json")).read_text(encoding="utf-8")
            )
            self.assertNotIn("alpha", partial_results["results"])
            self.assertFalse((partial / "samples_alpha_test.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
