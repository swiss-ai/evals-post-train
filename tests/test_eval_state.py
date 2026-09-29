import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.alignment.merge_split_results import merge_split_results
from scripts.eval_state import (
    RUN_CONFIG_FILENAME,
    make_run_config,
    read_tasks,
    scan_results,
)


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

    def test_scan_rejects_legacy_and_differently_configured_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = Path(tmp) / "harness"
            legacy = harness / "eval_001"
            limited = harness / "eval_002"
            full = harness / "eval_003"
            for result_dir in (legacy, limited, full):
                result_dir.mkdir(parents=True)
                (result_dir / "results_test.json").write_text(
                    json.dumps({"results": {"alpha": {}}}), encoding="utf-8"
                )

            limited_config = make_run_config(["model=test/model", "limit=5"])
            full_config = make_run_config(["limit=", "model=test/model"])
            self.assertEqual(
                full_config,
                make_run_config(["model=test/model", "limit="]),
            )
            (limited / RUN_CONFIG_FILENAME).write_text(
                json.dumps(limited_config), encoding="utf-8"
            )
            (full / RUN_CONFIG_FILENAME).write_text(
                json.dumps(full_config), encoding="utf-8"
            )

            completed, missing, result_dirs = scan_results(
                ["alpha"], harness, [], run_config=full_config
            )

            self.assertEqual(completed, {"alpha": full})
            self.assertEqual(missing, [])
            self.assertEqual(result_dirs, [full])

            another_limit = make_run_config(["model=test/model", "limit=10"])
            completed, missing, result_dirs = scan_results(
                ["alpha"], harness, [], run_config=another_limit
            )
            self.assertEqual(completed, {})
            self.assertEqual(missing, ["alpha"])
            self.assertEqual(result_dirs, [])

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
                # The environment-preparation path, even inside the prebuilt image.
                "EVAL_PREBUILT_ENV": "0",
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

    def test_failed_environment_submission_stops_the_launch(self) -> None:
        # e.g. an sbatch option it rejects (SBATCH_EXCLUSIVE=1): nothing may be
        # submitted after it, least of all an eval array without the dependency.
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / "bin"
            bin_dir.mkdir()
            calls = Path(tmp) / "sbatch_calls.txt"
            fake_sbatch = bin_dir / "sbatch"
            fake_sbatch.write_text(
                "#!/bin/bash\n"
                f'echo "$*" >> "{calls}"\n'
                'echo "sbatch: error: Invalid --exclusive specification" >&2\n'
                "exit 1\n",
                encoding="utf-8",
            )
            fake_sbatch.chmod(0o755)
            task_file = Path(tmp) / "tasks.txt"
            task_file.write_text("a\nb\n", encoding="utf-8")
            env = os.environ | {
                "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                "TASKS": str(task_file),
                "TABLE_METRICS": str(task_file),
                "LOGS_ROOT": str(Path(tmp) / "logs"),
                "WANDB_ENTITY": "test",
                "WANDB_PROJECT": "test",
                "EVAL_FAILURE_POLICY": "resume",
                "EVAL_CHUNK_SIZE": "2",
                "EVAL_FORCE_TASKS": "",
                "SBATCH_SCRIPT": "scripts/evaluate.sbatch",
                "LM_EVAL_BACKEND": "vllm",
                "EVAL_PREBUILT_ENV": "0",
                "JUDGE_MODE": "none",
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
                text=True,
                capture_output=True,
            )
            output = completed.stdout + completed.stderr
            self.assertNotEqual(completed.returncode, 0, output)
            self.assertIn("could not submit the environment preparation job", output)
            self.assertNotIn("Environment preparation job:", output)
            submitted = calls.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(submitted), 1, submitted)
        self.assertIn("prepare_eval_env.sbatch", submitted[0])

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
