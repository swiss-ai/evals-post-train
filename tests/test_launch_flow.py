import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _dry_run(*arguments: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        completed = subprocess.run(
            [
                "bash",
                "scripts/launch_evaluations.sh",
                *arguments,
                "--logs-root",
                tmp,
                "--debug",
            ],
            cwd=REPO_ROOT,
            env=os.environ | {"WANDB_PROJECT": "launcher-test"},
            check=True,
            text=True,
            capture_output=True,
        )
    return completed.stdout + completed.stderr


def _prebuilt_dry_run(*arguments: str, **extra_env: str) -> tuple[subprocess.CompletedProcess, str]:
    """A dry run with EVAL_PREBUILT_ENV=1; returns the run and its manifest."""
    with tempfile.TemporaryDirectory() as tmp:
        completed = subprocess.run(
            [
                "bash",
                "scripts/launch_evaluations.sh",
                *arguments,
                "--logs-root",
                tmp,
                "--debug",
            ],
            cwd=REPO_ROOT,
            env=os.environ
            | {"WANDB_PROJECT": "launcher-test", "EVAL_PREBUILT_ENV": "1"}
            | extra_env,
            text=True,
            capture_output=True,
        )
        manifests = list(Path(tmp).rglob("environment.sh"))
        manifest = manifests[0].read_text() if manifests else ""
    return completed, manifest


class LaunchFlowTests(unittest.TestCase):
    def test_omitted_mode_defaults_to_posttrain(self) -> None:
        output = _dry_run("--model", "test/model")
        self.assertIn("Mode:   posttrain", output)

    def test_openai_backend_selects_cpu_only_sbatch_wrapper(self) -> None:
        output = _dry_run(
            "single",
            "--task",
            "hellaswag",
            "--backend",
            "openai",
            "--api-base-url",
            "http://localhost:8000",
            "--api-model-name",
            "test-model",
        )
        self.assertIn("scripts/evaluate_api.sbatch test-model test-model", output)
        self.assertNotIn("scripts/evaluate.sbatch test-model test-model", output)
        wrapper = (REPO_ROOT / "scripts/evaluate_api.sbatch").read_text()
        self.assertNotIn("#SBATCH --gres", wrapper)
        self.assertNotIn("#SBATCH --exclusive", wrapper)

    def test_debug_is_forwarded_to_deferred_judge_launch(self) -> None:
        orchestrator = (REPO_ROOT / "scripts/evaluation_orchestrator.sh").read_text()
        self.assertIn("launch_args+=(--dry-run)", orchestrator)
        self.assertIn('_eval_launch_judge "$state_dir/missing_0.txt"', orchestrator)

    def test_prebuilt_environment_skips_preparation_and_pins_the_harness(self) -> None:
        pins = dict(
            line.split()
            for line in (REPO_ROOT / "requirements/lm-eval-harness.txt").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        for policy in ("resume", "fail-fast"):
            with self.subTest(policy=policy):
                completed, manifest = _prebuilt_dry_run(
                    "single", "--task", "hellaswag", "--model", "test/model",
                    "--failure-policy", policy,
                )
                output = completed.stdout + completed.stderr
                self.assertEqual(completed.returncode, 0, output)
                self.assertNotIn("prepare_eval_env.sbatch", output)
                self.assertIn("(no preparation job)", output)
                self.assertIn("EVAL_ENV_KIND=prebuilt", manifest)
                self.assertIn(
                    "EVAL_RESOLVED_HARNESS_COMMIT="
                    + pins["swiss-ai/lm-evaluation-harness"],
                    manifest,
                )
                self.assertIn(
                    "EVAL_HARNESS_OVERLAY=/opt/lm-eval-harness/swiss-ai/lm-evaluation-harness",
                    manifest,
                )

    def test_prebuilt_environment_refuses_a_harness_branch(self) -> None:
        completed, _ = _prebuilt_dry_run(
            "single", "--task", "hellaswag", "--model", "test/model",
            LM_EVAL_HARNESS_BRANCH="some-branch",
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("cannot be used with EVAL_PREBUILT_ENV=1", completed.stderr)

    def test_harness_pins_cover_both_forks(self) -> None:
        pins = [
            line.split()
            for line in (REPO_ROOT / "requirements/lm-eval-harness.txt").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(
            sorted(repo for repo, _ in pins),
            ["swiss-ai/lm-evaluation-harness", "ymetz/lm-evaluation-harness"],
        )
        for _, commit in pins:
            self.assertRegex(commit, r"^[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
