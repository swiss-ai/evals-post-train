import json
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


def _launch(*arguments: str) -> subprocess.CompletedProcess:
    """A dry run that may fail; the caller passes --logs-root."""
    return subprocess.run(
        ["bash", "scripts/launch_evaluations.sh", *arguments, "--debug"],
        cwd=REPO_ROOT,
        env=os.environ | {"WANDB_PROJECT": "launcher-test"},
        text=True,
        capture_output=True,
    )


def _run_config(logs_root: str) -> dict:
    configs = list(Path(logs_root).rglob("run_config.json"))
    assert len(configs) == 1, configs
    return json.loads(configs[0].read_text())


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

    def test_reasoning_effort_is_a_named_part_of_the_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = _launch(
                "single", "--task", "aime25", "--model", "openai/gpt-oss-120b",
                "--reasoning-effort", "high", "--logs-root", tmp,
            )
            output = completed.stdout + completed.stderr
            self.assertEqual(completed.returncode, 0, output)
            self.assertIn("Reasoning effort: high", output)
            self.assertIn("Name:   gpt-oss-120b-effort-high", output)
            self.assertIn("Chat:   true", output)
            config = _run_config(tmp)
        self.assertEqual(config["configuration"]["reasoning_effort"], "high")

    def test_no_reasoning_effort_keeps_the_run_config_unchanged(self) -> None:
        # run_config_matches() compares exactly: a new always-present key would make
        # every earlier result look foreign and rerun it.
        with tempfile.TemporaryDirectory() as tmp:
            completed = _launch(
                "single", "--task", "aime25", "--model", "test/model", "--logs-root", tmp,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            config = _run_config(tmp)
        self.assertNotIn("reasoning_effort", config["configuration"])

    def test_reasoning_effort_is_refused_where_it_cannot_reach_the_template(self) -> None:
        cases = {
            "openai backend": (
                ["--backend", "openai", "--api-base-url", "http://localhost:8000",
                 "--api-model-name", "test-model", "--reasoning-effort", "high"],
                "not supported with the openai backend",
            ),
            "sglang backend": (
                ["--backend", "sglang", "--reasoning-effort", "high"],
                "not supported with the sglang backend",
            ),
            "no chat template": (
                ["--no-chat-template", "--reasoning-effort", "high"],
                "drop --no-chat-template",
            ),
            "not a level": (["--reasoning-effort", "very high"], "expects a level"),
        }
        for name, (arguments, message) in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                completed = _launch(
                    "single", "--task", "aime25", "--model", "test/model",
                    *arguments, "--logs-root", tmp,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(message, completed.stdout + completed.stderr)

    def test_evaluate_passes_reasoning_effort_as_its_own_model_args_token(self) -> None:
        # The harness refuses a first --model_args token containing '{', and argparse gives
        # `--model_args='...'` exactly one value: the dict must be a second, space-separated token.
        sbatch = (REPO_ROOT / "scripts/evaluate.sbatch").read_text()
        self.assertIn(
            r"""CHAT_TEMPLATE_ARGS=" 'chat_template_args={\"reasoning_effort\":\"${REASONING_EFFORT}\"}'" """.rstrip(),
            sbatch,
        )
        for launcher in ("python -m lm_eval", "accelerate launch --num_processes=4  -m lm_eval"):
            self.assertIn(
                f"{launcher} --model $LM_EVAL_BACKEND --model_args '$COMMON_MODEL_ARGS'$CHAT_TEMPLATE_ARGS ",
                sbatch,
            )

    def test_harness_pins_only_the_swiss_ai_fork(self) -> None:
        pins = [
            line.split()
            for line in (REPO_ROOT / "requirements/lm-eval-harness.txt").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(
            sorted(repo for repo, _ in pins),
            ["swiss-ai/lm-evaluation-harness"],
        )
        for _, commit in pins:
            self.assertRegex(commit, r"^[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
