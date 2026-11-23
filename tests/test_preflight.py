"""EVAL_PREFLIGHT=1: every runner a direct-serving job runs stops after checking its arguments and keys."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENDPOINT = ["--api-base-url", "http://127.0.0.1:1/v1"]
# Each runner with the least that passes its argument checks.
RUNNERS = {
    "scripts/custom/run_bfcl.sh": ["--version", "v4", "--model", "m", *ENDPOINT],
    "scripts/custom/run_tau2_bench.sh": ["--domain", "airline", "--model", "m", *ENDPOINT],
    "scripts/inspect_ai/run_inspect_eval.sh": ["--task", "gsm8k", "--model", "m", *ENDPOINT],
    "scripts/harbor/run_harbor_eval.sh": ["--model", "m", *ENDPOINT],
    "scripts/evalscope/run_evalscope_eval.sh": ["--benchmark", "live_code_bench", "--model", "m", *ENDPOINT],
    "scripts/lm_eval_harness/launch_evaluations.sh": ["single", "--task", "gsm8k", "--model", "m", "--backend", "openai", "--api-model-name", "m", *ENDPOINT],
}


class PreflightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A copy of the scripts without scripts/secrets.env, which would give every run a key.
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        shutil.copytree(REPO / "scripts", cls.root / "scripts", ignore=shutil.ignore_patterns("secrets.env", "__pycache__"))
        shutil.copytree(REPO / "configs", cls.root / "configs")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _run(self, runner, *extra, **env):
        return subprocess.run(
            ["bash", str(self.root / runner), *RUNNERS[runner], *extra],
            cwd=self.root, capture_output=True, text=True, timeout=120,
            env={"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(self.root), "EVAL_PREFLIGHT": "1", **env},
        )

    def test_each_runner_stops_after_its_checks_without_installing_or_submitting(self):
        for runner in RUNNERS:
            with self.subTest(runner=runner):
                result = self._run(runner, *([] if "tau2" in runner else ["--no-wandb"]))
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f"preflight ok: {Path(runner).name}", result.stdout)

    def test_a_requested_wandb_upload_without_a_key_fails_the_preflight(self):
        for runner in RUNNERS:
            if "tau2" in runner:
                continue  # no W&B upload
            with self.subTest(runner=runner):
                result = self._run(runner, "--wandb-project", "p")
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("WANDB_API_KEY is not set", result.stderr)
                self.assertNotIn("preflight ok", result.stdout)

    def test_a_requested_wandb_upload_with_a_key_passes(self):
        for runner in RUNNERS:
            if "tau2" in runner:
                continue
            with self.subTest(runner=runner):
                result = self._run(runner, "--wandb-project", "p", WANDB_API_KEY="k")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_a_key_that_is_only_in_secrets_env_counts(self):
        secrets = self.root / "scripts/secrets.env"
        secrets.write_text("WANDB_API_KEY=from-the-file\n")
        self.addCleanup(secrets.unlink)
        for runner in RUNNERS:
            if "tau2" in runner:
                continue
            with self.subTest(runner=runner):
                result = self._run(runner, "--wandb-project", "p")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
