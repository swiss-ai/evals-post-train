"""Offline checks of the standalone AA runners' environment handling (no network, model or venv)."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run(script, *args, **env):
    return subprocess.run(
        ["bash", f"scripts/aaii/{script}", *args],
        cwd=REPO,
        env={"PATH": "/usr/bin:/bin", "HOME": env.pop("HOME"), **env},
        capture_output=True,
        text=True,
        timeout=60,
    )


class GdpvalTest(unittest.TestCase):
    def test_a_failed_run_exits_non_zero(self):
        # A venv whose python fails stands in for the run: its exit code must come through.
        with tempfile.TemporaryDirectory() as d:
            venv = Path(d, "venv/bin")
            venv.mkdir(parents=True)
            for name in ("python", "python3"):
                (venv / name).write_text("#!/bin/sh\nexit 3\n")
                (venv / name).chmod(0o755)
            result = run(
                "run_gdpval.sh", "--model", "m", "--sandbox-backend", "local", "--num-tasks", "1",
                "--no-grade", "--workdir", os.path.join(d, "work"),
                HOME=d, EVAL_PREBUILT_ENV="1", GDPVAL_VENV=str(venv.parent),
            )
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)


class Tau2Test(unittest.TestCase):
    def test_a_ref_other_than_the_locks_pin_is_refused_before_installing(self):
        # A full commit needs no lookup: outside the prebuilt image the lock's pin is what runs.
        with tempfile.TemporaryDirectory() as d:
            result = run(
                "run_tau2_bench.sh", "--model", "m", "--git-ref", "0" * 40, "--workdir", os.path.join(d, "work"),
                HOME=d,
            )
            self.assertFalse(Path(d, "work/tau2-bench").exists())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requirements/tau2-runtime.lock has tau2-bench@", result.stderr)
        self.assertIn("Bump requirements/tau2-runtime.txt", result.stderr)


if __name__ == "__main__":
    unittest.main()
