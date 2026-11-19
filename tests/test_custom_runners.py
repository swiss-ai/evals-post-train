"""Offline checks of scripts/custom/run_tau2_bench.sh's argument handling (no network, model or venv)."""

import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run(*args):
    return subprocess.run(
        ["bash", "scripts/custom/run_tau2_bench.sh", *args],
        cwd=REPO,
        env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
        capture_output=True,
        text=True,
        timeout=60,
    )


class CustomTau2Test(unittest.TestCase):
    def test_help_documents_the_ref_and_the_cache(self):
        result = run("--help")
        self.assertEqual(result.returncode, 0)
        for flag in ("--git-ref", "--repo", "--domain", "--cache-dir", "--user-llm"):
            self.assertIn(flag, result.stdout)

    def test_domain_and_model_are_required(self):
        self.assertIn("Missing --domain", run("--model", "m").stderr)
        self.assertIn("Missing --model", run("--domain", "retail").stderr)

    def test_domain_and_save_to_are_plain_names(self):
        self.assertIn("domain name", run("--domain", "a b; rm", "--model", "m").stderr)
        self.assertIn("plain name", run("--domain", "retail", "--model", "m", "--save-to", "../x").stderr)

    def test_unknown_argument_is_refused(self):
        result = run("--domain", "retail", "--model", "m", "--bogus")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown argument: --bogus", result.stderr)


if __name__ == "__main__":
    unittest.main()
