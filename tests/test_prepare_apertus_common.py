import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "scripts/lm_eval_harness/prepare_apertus_common.sh"
PLUGIN_PIN = "requirements/apertus2-vllm-plugin.txt"


def _prepare(*arguments: str, cache: str) -> subprocess.CompletedProcess:
    env = {"PATH": os.environ["PATH"], "EVAL_DRY_RUN": "true", "EVAL_ENV_CACHE_ROOT": cache}
    return subprocess.run(["bash", SCRIPT, *arguments], cwd=REPO_ROOT, env=env, capture_output=True, text=True)


def _commit(requirements: str) -> str:
    return (REPO_ROOT / requirements).read_text().rsplit("@", 1)[1].strip()


class PrepareApertusCommonTest(unittest.TestCase):
    def test_default_is_the_pin_of_the_lm_eval_apertus2_runs(self):
        with tempfile.TemporaryDirectory() as cache:
            done = _prepare(cache=cache)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(done.stdout.strip(), f"{cache}/apertus-common/{_commit('requirements/apertus2.txt')}")

    def test_the_vllm_plugin_pin_names_its_own_wheel(self):
        self.assertRegex(_commit(PLUGIN_PIN), re.compile(r"^[0-9a-f]{40}$"))
        with tempfile.TemporaryDirectory() as cache:
            done = _prepare(PLUGIN_PIN, cache=cache)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(done.stdout.strip(), f"{cache}/apertus-common/{_commit(PLUGIN_PIN)}")

    def test_a_file_without_a_commit_pin_is_refused(self):
        with tempfile.TemporaryDirectory() as cache:
            bad = Path(cache) / "bad.txt"
            bad.write_text("apertus-common @ git+https://github.com/swiss-ai/apertus-common.git@main\n")
            done = _prepare(str(bad), cache=cache)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("full commit sha", done.stderr)
            empty = Path(cache) / "empty.txt"
            empty.write_text("# nothing\n")
            done = _prepare(str(empty), cache=cache)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("has no 'apertus-common @ git+", done.stderr)


if __name__ == "__main__":
    unittest.main()
