"""Offline checks of scripts/custom/run_tau2_bench.sh's argument handling (no network, model or venv)."""

import json
import subprocess
import tempfile
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

    def test_the_sampling_flags_reach_the_sampling_helper(self):
        # Past the sampling step the run needs uv, which this environment lacks: that is where it must stop.
        result = run("--domain", "retail", "--model", "m", "--api-base-url", "http://x/v1", "--sampling-profile", "thinking",
                     "--temperature", "0.6", "--top-p", "0.95", "--reasoning-effort", "high")
        self.assertNotIn("unrecognized arguments", result.stderr)
        self.assertIn("uv is required", result.stderr)
        bad = run("--domain", "retail", "--model", "m", "--api-base-url", "http://x/v1", "--sampling-profile", "nope")
        self.assertNotEqual(bad.returncode, 0)
        self.assertNotIn("uv is required", bad.stderr)

    def test_unknown_argument_is_refused(self):
        result = run("--domain", "retail", "--model", "m", "--bogus")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown argument: --bogus", result.stderr)


class RetrievalDefaultTest(unittest.TestCase):
    """Runs the script in prebuilt mode against a stub `tau2` that records its arguments."""

    def _tau2_args(self, domain, *extra):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d, "upstream")
            (root / ".venv/bin").mkdir(parents=True)
            (root / "data").mkdir()
            (root / "data/x").write_text("")
            stub = root / ".venv/bin/tau2"
            stub.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {d}/args\nexit 0\n')
            stub.chmod(0o755)
            ref = Path(d, "ref")
            ref.write_text("v2.0.0 " + "a" * 40 + "\n")
            subprocess.run(
                ["bash", "scripts/custom/run_tau2_bench.sh", "--domain", domain, "--model", "m", "--api-base-url", "http://x/v1",
                 "--workdir", f"{d}/work", *extra],
                cwd=REPO, capture_output=True, text=True, timeout=60,
                env={"PATH": "/usr/bin:/bin", "HOME": d, "EVAL_PREBUILT_ENV": "1",
                     "TAU2_UPSTREAM_DIR": str(root), "TAU2_UPSTREAM_REF_FILE": str(ref)},
            )
            return Path(d, "args").read_text().split()

    def test_banking_defaults_to_the_offline_bm25(self):
        args = self._tau2_args("banking_knowledge")
        self.assertEqual(args[args.index("--retrieval-config") + 1], "bm25")

    def test_a_named_config_wins_and_other_domains_get_none(self):
        args = self._tau2_args("banking_knowledge", "--retrieval-config", "bm25_grep")
        self.assertEqual(args[args.index("--retrieval-config") + 1], "bm25_grep")
        self.assertNotIn("--retrieval-config", self._tau2_args("retail"))


class SummaryTest(unittest.TestCase):
    """The summary of results.json, with a stub tau2 that writes one."""

    def _run(self, sims, *extra):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d, "upstream")
            (root / ".venv/bin").mkdir(parents=True)
            (root / "data").mkdir()
            (root / "data/x").write_text("")
            results = Path(d, "results.json")
            results.write_text(json.dumps({"simulations": sims}))
            stub = root / ".venv/bin/tau2"
            stub.write_text(f'#!/bin/sh\nmkdir -p "$TAU2_DATA_DIR/simulations/t" && cp {results} "$TAU2_DATA_DIR/simulations/t/results.json"\n')
            stub.chmod(0o755)
            ref = Path(d, "ref")
            ref.write_text("v2.0.0 " + "a" * 40 + "\n")
            return subprocess.run(
                ["bash", "scripts/custom/run_tau2_bench.sh", "--domain", "retail", "--model", "m", "--api-base-url", "http://x/v1",
                 "--workdir", f"{d}/work", "--save-to", "t", *extra],
                cwd=REPO, capture_output=True, text=True, timeout=60,
                env={"PATH": "/usr/bin:/bin", "HOME": d, "EVAL_PREBUILT_ENV": "1",
                     "TAU2_UPSTREAM_DIR": str(root), "TAU2_UPSTREAM_REF_FILE": str(ref)},
            )

    @staticmethod
    def _sim(task, reward, why="user_stop"):
        return {"task_id": task, "termination_reason": why, "reward_info": {"reward": reward} if reward is not None else None}

    def test_a_complete_result_exits_zero(self):
        result = self._run([self._sim("0", 1.0), self._sim("1", 0.0)], "--num-tasks", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout[result.stdout.index("{"):])
        self.assertEqual((summary["avg_reward"], summary["n_unscored"], summary["n_missing"]), (0.5, 0, 0))

    def test_a_task_without_a_reward_is_reported_and_fails_the_run(self):
        result = self._run([self._sim("0", 1.0), self._sim("2", None, "infrastructure_error")], "--num-tasks", "2")
        self.assertEqual(result.returncode, 1)
        self.assertIn('"n_unscored": 1', result.stdout)
        self.assertIn("task 2: no reward (infrastructure_error)", result.stderr)

    def test_a_simulation_missing_from_the_file_fails_the_run(self):
        result = self._run([self._sim("0", 1.0)], "--num-tasks", "3")
        self.assertEqual(result.returncode, 1)
        self.assertIn("2 of 3 simulations are not in results.json", result.stderr)


if __name__ == "__main__":
    unittest.main()
