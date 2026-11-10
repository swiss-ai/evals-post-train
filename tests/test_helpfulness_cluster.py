"""Slurm wrapper wiring tests; mock CLI calls are not scientific acceptance."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.helpfulness import native


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "helpfulness" / "run_acceptance.sh"
BATCH = ROOT / "scripts" / "helpfulness" / "run_acceptance.sbatch"


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux Bash required")
class ClusterWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="helpfulness-cluster-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.log = self.directory / "calls.jsonl"
        self.python = self.directory / "mock python"
        self.python.write_text(f"#!{sys.executable}\n" + """
import json, os, pathlib, subprocess, sys
if sys.argv[1] == '-c':
    raise SystemExit(subprocess.run([sys.executable, *sys.argv[1:]]).returncode)
args = sys.argv[1:]
with open(os.environ['MOCK_LOG'], 'a') as stream:
    stream.write(json.dumps(args) + '\\n')
command = args[2]
if command == os.environ.get('MOCK_FAIL'):
    raise SystemExit(7)
if command == 'plan':
    config = pathlib.Path(args[args.index('--config') + 1])
    output = pathlib.Path(args[args.index('--output') + 1])
    output.write_text(json.dumps({'kind': config.read_text().strip()}))
""", encoding="utf-8")
        self.python.chmod(0o700)
        self.config = self.directory / "config with spaces.toml"
        self.endpoint = self.directory / "endpoint.toml"
        self.endpoint.write_text("mock endpoint, not a deployment", encoding="utf-8")
        self.output = self.directory / "output with spaces"
        self.environment = os.environ.copy()
        for name in ("EVAL_ENV_MANIFEST", "EVAL_BASE_ENV", "EVAL_HARNESS_OVERLAY"):
            self.environment.pop(name, None)
        self.environment.update(EVAL_REPO_ROOT=str(ROOT), EVAL_PYTHON=str(self.python),
                                MOCK_LOG=str(self.log))

    def invoke(self, *args):
        return subprocess.run(["bash", str(WORKER), *map(str, args)],
                              env=self.environment, text=True, capture_output=True)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_shell_syntax(self):
        for path in (WORKER, BATCH, ROOT / "scripts/helpfulness/prepare_regression.sh",
                     ROOT / "scripts/helpfulness/prepare_regression.sbatch",
                     ROOT / "scripts/lm_eval_harness/build_eval_env.sh",
                     ROOT / "scripts/helpfulness/runtime_isolation.sh"):
            result = subprocess.run(["bash", "-n", str(path)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_replay_and_quoted_paths(self):
        self.config.write_text("scorer_replay", encoding="utf-8")
        result = self.invoke("replay", self.config, self.output)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual([call[2] for call in calls], ["plan", "run"])
        self.assertIn(str(self.config), calls[0])
        self.assertNotIn("--candidate", calls[1])

    def test_regeneration_and_calibration(self):
        for mode, kind, producer, expected in (
            ("regenerate", "reference_regeneration", "regenerate", ["plan", "regenerate", "run"]),
            ("calibrate", "judge_calibration", "judge", ["plan", "judge", "run", "freeze"]),
        ):
            with self.subTest(mode=mode):
                self.config.write_text(kind, encoding="utf-8")
                result = self.invoke(mode, self.config, self.directory / mode, self.endpoint)
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = self.calls()[-len(expected):]
                self.assertEqual([call[2] for call in calls], expected)
                self.assertIn(str(self.endpoint), calls[1])
                self.assertEqual(calls[1][2], producer)
                self.assertIn("--candidate", calls[2])

    def test_wrong_lane_makes_no_model_calls(self):
        self.config.write_text("judge_calibration", encoding="utf-8")
        result = self.invoke("regenerate", self.config, self.output, self.endpoint)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([call[2] for call in self.calls()], ["plan"])

    def test_failure_stops_downstream_steps(self):
        self.config.write_text("judge_calibration", encoding="utf-8")
        for failure, expected in (
            ("judge", ["plan", "judge"]),
            ("run", ["plan", "judge", "run"]),
        ):
            with self.subTest(failure=failure):
                self.environment["MOCK_FAIL"] = failure
                result = self.invoke("calibrate", self.config, self.directory / failure, self.endpoint)
                self.assertEqual(result.returncode, 7, result.stderr)
                self.assertEqual([call[2] for call in self.calls()[-len(expected):]], expected)

    def test_existing_output_cannot_be_reused(self):
        self.config.write_text("scorer_replay", encoding="utf-8")
        self.output.mkdir()
        result = self.invoke("replay", self.config, self.output)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls(), [])

    def test_command_preserves_external_plan(self):
        result = self.invoke("command", "run", "--plan", "sealed plan.json",
                             "--candidate", "external.json", "--output", "new report")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls()[0], ["-m", "scripts.helpfulness.acceptance", "run",
                                         "--plan", "sealed plan.json", "--candidate",
                                         "external.json", "--output", "new report"])

    def test_manifest_selects_prepared_runtime(self):
        base = self.directory / "base"
        (base / "bin").mkdir(parents=True)
        (base / ".complete").touch()
        (base / "bin" / "python").symlink_to(self.python)
        overlay = self.directory / "overlay"
        overlay.mkdir()
        (overlay / ".complete").touch()
        manifest = self.directory / "environment.sh"
        manifest.write_text(f"EVAL_BASE_ENV='{base}'\nEVAL_HARNESS_OVERLAY='{overlay}'\n",
                            encoding="utf-8")
        self.environment["EVAL_ENV_MANIFEST"] = str(manifest)
        self.environment["EVAL_PYTHON"] = "/intentionally/not/used"
        result = self.invoke("command", "run", "--plan", "plan", "--output", "report")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_batch_requires_container_before_srun(self):
        self.environment.pop("EVAL_CONTAINER_CONFIG", None)
        result = subprocess.run(["bash", str(BATCH), "command", "run"], env=self.environment,
                                text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("EVAL_CONTAINER_CONFIG", result.stderr)

    def test_batch_enters_requested_container_and_forwards_arguments(self):
        srun = self.directory / "srun"
        srun.write_text(f"#!{sys.executable}\n" + """
import os, subprocess, sys
args = sys.argv[1:]
assert '--environment=approved-eval' in args
assert '--ntasks=1' in args and '--export=ALL' in args
raise SystemExit(subprocess.run(args[args.index('bash'):]).returncode)
""", encoding="utf-8")
        srun.chmod(0o700)
        self.environment["PATH"] = f"{self.directory}:{self.environment['PATH']}"
        self.environment["EVAL_CONTAINER_CONFIG"] = "approved-eval"
        result = subprocess.run(["bash", str(BATCH), "command", "run", "--plan",
                                 "sealed plan.json", "--output", "report"],
                                env=self.environment, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("sealed plan.json", self.calls()[0])

    def test_harness_pin_accepts_windows_and_linux_line_endings(self):
        checkout = self.directory / "checkout"
        requirements = checkout / "requirements"
        requirements.mkdir(parents=True)
        expected = "5ed4beb1c26ebf44906fd003aee7d64fab399af9"
        for newline in ("\r\n", "\n"):
            (requirements / "lm-eval-harness.txt").write_bytes(
                f"# source{newline}swiss-ai/lm-evaluation-harness {expected}{newline}".encode()
            )
            with self.subTest(newline=repr(newline)), patch.object(
                native, "__file__", str(checkout / "scripts/helpfulness/native.py")
            ):
                self.assertEqual(native.harness_pin(), expected)

    def test_builder_normalizes_commit_before_any_remote_resolution(self):
        binary = self.directory / "bin"
        binary.mkdir()
        python = binary / "python"
        python.write_text(f"#!{sys.executable}\n" + """
from pathlib import Path
import os, sys
args = sys.argv[1:]
with open(os.environ['BUILDER_CALLS'], 'a') as stream:
    stream.write(repr(args) + '\\n')
if os.environ.get('EVAL_ENV_ISOLATED') == 'true':
    assert '/container/dist-packages' not in os.environ.get('PYTHONPATH', '')
    assert 'PYTHONHOME' not in os.environ
    assert '/container/torch/lib' not in os.environ.get('LD_LIBRARY_PATH', '')
if args == ['-VV']:
    print('Synthetic test Python; no dependencies installed')
elif args[:2] == ['-m', 'venv']:
    base = Path(args[-1])
    (base / 'bin').mkdir(parents=True)
    (base / 'bin/python').symlink_to(os.environ['FAKE_PYTHON'])
elif args[:3] == ['-m', 'pip', 'install'] or args[:1] == ['-c']:
    pass
else:
    raise SystemExit('Unexpected test command')
""", encoding="utf-8")
        python.chmod(0o700)
        git = binary / "git"
        git.write_text("#!/bin/bash\necho 'Unexpected network resolution' >&2\nexit 66\n",
                       encoding="utf-8")
        git.chmod(0o700)
        requirements = self.directory / "runtime.txt"
        requirements.write_text("# synthetic, no installation\n", encoding="utf-8")
        edf = self.directory / "environment.toml"
        edf.write_text("# synthetic, no container\n", encoding="utf-8")
        environment = dict(self.environment,
                           PATH=f"{binary}:{self.environment['PATH']}",
                           FAKE_PYTHON=str(python), EVAL_HARNESS_REF=native.harness_pin() + "\r\n",
                           EVAL_HARNESS_REPO="swiss-ai/lm-evaluation-harness",
                           EVAL_ENV_CACHE_ROOT=str(self.directory / "runtime-cache"),
                           EVAL_ENV_MANIFEST=str(self.directory / "prepared.sh"),
                           EVAL_RUNTIME_REQUIREMENTS=str(requirements), EVAL_CONTAINER_CONFIG=str(edf),
                           BUILDER_CALLS=str(self.directory / "builder-calls.txt"))
        environment.pop("EVAL_APERTUS2_REQUIREMENTS", None)
        environment["EVAL_ENV_ISOLATED"] = "false"
        result = subprocess.run(["bash", str(ROOT / "scripts/lm_eval_harness/build_eval_env.sh")],
                                env=environment, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"EVAL_RESOLVED_HARNESS_COMMIT={native.harness_pin()}",
                      (self.directory / "prepared.sh").read_text())
        inherited_manifest = (self.directory / "prepared.sh").read_text()
        self.assertIn("'--system-site-packages'", (self.directory / "builder-calls.txt").read_text())
        environment.update(EVAL_ENV_ISOLATED="true", PYTHONPATH="/container/dist-packages",
                           PYTHONHOME="/invalid/container/python",
                           LD_LIBRARY_PATH="/container/torch/lib:/usr/local/cuda/lib64")
        result = subprocess.run(["bash", str(ROOT / "scripts/lm_eval_harness/build_eval_env.sh")],
                                env=environment, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        isolated_manifest = (self.directory / "prepared.sh").read_text()
        self.assertIn("EVAL_ENV_ISOLATED=true", isolated_manifest)
        self.assertNotEqual(inherited_manifest, isolated_manifest)
        calls = (self.directory / "builder-calls.txt").read_text().splitlines()
        venv_calls = [line for line in calls if "'venv'" in line]
        self.assertEqual(len(venv_calls), 2)
        self.assertNotIn("--system-site-packages", venv_calls[1])

    def test_isolation_removes_container_import_and_library_paths(self):
        helper = ROOT / "scripts/helpfulness/runtime_isolation.sh"
        script = (
            'export PYTHONPATH=/container/dist-packages PYTHONHOME=/container/python; '
            'export LD_LIBRARY_PATH=/container/site-packages/torch/lib:/other/torch/lib:/usr/local/cuda/lib64; '
            f'source "{helper}"; '
            'test -z "${PYTHONPATH+x}" && test -z "${PYTHONHOME+x}" && '
            'test "$PYTHONNOUSERSITE" = 1 && '
            'test "$LD_LIBRARY_PATH" = /usr/local/cuda/lib64'
        )
        result = subprocess.run(["bash", "-c", script], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
