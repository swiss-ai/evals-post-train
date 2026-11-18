import io
import shutil
import subprocess
import tempfile
import tomllib
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from swiss_ai_model_launch.launchers import framework as ml_framework

from scripts.direct_serving import job


def _launch_args(nodes: int = 2):
    env = Path(tempfile.mkdtemp()) / "env.toml"
    env.write_text('image = "x"\n')
    return job.build_launch_args(
        model="/store/models/chonk iter",
        served_model_name="chonk",
        nodes=nodes,
        environment=str(env),
        framework="vllm",
        framework_args="--tensor-parallel-size 1",
        account="infra01",
        partition="normal",
        time="04:00:00",
        job_name="serve-eval-test",
    )


def _render(evals: list[str], nodes: int = 2, **kwargs) -> str:
    return job.render(
        _launch_args(nodes),
        image_commit="abc123",
        mounts=["/repo", "/scratch/${USER}"],
        evals=evals,
        ready_timeout=600,
        **kwargs,
    )


class LaunchArgsTests(unittest.TestCase):
    def test_server_is_reachable_only_from_the_job(self) -> None:
        args = _launch_args()

        self.assertTrue(args.disable_opentela)
        self.assertTrue(args.disable_metrics)
        self.assertIsNone(args.telemetry_endpoint)
        self.assertEqual(args.total_nodes, 2)

    def test_model_and_name_are_passed_to_the_server_quoted(self) -> None:
        args = _launch_args()

        self.assertTrue(args.framework_args.startswith("--model '/store/models/chonk iter' "))
        self.assertIn("--served-model-name chonk ", args.framework_args)
        self.assertTrue(args.framework_args.endswith("--tensor-parallel-size 1"))


class RenderTests(unittest.TestCase):
    def test_header_has_the_nodes_and_time_of_the_launch(self) -> None:
        script = _render(["echo hi"], reservation="res", qos="highprio")

        self.assertTrue(script.startswith("#!/bin/bash\n#SBATCH --job-name=serve-eval-test\n"))
        for option in ("--nodes=2", "--time=04:00:00", "--exclusive", "--reservation=res", "--qos=highprio"):
            self.assertIn(f"#SBATCH {option}\n", script)

    def test_model_launch_footer_is_replaced(self) -> None:
        script = _render(["echo hi"])

        self.assertNotIn(ml_framework._render_footer(), script)
        self.assertNotIn("Master finished at", script)
        self.assertIn("critical_pids+=($!)", script)  # its serving steps are kept

    def test_each_eval_runs_in_order_in_the_eval_image(self) -> None:
        script = _render(["echo one", "echo 'two words'"])

        first = script.index("run_eval 1 ")
        second = script.index("run_eval 2 ")
        self.assertIn(f"cd {job.REPO_ROOT} && export PATH={job.INLINE_SLURM_DIR}:$PATH && echo one'", script)
        self.assertIn("&& echo '\"'\"'two words'\"'\"''", script)
        self.assertLess(first, second)
        self.assertIn('--environment="$eval_edf" bash -c', script)
        self.assertIn("export EVAL_API_BASE_URL=http://127.0.0.1:8080/v1", script)
        self.assertIn("export TARGET_API_KEY=local", script)
        self.assertNotIn("export CSCS_SERVING_API", script)
        self.assertIn("curl -sf --noproxy '*' --max-time 5 http://127.0.0.1:8080/health", script)
        self.assertTrue(script.rstrip().endswith('exit "$failed"'))

    def test_changed_model_launch_footer_is_refused(self) -> None:
        with patch.object(ml_framework, "render_master", return_value="echo a script without it\n"):
            with self.assertRaisesRegex(RuntimeError, "last section"):
                _render(["echo hi"])

    @unittest.skipUnless(shutil.which("bash"), "bash is needed")
    def test_script_is_valid_bash(self) -> None:
        for nodes in (1, 2):
            with self.subTest(nodes=nodes):
                result = subprocess.run(
                    ["bash", "-n"],
                    input=_render(["echo one", "echo two"], nodes=nodes),
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


class EvalImageTests(unittest.TestCase):
    def test_edf_starts_the_image_of_the_commit_with_the_mounts_once_each(self) -> None:
        edf = job.eval_edf("abc123", ["/repo", "/scratch/${USER}", "/repo"])

        self.assertIn('image = "ghcr.io#swiss-ai/evals-post-train-eval:abc123"', edf)
        self.assertEqual(edf.count('"/repo"'), 1)
        self.assertIn('"/scratch/${USER}"', edf)
        tomllib.loads(edf)

    def test_job_writes_the_edf_before_the_first_eval(self) -> None:
        script = _render(["echo hi"])

        self.assertLess(script.index("EVAL_EDF_EOF'"), script.index("run_eval 1 "))
        self.assertIn('image = "ghcr.io#swiss-ai/evals-post-train-eval:abc123"', script)

    def test_image_commit_is_the_last_one_that_changed_its_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)

            def git(*args: str) -> str:
                return subprocess.run(
                    ["git", "-C", tmp, "-c", "user.name=t", "-c", "user.email=t@t", *args],
                    capture_output=True, text=True, check=True,
                ).stdout.strip()  # fmt: skip

            git("init", "-q")
            (repo / "requirements").mkdir()
            (repo / "requirements" / "a.lock").write_text("1")
            git("add", ".")
            git("commit", "-qm", "inputs")
            inputs = git("rev-parse", "HEAD")
            (repo / "README.md").write_text("docs only")
            git("add", ".")
            git("commit", "-qm", "docs")

            self.assertNotEqual(git("rev-parse", "HEAD"), inputs)
            self.assertEqual(job.eval_image_commit(repo), inputs)

    def test_published_is_false_for_a_missing_image_and_none_when_ghcr_is_unreachable(self) -> None:
        token = io.BytesIO(b'{"token": "t"}')
        missing = urllib.error.HTTPError("u", 404, "not found", {}, None)
        with patch.object(job.urllib.request, "urlopen", side_effect=[token, missing]):
            self.assertIs(job.image_published("abc123"), False)
        with patch.object(job.urllib.request, "urlopen", side_effect=urllib.error.URLError("down")):
            self.assertIsNone(job.image_published("abc123"))


@unittest.skipUnless(shutil.which("bash"), "bash is needed")
class InlineSlurmTests(unittest.TestCase):
    def _run(self, tool: str, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [str(job.INLINE_SLURM_DIR / tool), *args], capture_output=True, text=True, check=False
        )

    def test_sbatch_runs_the_script_now_and_prints_job_id_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "batch.sh"
            script.write_text('echo "ran with $FOO and $1"\n')
            result = self._run(
                "sbatch", "-p", "highprio", "--time=01:00:00", "--export=FOO=bar,ALL", str(script), "arg"
            )

        self.assertEqual(result.stdout, "0\n")  # a caller reads the id from stdout
        self.assertEqual(result.stderr, "ran with bar and arg\n")
        self.assertEqual(result.returncode, 0)

    def test_sbatch_passes_the_scripts_failure_on(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "batch.sh"
            script.write_text("exit 3\n")

            self.assertEqual(self._run("sbatch", str(script)).returncode, 3)

    def test_srun_drops_slurm_flags_and_runs_the_command_here(self) -> None:
        result = self._run("srun", "--environment", "/edf.toml", "-N", "1", "--overlap", "echo", "hi")

        self.assertEqual((result.stdout, result.returncode), ("hi\n", 0))


class MainTests(unittest.TestCase):
    def test_prints_the_script_without_submitting(self) -> None:
        env = Path(tempfile.mkdtemp()) / "env.toml"
        env.write_text('image = "x"\n')
        argv = [
            "--model", "/store/m", "--environment", str(env), "--eval-image-commit", "abc123",
            "--account", "infra01", "--eval", "echo hi", "--mount", "/extra",
        ]  # fmt: skip
        with (
            patch("sys.stdout.write") as write,
            patch.object(job.subprocess, "run") as run,
            patch.object(job, "image_published", return_value=True),
        ):
            self.assertEqual(job.main(argv), 0)

        run.assert_not_called()
        script = write.call_args.args[0]
        self.assertIn("$PATH && echo hi'", script)
        self.assertIn('    "/extra"\n', script)

    def test_missing_image_stops_before_a_script_is_made(self) -> None:
        env = Path(tempfile.mkdtemp()) / "env.toml"
        env.write_text('image = "x"\n')
        argv = ["--model", "/store/m", "--environment", str(env), "--account", "a", "--eval", "x"]
        with (
            patch.object(job, "eval_image_commit", return_value="abc123"),
            patch.object(job, "image_published", return_value=False),
            patch("sys.stdout.write") as write,
        ):
            with self.assertRaisesRegex(SystemExit, "abc123 does not exist"):
                job.main(argv)

        write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
