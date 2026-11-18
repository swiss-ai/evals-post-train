import contextlib
import io
import re
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


def _env_file(test: unittest.TestCase) -> str:
    """A serving EDF in a directory that goes away with the test."""
    directory = tempfile.TemporaryDirectory()
    test.addCleanup(directory.cleanup)
    path = Path(directory.name) / "env.toml"
    path.write_text('image = "x"\n')
    return str(path)


def _launch_args(test: unittest.TestCase, nodes: int = 2):
    return job.build_launch_args(
        model="/store/models/chonk iter",
        served_model_name="chonk",
        nodes=nodes,
        environment=_env_file(test),
        framework="vllm",
        framework_args="--tensor-parallel-size 1",
        account="infra01",
        partition="normal",
        time="04:00:00",
        job_name="serve-eval-test",
    )


def _render(test: unittest.TestCase, evals: list[str], nodes: int = 2, **kwargs) -> str:
    return job.render(
        _launch_args(test, nodes),
        image_commit="abc123",
        mounts=["/repo", "/scratch/${USER}"],
        evals=evals,
        ready_timeout=600,
        **kwargs,
    )


class LaunchArgsTests(unittest.TestCase):
    def test_server_is_reachable_only_from_the_job(self) -> None:
        args = _launch_args(self)

        self.assertTrue(args.disable_opentela)
        self.assertTrue(args.disable_metrics)
        self.assertIsNone(args.telemetry_endpoint)
        self.assertEqual(args.total_nodes, 2)

    def test_model_and_name_are_passed_to_the_server_quoted(self) -> None:
        self.assertEqual(
            _launch_args(self).framework_args,
            "--model '/store/models/chonk iter' --served-model-name chonk --host 0.0.0.0 --tensor-parallel-size 1",
        )


class RenderTests(unittest.TestCase):
    def test_header_has_the_nodes_and_time_of_the_launch(self) -> None:
        script = _render(self, ["echo hi"], reservation="res", qos="highprio")

        for option in ("--nodes=2", "--time=04:00:00", "--exclusive", "--reservation=res", "--qos=highprio"):
            self.assertIn(f"#SBATCH {option}\n", script)

    def test_model_launch_footer_is_replaced(self) -> None:
        script = _render(self, ["echo hi"])

        self.assertNotIn(ml_framework._render_footer(), script)
        self.assertNotIn("Master finished at", script)
        self.assertIn("critical_pids+=($!)", script)  # its serving steps are kept

    def test_each_eval_runs_in_order_from_the_checkout_in_the_eval_image(self) -> None:
        script = _render(self, ["echo one", "echo 'two words'"])

        self.assertLess(script.index("run_eval 1 "), script.index("run_eval 2 "))
        self.assertIn(f"cd {job.REPO_ROOT} && export PATH={job.INLINE_SLURM_DIR}:$PATH && echo one'", script)
        self.assertIn("&& echo '\"'\"'two words'\"'\"''", script)  # the command's quoting survives
        self.assertIn('--environment="$eval_edf" bash -c', script)
        self.assertIn("export EVAL_API_BASE_URL=http://127.0.0.1:8080/v1", script)

    def test_endpoint_key_is_a_placeholder_that_never_replaces_a_key_that_is_set(self) -> None:
        script = _render(self, ["echo hi"])

        self.assertIn('export TARGET_API_KEY="${TARGET_API_KEY:-local}"', script)
        self.assertNotIn("export CSCS_SERVING_API", script)

    def test_eval_image_is_pulled_while_the_model_loads(self) -> None:
        script = _render(self, ["echo hi"])

        pull = script.index('--environment="$eval_edf" true &')
        self.assertLess(script.index("EVAL_EDF_EOF'"), pull)
        self.assertLess(pull, script.index("until ready"))
        self.assertLess(script.index("The model server is ready after"), script.index('wait "$prefetch_pid"'))
        self.assertLess(script.index('wait "$prefetch_pid"'), script.index("run_eval 1 "))

    def test_changed_model_launch_footer_is_refused(self) -> None:
        with patch.object(ml_framework, "render_master", return_value="echo a script without it\n"):
            with self.assertRaisesRegex(RuntimeError, "last section"):
                _render(self, ["echo hi"])

    @unittest.skipUnless(shutil.which("bash"), "bash is needed")
    def test_script_is_valid_bash(self) -> None:
        for nodes in (1, 2):
            with self.subTest(nodes=nodes):
                result = subprocess.run(
                    ["bash", "-n"],
                    input=_render(self, ["echo one", "echo two"], nodes=nodes),
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
        script = _render(self, ["echo hi"])

        self.assertLess(script.index("EVAL_EDF_EOF'"), script.index("run_eval 1 "))

    def test_image_inputs_are_the_paths_ci_builds_the_image_for(self) -> None:
        workflow = (job.REPO_ROOT / ".github" / "workflows" / "eval-image.yml").read_text()
        paths = re.search(r"push:\n\s+paths:\n((?:\s+- .+\n)+)", workflow)
        self.assertIsNotNone(paths)
        listed = {line.strip()[2:].removesuffix("/**") for line in paths.group(1).splitlines()}

        self.assertEqual(listed, set(job.IMAGE_INPUTS))

    def _repo(self) -> tuple[Path, "callable"]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        repo = Path(directory.name) / "repo"
        repo.mkdir()

        def git(*args: str, cwd: Path = repo) -> str:
            return subprocess.run(
                ["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                capture_output=True, text=True, check=True,
            ).stdout.strip()  # fmt: skip

        git("init", "-q")
        (repo / "requirements").mkdir()
        (repo / "requirements" / "a.lock").write_text("1")
        git("add", ".")
        git("commit", "-qm", "inputs")
        (repo / "README.md").write_text("docs only")
        git("add", ".")
        git("commit", "-qm", "docs")
        return repo, git

    def test_image_commit_is_the_last_one_that_changed_its_inputs(self) -> None:
        repo, git = self._repo()

        self.assertEqual(job.eval_image_commit(repo), git("rev-parse", "HEAD~1"))

    def test_image_commit_is_refused_in_a_shallow_clone(self) -> None:
        repo, git = self._repo()
        shallow = repo.parent / "shallow"
        git("clone", "-q", "--depth", "1", f"file://{repo}", str(shallow))

        with self.assertRaisesRegex(RuntimeError, "shallow clone"):
            job.eval_image_commit(shallow)

    def test_published_is_false_for_a_missing_image_and_none_when_ghcr_is_unreachable(self) -> None:
        token = io.BytesIO(b'{"token": "t"}')
        missing = urllib.error.HTTPError("u", 404, "not found", {}, None)
        with patch.object(job.urllib.request, "urlopen", side_effect=[token, missing]):
            self.assertIs(job.image_published("abc123"), False)
        with patch.object(job.urllib.request, "urlopen", side_effect=urllib.error.URLError("down")):
            self.assertIsNone(job.image_published("abc123"))

    def test_check_refuses_a_missing_image_and_only_warns_when_ghcr_cannot_be_asked(self) -> None:
        with patch.object(job, "image_published", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "abc123 does not exist"):
                job.check_eval_image("abc123")
        with patch.object(job, "image_published", return_value=None):
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                job.check_eval_image("abc123")
        self.assertIn("could not check", stderr.getvalue())


@unittest.skipUnless(shutil.which("bash"), "bash is needed")
class InlineSlurmTests(unittest.TestCase):
    def _run(self, tool: str, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [str(job.INLINE_SLURM_DIR / tool), *args], capture_output=True, text=True, check=False
        )

    def _script(self, body: str) -> str:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "batch.sh"
        path.write_text(body)
        return str(path)

    def test_sbatch_runs_the_script_now_and_prints_job_id_zero(self) -> None:
        script = self._script('echo "ran with $FOO and $1"\n')

        result = self._run("sbatch", "-p", "highprio", "--time=01:00:00", "--export=FOO=bar,ALL", script, "arg")

        self.assertEqual(result.stdout, "0\n")  # a caller reads the id from stdout
        self.assertEqual(result.stderr, "ran with bar and arg\n")
        self.assertEqual(result.returncode, 0)

    def test_sbatch_passes_the_scripts_failure_on(self) -> None:
        self.assertEqual(self._run("sbatch", self._script("exit 3\n")).returncode, 3)

    def test_srun_drops_slurm_flags_and_runs_the_command_here(self) -> None:
        result = self._run("srun", "--environment", "/edf.toml", "-N", "1", "--overlap", "echo", "hi")

        self.assertEqual((result.stdout, result.returncode), ("hi\n", 0))


class MainTests(unittest.TestCase):
    def _argv(self, *extra: str) -> list[str]:
        return ["--model", "/store/m", "--environment", _env_file(self), "--account", "infra01", *extra]

    def test_prints_the_script_without_submitting(self) -> None:
        argv = self._argv("--eval-image-commit", "abc123", "--eval", "echo hi", "--mount", "/extra")
        with (
            contextlib.redirect_stdout(io.StringIO()) as stdout,
            patch.object(job.subprocess, "run") as run,
            patch.object(job, "image_published", return_value=True),
        ):
            self.assertEqual(job.main(argv), 0)

        run.assert_not_called()
        self.assertIn("$PATH && echo hi'", stdout.getvalue())
        self.assertIn('    "/extra"\n', stdout.getvalue())

    def test_missing_image_stops_before_a_script_is_made(self) -> None:
        with (
            patch.object(job, "eval_image_commit", return_value="abc123"),
            patch.object(job, "image_published", return_value=False),
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):
            with self.assertRaisesRegex(SystemExit, "abc123 does not exist"):
                job.main(self._argv("--eval", "x"))

        self.assertEqual(stdout.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
