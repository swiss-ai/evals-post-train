import asyncio
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.harbor import run, task_patches

HAS_HARBOR = importlib.util.find_spec("harbor") is not None and importlib.util.find_spec("dockerfile_parse") is not None
if HAS_HARBOR:
    from scripts.common import dockerfile_replay as replay
    from scripts.harbor import k8s_environment as k8s_env
HAS_OPENSANDBOX = HAS_HARBOR and importlib.util.find_spec("opensandbox") is not None
if HAS_OPENSANDBOX:
    from scripts.harbor import opensandbox_environment as osb_env


class RunArgsTest(unittest.TestCase):
    def test_model_or_gold_required(self):
        with self.assertRaises(SystemExit):
            run.parse_args(["--benchmark", "swe_bench", "--work-dir", "w", "--sandbox", "docker"])
        args = run.parse_args(["--benchmark", "swe_bench,terminal_bench_4", "--gold", "--work-dir", "w", "--sandbox", "k8s"])
        self.assertEqual(args.benchmarks, ["swe_bench", "terminal_bench_4"])

    def test_unknown_benchmark(self):
        with self.assertRaises(SystemExit):
            run.parse_args(["--benchmark", "swe_bench_v2", "--gold", "--work-dir", "w", "--sandbox", "docker"])

    def test_wildcard_allowlist_benchmarks_refuse_k8s(self):
        for name in ("senior_swe_bench",):
            with self.assertRaises(SystemExit):
                run.parse_args(["--benchmark", name, "--gold", "--work-dir", "w", "--sandbox", "k8s"])
            run.parse_args(["--benchmark", name, "--gold", "--work-dir", "w", "--sandbox", "docker"])
            with mock.patch.dict(os.environ, {"OPENSANDBOX_API_KEY": "k"}):
                run.parse_args(["--benchmark", name, "--gold", "--work-dir", "w", "--sandbox", "opensandbox"])

    def test_senior_swe_bench_model_runs_need_a_grader_key(self):
        model = ["--benchmark", "senior_swe_bench", "--model", "m", "--api-base-url", "u", "--work-dir", "w", "--sandbox", "docker"]
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                run.parse_args(model)
            # --gold checks environments and tests, without the judges.
            run.parse_args(["--benchmark", "senior_swe_bench", "--gold", "--work-dir", "w", "--sandbox", "docker"])
        for key in ("ANTHROPIC_API_KEY", "PORTKEY_API_KEY"):
            with mock.patch.dict(os.environ, {key: "k"}, clear=True):
                run.parse_args(model)

    def test_every_dataset_is_pinned(self):
        for name, spec in run.BENCHMARKS.items():
            if "dataset" in spec:
                self.assertRegex(spec["dataset"], r"@(\d|sha256:[0-9a-f]{64}$)", name)


class HarborCommandTest(unittest.TestCase):
    def test_model_run_on_k8s(self):
        args = run.parse_args(
            ["--benchmark", "terminal_bench_2", "--model", "org/m", "--api-base-url", "https://x/v1", "--work-dir", "w",
             "--sandbox", "k8s", "--limit", "2", "--task", "fix-*", "--agent-kwarg", "temperature=0"]
        )
        cmd = run.harbor_command(args, "terminal_bench_2", Path("w/jobs"))
        self.assertEqual(cmd[1:3], ["run", "-o"])
        for flags in (["-d", "terminal-bench@2.0"], ["-i", "fix-*"], ["-l", "2"], ["-a", "terminus-2", "-m", "openai/org/m"],
                      ["--ak", "temperature=0"], ["--env", run.K8S_ENVIRONMENT]):
            self.assertTrue(any(cmd[i : i + len(flags)] == flags for i in range(len(cmd))), flags)

    def test_gold_on_docker(self):
        args = run.parse_args(["--benchmark", "swe_bench", "--gold", "--work-dir", "w", "--sandbox", "docker"])
        cmd = run.harbor_command(args, "swe_bench", Path("w/jobs"))
        self.assertIn("oracle", cmd)
        self.assertEqual(cmd[cmd.index("--env") + 1], run.DOCKER_ENVIRONMENT)
        self.assertNotIn("-m", cmd)

    def test_adapter_tasks_are_generated(self):
        args = run.parse_args(["--benchmark", "multi_swe_bench", "--gold", "--work-dir", "w", "--sandbox", "docker", "--limit", "3"])
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(run.subprocess, "run") as sub, \
                mock.patch.object(run, "download_dataset", return_value=Path("/hf/msb")) as download:
            cmd = run.harbor_command(args, "multi_swe_bench", Path("w/jobs"))
        download.assert_called_once_with("ByteDance-Seed/Multi-SWE-bench", run.BENCHMARKS["multi_swe_bench"]["hf_dataset"][1])
        generate = sub.call_args[0][0]
        self.assertEqual(generate[:2], ["uvx", "--from"])
        self.assertIn(f"@v{run.HARBOR_VERSION}#subdirectory=adapters/multi-swe-bench", generate[2])
        self.assertEqual(generate[-4:], ["--dataset-path", "/hf/msb", "--limit", "3"])
        self.assertEqual(cmd[cmd.index("-p") + 1], str(Path("w/tasks/multi-swe-bench")))

    def test_patched_dataset_is_downloaded_and_run_locally(self):
        def download(cmd, check):  # harbor download --export: <out>/<dataset name>/<task>/
            Path(cmd[cmd.index("-o") + 1], "aider-polyglot").mkdir(parents=True)

        patch = mock.Mock()
        with tempfile.TemporaryDirectory() as w:
            args = run.parse_args(["--benchmark", "aider_polyglot", "--gold", "--work-dir", w, "--sandbox", "docker",
                                   "--task", "polyglot_rust_*"])
            with mock.patch.object(run.subprocess, "run", side_effect=download) as sub, \
                    mock.patch.dict(run.BENCHMARKS["aider_polyglot"], patch_tasks=patch):
                cmd = run.harbor_command(args, "aider_polyglot", Path(w, "jobs"))
        tasks = Path(w, "tasks", "aider_polyglot", "aider-polyglot")
        self.assertEqual(sub.call_args[0][0][1:3], ["download", "aider-polyglot@1.0"])
        self.assertIn("--overwrite", sub.call_args[0][0])
        patch.assert_called_once_with(tasks)
        self.assertEqual(cmd[cmd.index("-p") + 1], str(tasks))
        self.assertEqual(cmd[cmd.index("-i") + 1], "polyglot_rust_*")
        self.assertNotIn("-d", cmd)

    def test_installed_adapter_is_used(self):
        with tempfile.TemporaryDirectory() as root:
            binary = Path(root, "multi-swe-bench", "bin", "multi-swe-bench")
            binary.parent.mkdir(parents=True)
            binary.touch()
            with mock.patch.dict(os.environ, {"HARBOR_ADAPTERS_ROOT": root, "EVAL_PREBUILT_ENV": "1"}, clear=True):
                self.assertEqual(run.adapter_command("multi-swe-bench"), [str(binary)])
                # A prebuilt image without it refuses to fetch it.
                with self.assertRaises(SystemExit):
                    run.adapter_command("other-adapter")

    def test_model_env(self):
        args = run.parse_args(["--benchmark", "swe_bench", "--model", "m", "--api-base-url", "https://x/v1", "--work-dir", "w",
                               "--sandbox", "docker"])
        with mock.patch.dict(os.environ, {"EVAL_MODEL_API_KEY": "k"}, clear=True), mock.patch.object(run.platform, "machine", return_value="arm64"):
            env = run.harbor_env(args)
        self.assertEqual((env["OPENAI_BASE_URL"], env["OPENAI_API_KEY"]), ("https://x/v1", "k"))
        self.assertEqual(env["DOCKER_DEFAULT_PLATFORM"], "linux/amd64")
        self.assertTrue(env["PYTHONPATH"].startswith(str(run.REPO_ROOT)))


class SummaryTest(unittest.TestCase):
    def test_unscored_trials_are_errors(self):
        with tempfile.TemporaryDirectory() as job:
            trials = {
                "a__1": {"task_name": "a", "trial_name": "a__1", "verifier_result": {"rewards": {"reward": 1.0}}},
                # An agent timeout still gets a verifier reward: a normal, scored trial.
                "b__1": {"task_name": "b", "trial_name": "b__1", "verifier_result": {"rewards": {"reward": 0.0}},
                         "exception_info": {"exception_type": "AgentTimeoutError"}},
                "c__1": {"task_name": "c", "trial_name": "c__1",
                         "exception_info": {"exception_type": "RuntimeError", "exception_message": "pod OOMKilled"}},
            }
            for name, trial in trials.items():
                Path(job, name).mkdir()
                Path(job, name, "result.json").write_text(json.dumps(trial))
            summary = run.summarize_job(Path(job))
        self.assertEqual((summary["mean_reward"], summary["n_trials"], summary["n_scored"]), (0.5, 3, 2))
        self.assertEqual(summary["evaluation_errors"], [{"task": "c", "trial": "c__1", "error": "RuntimeError: pod OOMKilled"}])

    def test_named_metrics_without_reward(self):
        with tempfile.TemporaryDirectory() as job:
            for i, rate in enumerate((1.0, 0.5)):
                Path(job, f"t{i}").mkdir()
                Path(job, f"t{i}", "result.json").write_text(
                    json.dumps({"verifier_result": {"rewards": {"core_pass_rate": rate, "erosion": 0.25}}})
                )
            summary = run.summarize_job(Path(job))
        self.assertIsNone(summary["mean_reward"])
        self.assertEqual(summary["metrics"], {"core_pass_rate": 0.75, "erosion": 0.25})

    def test_missing_reward_file(self):
        """An evaluation error by default; a 0 for a benchmark that says a missing reward is one."""
        with tempfile.TemporaryDirectory() as job:
            trials = {
                "a__1": {"task_name": "a", "trial_name": "a__1", "verifier_result": {"rewards": {"reward": 1.0}}},
                "b__1": {"task_name": "b", "trial_name": "b__1",
                         "exception_info": {"exception_type": "RewardFileNotFoundError", "exception_message": "x"}},
                "c__1": {"task_name": "c", "trial_name": "c__1",
                         "exception_info": {"exception_type": "RuntimeError", "exception_message": "pod OOMKilled"}},
            }
            for name, trial in trials.items():
                Path(job, name).mkdir()
                Path(job, name, "result.json").write_text(json.dumps(trial))
            default = run.summarize_job(Path(job))
            zero = run.summarize_job(Path(job), missing_reward_is_zero=True)
        self.assertEqual((default["mean_reward"], default["n_scored"], len(default["evaluation_errors"])), (1.0, 1, 2))
        self.assertEqual((zero["mean_reward"], zero["n_scored"], len(zero["evaluation_errors"])), (0.5, 2, 1))
        self.assertEqual(zero["no_reward_file"], [{"task": "b", "trial": "b__1"}])
        self.assertTrue(run.BENCHMARKS["aider_polyglot"]["missing_reward_is_zero"])

    def test_pass_at_k_from_job_stats(self):
        with tempfile.TemporaryDirectory() as job:
            Path(job, "t0").mkdir()
            Path(job, "t0", "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": 1.0}}}))
            Path(job, "result.json").write_text(
                json.dumps({"stats": {"evals": {"terminus-2__tb": {"pass_at_k": {"2": 0.5, "3": 0.75}}}}})
            )
            summary = run.summarize_job(Path(job))
        self.assertEqual(summary["pass_at_k"], {"2": 0.5, "3": 0.75})


class TaskPatchesTest(unittest.TestCase):
    CPP_BUILD = 'if cmake .. && make; then\n    run\nelse\n        echo "CMake build failed"\n        exit 1\nfi\n'

    def make_task(self, root, name, test_sh, files=None):
        tests = Path(root, name, "tests")
        tests.mkdir(parents=True)
        (tests / "test.sh").write_text(test_sh)
        for rel, text in (files or {}).items():
            Path(tests, rel).parent.mkdir(parents=True, exist_ok=True)
            Path(tests, rel).write_text(text)
        return tests

    def test_aider_polyglot_runs_every_test(self):
        with tempfile.TemporaryDirectory() as d:
            rust = self.make_task(d, "polyglot_rust_bowling", "    cargo test --verbose\nelse\n")
            cpp = self.make_task(d, "polyglot_cpp_queen-attack", self.CPP_BUILD)
            js = self.make_task(d, "polyglot_javascript_grep", "npm test\n", {
                "grep.spec.js": "test('a', f);\nxtest('b', f);\nxit('c', f);\ntest.skip('slow', f);\n"})
            java = self.make_task(d, "polyglot_java_forth", "gradle test\n", {"src/test/java/T.java": (
                'import org.junit.jupiter.api.Disabled;\n@Disabled("Remove to run test")\n@Disabled("Remove to run")\n'
                "@Disabled\n@Test\n")})
            go = self.make_task(d, "polyglot_go_bowling", "go test\n")
            macros = self.make_task(d, "polyglot_rust_macros", "    cargo test --verbose\nelse\n")
            task_patches.patch_aider_polyglot(Path(d))
            self.assertIn("cargo test --verbose -- --include-ignored\n", (rust / "test.sh").read_text())
            self.assertEqual((cpp / "test.sh").read_text(), self.CPP_BUILD.replace(
                "cmake ..", "cmake -DEXERCISM_RUN_ALL_TESTS=1 ..").replace("exit 1", "false"))
            self.assertEqual((js / "grep.spec.js").read_text(), "test('a', f);\ntest('b', f);\nit('c', f);\ntest.skip('slow', f);\n")
            self.assertEqual((java / "src/test/java/T.java").read_text(), "import org.junit.jupiter.api.Disabled;\n\n\n\n@Test\n")
            self.assertEqual((go / "test.sh").read_text(), "go test\n")
            self.assertIn("--include-ignored --skip compile_fails_\n", (macros / "test.sh").read_text())

    def test_changed_dataset_fails_loudly(self):
        with tempfile.TemporaryDirectory() as d:
            self.make_task(d, "polyglot_rust_bowling", "cargo test --release\n")
            with self.assertRaisesRegex(ValueError, "found it 0 times"):
                task_patches.patch_aider_polyglot(Path(d))


class MainTest(unittest.TestCase):
    def test_stopped_run_keeps_finished_scores(self):
        """SIGTERM mid-run: summary.json has the benchmarks so far, the current one interrupted."""
        import signal
        import threading

        def fake_command(args, name, jobs_dir):
            trial = jobs_dir / name / "t1"
            trial.mkdir(parents=True, exist_ok=True)
            (trial / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": 1.0}}}))
            return ["sleep", "0" if name == "swe_bench" else "30"]

        with tempfile.TemporaryDirectory() as d, mock.patch.object(run, "harbor_command", fake_command):
            timer = threading.Timer(1.0, os.kill, (os.getpid(), signal.SIGTERM))
            timer.start()
            try:
                code = run.main(["--benchmark", "swe_bench,terminal_bench_2,deep_swe", "--gold", "--work-dir", d,
                                 "--sandbox", "docker"])
            finally:
                timer.cancel()
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
            summary = json.loads((Path(d) / "summary.json").read_text())["benchmarks"]
        self.assertEqual(code, 128 + signal.SIGTERM)
        self.assertEqual(list(summary), ["swe_bench", "terminal_bench_2"])
        self.assertNotIn("interrupted", summary["swe_bench"])
        self.assertTrue(summary["terminal_bench_2"]["interrupted"])
        self.assertEqual(summary["terminal_bench_2"]["mean_reward"], 1.0)


class FakeEnvironment:
    """The parts of a Harbor environment the Dockerfile replay uses; records its commands."""

    instances: list = []

    def __init__(self, environment_dir, name="task", session_id="s", trial_paths=None, config=None, logger=None,
                 docker_image=None, fail=None):
        from harbor.models.task.config import EnvironmentConfig

        self.environment_dir = environment_dir
        self.environment_name = name
        self.session_id = session_id
        self.trial_paths = trial_paths
        self.logger = logger
        self.task_env_config = config or EnvironmentConfig(docker_image=docker_image)
        self.fail = fail
        self.calls, self.uploads, self.events = [], {}, []
        FakeEnvironment.instances.append(self)

    async def start(self, force_build):
        self.events.append("start")

    async def stop(self, delete):
        self.events.append("stop")

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        from harbor.environments.base import ExecResult

        self.calls.append({"command": command, "cwd": cwd, "user": user})
        if command == replay._READ_ENV:
            return ExecResult(stdout="PATH=/usr/bin\0HOME=/root\0", stderr="", return_code=0)
        if self.fail and self.fail in command:
            return ExecResult(stdout="", stderr="boom", return_code=1)
        return ExecResult(stdout="", stderr="", return_code=0)

    async def upload_file(self, source_path, target_path):
        self.uploads[target_path] = Path(source_path).read_bytes()

    async def download_file(self, source_path, target_path):
        Path(target_path).write_bytes(b"packed:" + source_path.encode())


@unittest.skipUnless(HAS_HARBOR, "harbor not installed (requirements/harbor-runtime.lock)")
class DockerfileReplayTest(unittest.TestCase):
    def setUp(self):
        FakeEnvironment.instances = []

    def test_substitute(self):
        env = {"HOME": "/root", "EMPTY": ""}
        self.assertEqual(replay.substitute("$HOME/x:${HOME}/y", env), "/root/x:/root/y")
        self.assertEqual(replay.substitute("${EMPTY:-d} ${HOME:+set} ${NOPE:+set}", env), "d set ")
        self.assertEqual(replay.substitute("\\$HOME", env), "$HOME")

    def test_parse_assignments(self):
        self.assertEqual(
            replay.parse_assignments('A=1 B="two words" C=$A/x', {}), {"A": "1", "B": "two words", "C": "1/x"}
        )
        self.assertEqual(replay.parse_assignments("PATH /opt/bin:$PATH", {"PATH": "/bin"}), {"PATH": "/opt/bin:/bin"})

    def test_dockerfile_stages(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d, "Dockerfile")
            path.write_text("ARG V=1\nFROM debian AS fetch\nRUN true\nFROM busybox:$V\nCOPY --from=fetch /x /x\n")
            stages = replay.dockerfile_stages(path)
            self.assertEqual([s["name"] for s in stages], ["fetch", None])
            # Each stage sees the ARGs declared before the first FROM.
            self.assertEqual([s["instruction"] for s in stages[1]["steps"]], ["ARG", "FROM", "COPY"])
            path.write_text("FROM debian AS base\nRUN true\nFROM base\nRUN true\n")
            with self.assertRaisesRegex(RuntimeError, "earlier stage"):
                replay.dockerfile_stages(path)
            path.write_text("# comment\nFROM swebench/x:latest\nWORKDIR /testbed\nRUN mkdir -p /logs\n")
            (stage,) = replay.dockerfile_stages(path)
            self.assertEqual([s["instruction"] for s in stage["steps"]], ["FROM", "WORKDIR", "RUN"])

    def test_copies_into(self):
        self.assertFalse(replay.copies_into("/opt/python", 1))
        for dest, n in (("/opt/", 1), ("/opt", 2), (".", 1), ("/", 1)):
            self.assertTrue(replay.copies_into(dest, n), dest)

    def test_wrappers_run(self):
        # in_dir/as_user produce plain sh commands: run one here.
        with tempfile.TemporaryDirectory() as d:
            argv = replay.in_dir(replay.as_user(["sh", "-c", 'printf "%s %s" "$X" "$(pwd)"'], None), d, {"X": "a b"})
            out = subprocess.run(argv, capture_output=True, text=True, check=True).stdout
        self.assertIn(out, {f"a b {d}", f"a b {os.path.realpath(d)}"})  # pwd may resolve symlinks (macOS /var)
        self.assertIn("runuser -u", replay.as_user(["true"], "agent")[2])

    def test_pack_tree(self):
        import tarfile

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "usr/local/app").mkdir(parents=True)
            (root / "usr/local/app/run.sh").write_text("x")
            (root / "usr/local/app/sub").mkdir()
            os.symlink("run.sh", root / "usr/local/app/link")
            data = replay.pack_tree(root, "/usr/local/app")
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            members = {m.name: m for m in archive.getmembers()}
        # Existing directories (/usr, /usr/local, the destination) keep their owner and mode.
        self.assertEqual(sorted(members), ["usr/local/app/link", "usr/local/app/run.sh", "usr/local/app/sub"])
        self.assertTrue(members["usr/local/app/link"].issym())
        self.assertEqual({(m.uid, m.gid, m.uname) for m in members.values()}, {(0, 0, "root")})

    def write_task(self, root, dockerfile, **files):
        environment = root / "environment"
        environment.mkdir()
        (environment / "Dockerfile").write_text(dockerfile)
        for name, content in files.items():
            (environment / name).write_text(content)
        return environment

    def test_replay(self):
        import tarfile

        with tempfile.TemporaryDirectory() as d:
            environment = self.write_task(
                Path(d),
                "ARG BASE=debian\nFROM $BASE:12\nARG V=3\nENV TOOL=/opt/tool PATH=/opt/tool/bin:$PATH\n"
                "WORKDIR /testbed\nCOPY setup.sh ./\nRUN ./setup.sh $V\nUSER agent\nRUN whoami\nCMD bash\n",
                **{"setup.sh": "echo hi"},
            )
            env = FakeEnvironment(environment)
            plan = replay.DockerfileReplay(env, force_build=False)
            self.assertEqual(plan.image, "debian:12")
            self.assertEqual(asyncio.run(plan.export_stages()), {})
            image = asyncio.run(plan.replay({}))
        self.assertEqual(image.env, {"TOOL": "/opt/tool", "PATH": "/opt/tool/bin:/usr/bin"})
        self.assertEqual((image.workdir, image.user), ("/testbed", "agent"))
        self.assertEqual({call["user"] for call in env.calls}, {"root"})
        commands = [call["command"] for call in env.calls]
        self.assertIn("mkdir -p /testbed", commands[1])
        self.assertIn("tar xzf /.sandbox-replay-", commands[2])
        setup, whoami = commands[3], commands[4]
        self.assertIn("cd /testbed", setup)
        self.assertIn("V=3", setup)
        self.assertNotIn("BASE=", setup)  # an ARG before FROM is only seen by FROM
        self.assertIn("./setup.sh $V", setup)
        self.assertIn("runuser -u", whoami)
        ((remote, data),) = env.uploads.items()
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            self.assertEqual(archive.getnames(), ["testbed/setup.sh"])

    def test_replay_failure_names_the_step(self):
        with tempfile.TemporaryDirectory() as d:
            env = FakeEnvironment(self.write_task(Path(d), "FROM debian\nRUN false\n"), fail="false")
            plan = replay.DockerfileReplay(env, force_build=False)
            with self.assertRaisesRegex(RuntimeError, "RUN false failed .exit 1.: boom"):
                asyncio.run(plan.replay({}))

    def test_multi_stage_replay(self):
        with tempfile.TemporaryDirectory() as d:
            environment = self.write_task(
                Path(d),
                "FROM debian AS fetch\nWORKDIR /src\nRUN fetch python\n"
                "FROM swebench/x:1\nCOPY --from=fetch python /opt/python\nCOPY --from=busybox:1 /bin/busybox /bin/\n",
            )
            env = FakeEnvironment(environment)
            plan = replay.DockerfileReplay(env, force_build=False)
            self.assertEqual(plan.image, "swebench/x:1")
            exports = asyncio.run(plan.export_stages())
            self.assertEqual(sorted(exports), [1, 2])
            stages = [e for e in FakeEnvironment.instances if e is not env]
            # One sandbox per source, each from its image, started and deleted.
            self.assertEqual([s.task_env_config.docker_image for s in stages], ["debian", "busybox:1"])
            self.assertTrue(all(s.events == ["start", "stop"] for s in stages))
            fetch = [call["command"] for call in stages[0].calls]
            self.assertTrue(any("cd /src" in c and "fetch python" in c for c in fetch))
            # A single source to a path lands under the destination's name.
            self.assertTrue(any("cp -a /src/python" in c and "/opt/python" not in c and "/python" in c for c in fetch))
            asyncio.run(plan.replay(exports))
        unpack = [call["command"] for call in env.calls if "tar xzf" in call["command"]]
        self.assertTrue(any("-C /opt" in c for c in unpack), unpack)
        self.assertTrue(any("-C /bin" in c for c in unpack), unpack)

    def test_prebuilt_image_has_nothing_to_replay(self):
        with tempfile.TemporaryDirectory() as d:
            environment = self.write_task(Path(d), "FROM debian\nRUN true\n")
            env = FakeEnvironment(environment, docker_image="repo/image:1")
            self.assertEqual(replay.DockerfileReplay(env, force_build=False).stages, [])
            self.assertEqual(replay.DockerfileReplay(env, force_build=False).image, "repo/image:1")
            self.assertEqual(len(replay.DockerfileReplay(env, force_build=True).stages), 1)

    def test_network_policy(self):
        from harbor.models.task.config import NetworkMode, NetworkPolicy

        deny = k8s_env.network_policy_manifest("p", "pod-1", NetworkPolicy(network_mode=NetworkMode.NO_NETWORK), "r1")
        self.assertEqual(deny["spec"]["podSelector"], {"matchLabels": {"sandbox-pod": "pod-1"}})
        self.assertEqual(deny["metadata"]["labels"]["sandbox-run"], "r1")
        self.assertEqual(deny["spec"]["egress"], [])
        allow = k8s_env.network_policy_manifest(
            "p", "pod-1", NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=["10.0.0.0/8"]), "r1"
        )
        self.assertEqual(allow["spec"]["egress"][1], {"to": [{"ipBlock": {"cidr": "10.0.0.0/8"}}]})


class OpenSandboxRunTest(unittest.TestCase):
    ARGS = ["--benchmark", "terminal_bench_2", "--gold", "--work-dir", "w", "--sandbox", "opensandbox"]

    def test_needs_an_api_key(self):
        with mock.patch.dict(os.environ, {"OPENSANDBOX_API_KEY": ""}), self.assertRaises(SystemExit):
            run.parse_args(self.ARGS)

    def test_command_and_env(self):
        with mock.patch.dict(os.environ, {"OPENSANDBOX_API_KEY": "k"}):
            os.environ.pop("OPENSANDBOX_DOMAIN", None)
            args = run.parse_args(self.ARGS)
            first, second = run.harbor_env(args), run.harbor_env(args)
        self.assertIn(run.OPENSANDBOX_ENVIRONMENT, run.harbor_command(args, "terminal_bench_2", Path("w/jobs")))
        self.assertEqual(first["OPENSANDBOX_DOMAIN"], run.OPENSANDBOX_KATA_DOMAIN)
        self.assertNotEqual(first["OPENSANDBOX_RUN_LABEL"], second["OPENSANDBOX_RUN_LABEL"])

    def test_summary_records_the_endpoint_and_leftovers_are_deleted(self):
        def fake_command(args, name, jobs_dir):
            trial = jobs_dir / name / "t1"
            trial.mkdir(parents=True)
            (trial / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": 1.0}}}))
            return ["true"]

        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch.dict(os.environ, {"OPENSANDBOX_API_KEY": "k", "OPENSANDBOX_DOMAIN": "server.example"}),
            mock.patch.object(run, "harbor_command", fake_command),
            mock.patch.object(run, "delete_leftover_sandboxes") as delete,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            work_dir = ["--work-dir", d]
            code = run.main([*self.ARGS, *work_dir])
            summary = json.loads((Path(d) / "summary.json").read_text())
        self.assertEqual(code, 0)
        self.assertEqual(summary["sandbox_endpoint"], "server.example")
        delete.assert_called_once()


@unittest.skipUnless(HAS_OPENSANDBOX, "needs Harbor's venv")
class KataSandboxEnvironmentTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "environment").mkdir()
        patcher = mock.patch.dict(os.environ, {osb_env.RUN_LABEL_ENV: "run-1"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def environment(self, network_policy=None, docker_image="repo/image:1"):
        from harbor.models.task.config import EnvironmentConfig
        from harbor.models.trial.paths import TrialPaths

        return osb_env.KataSandboxEnvironment(
            self.root / "environment",
            "task",
            "session",
            TrialPaths(self.root / "trial"),
            EnvironmentConfig(docker_image=docker_image),
            network_policy=network_policy,
        )

    def built_policy(self, **policy):
        from harbor.models.task.config import NetworkPolicy

        env = self.environment(NetworkPolicy(**policy))
        return env._build_network_policy(env._load_opensandbox()).model_dump(by_alias=True, exclude_none=True)

    def test_every_mode_sends_an_explicit_policy(self):
        from harbor.models.task.config import NetworkMode

        self.assertEqual(self.built_policy(), {"defaultAction": "allow", "egress": []})
        self.assertEqual(self.built_policy(network_mode=NetworkMode.NO_NETWORK), {"defaultAction": "deny", "egress": []})
        allow = self.built_policy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=["pypi.org", "*.pythonhosted.org", "10.0.0.0/8"])
        self.assertEqual(allow["defaultAction"], "deny")
        self.assertEqual(
            [(rule["action"], rule["target"]) for rule in allow["egress"]],
            [("allow", "pypi.org"), ("allow", "*.pythonhosted.org"), ("allow", "10.0.0.0/8")],
        )

    def test_connection_and_run_label(self):
        env = self.environment()
        self.assertTrue(env._use_server_proxy)
        self.assertEqual(env._metadata, {"sandbox-run": "run-1"})

    def test_sandboxes_are_created_with_secure_access(self):
        from opensandbox import Sandbox

        sdk = self.environment()._load_opensandbox()
        with mock.patch.object(Sandbox, "create", mock.AsyncMock()) as create:
            asyncio.run(sdk["Sandbox"].create("image:1", timeout=None))
        self.assertTrue(create.await_args.kwargs["secure_access"])

    def test_dockerfile_only_task(self):
        from harbor.models.task.config import NetworkMode, NetworkPolicy

        with self.assertRaisesRegex(FileNotFoundError, "prebuilt image"):
            self.environment(docker_image=None)
        (self.root / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\nRUN true\n")
        env = self.environment(docker_image=None)
        with mock.patch.object(osb_env.OpenSandboxEnvironment, "start", mock.AsyncMock()) as start, \
                mock.patch.object(env, "exec", mock.AsyncMock(return_value=mock.Mock(return_code=0))) as exec_, \
                mock.patch.object(osb_env.DockerfileReplay, "replay", mock.AsyncMock(return_value=osb_env.ImageConfig())):
            asyncio.run(env.start(force_build=False))
        start.assert_awaited_once()
        self.assertIn("overcommit_memory", exec_.await_args.args[0])
        self.assertEqual(exec_.await_args.kwargs["user"], "root")
        self.assertEqual(env._resolve_image_spec({}), "ubuntu:24.04")
        self.assertEqual(env._build_network_policy({}).default_action, "allow")
        # The replay installs packages: a restricted task is created public and restricted after it.
        restricted = self.environment(NetworkPolicy(network_mode=NetworkMode.NO_NETWORK), docker_image=None)
        with mock.patch.object(osb_env.OpenSandboxEnvironment, "start", mock.AsyncMock()), \
                mock.patch.object(restricted, "exec", mock.AsyncMock(return_value=mock.Mock(return_code=0))), \
                mock.patch.object(restricted, "_apply_network_policy", mock.AsyncMock()) as apply, \
                mock.patch.object(osb_env.DockerfileReplay, "replay", mock.AsyncMock(return_value=osb_env.ImageConfig())):
            asyncio.run(restricted.start(force_build=False))
            self.assertEqual(restricted._build_network_policy({}).default_action, "allow")
        apply.assert_awaited_once_with(NetworkPolicy(network_mode=NetworkMode.NO_NETWORK))
        # A prebuilt image is created with the task's policy and needs no change.
        prebuilt = self.environment(NetworkPolicy(network_mode=NetworkMode.NO_NETWORK))
        with mock.patch.object(osb_env.OpenSandboxEnvironment, "start", mock.AsyncMock()), \
                mock.patch.object(prebuilt, "exec", mock.AsyncMock(return_value=mock.Mock(return_code=0))), \
                mock.patch.object(prebuilt, "_apply_network_policy", mock.AsyncMock()) as apply:
            asyncio.run(prebuilt.start(force_build=False))
        self.assertEqual(prebuilt._build_network_policy({}).default_action, "deny")
        apply.assert_not_awaited()

    def policy_switch(self, read_back):
        import httpx
        from harbor.models.task.config import NetworkMode, NetworkPolicy
        from opensandbox.config import ConnectionConfig
        from opensandbox.models.sandboxes import SandboxEndpoint

        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"status": "ok"})

        config = ConnectionConfig(domain="d", api_key="k", protocol="https", use_server_proxy=True, transport=httpx.MockTransport(handler))
        env = self.environment()
        env._sandbox = mock.Mock(
            id="s1",
            connection_config=config,
            get_endpoint=mock.AsyncMock(return_value=SandboxEndpoint(endpoint="d/sandboxes/s1/proxy/18080", headers={"X-Access": "t"})),
            get_egress_policy=mock.AsyncMock(return_value=read_back),
        )
        policy = NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=["pypi.org"])
        return env, policy, requests

    def test_network_policy_is_replaced_and_read_back(self):
        from harbor.models.task.config import NetworkMode, NetworkPolicy

        allow_pypi = osb_env.sandbox_network_policy(NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=["pypi.org"]))
        env, policy, requests = self.policy_switch(allow_pypi)
        self.assertTrue(env.capabilities.dynamic_network_policy)
        asyncio.run(env._apply_network_policy(policy))
        (request,) = requests
        self.assertEqual((request.method, str(request.url)), ("PUT", "https://d/sandboxes/s1/proxy/18080/policy"))
        self.assertEqual(json.loads(request.content), {"defaultAction": "deny", "egress": [{"action": "allow", "target": "pypi.org"}]})
        # The endpoint's access header (secure_access) and the key (server proxy) go with it.
        self.assertEqual((request.headers["X-Access"], request.headers["OPEN-SANDBOX-API-KEY"]), ("t", "k"))

    def test_network_policy_that_did_not_take_effect_fails(self):
        from harbor.models.task.config import NetworkPolicy

        env, policy, _ = self.policy_switch(osb_env.sandbox_network_policy(NetworkPolicy()))
        with self.assertRaisesRegex(RuntimeError, "egress policy is"):
            asyncio.run(env._apply_network_policy(policy))

    def test_exec_uses_the_replayed_image_config(self):
        from opensandbox.models.execd import Execution, ExecutionComplete

        env = self.environment()
        env._image = osb_env.ImageConfig(env={"PATH": "/opt/bin:/usr/bin"}, workdir="/testbed", user="agent")
        env._image_uid = 1000
        done = Execution(complete=ExecutionComplete(timestamp=1, execution_time_in_millis=1), exit_code=0)
        env._sandbox = mock.Mock(id="s1", commands=mock.Mock(run=mock.AsyncMock(return_value=done)))
        asyncio.run(env.exec("true"))
        opts = env._sandbox.commands.run.await_args.kwargs["opts"]
        self.assertEqual((opts.working_directory, opts.uid, opts.envs["PATH"]), ("/testbed", 1000, "/opt/bin:/usr/bin"))
        asyncio.run(env.exec("true", cwd="/", user="root"))
        opts = env._sandbox.commands.run.await_args.kwargs["opts"]
        self.assertEqual((opts.working_directory, opts.uid), ("/", 0))

    def run_exec(self, execution):
        env = self.environment()
        env._sandbox = mock.Mock(id="s1", commands=mock.Mock(run=mock.AsyncMock(return_value=execution)))
        return asyncio.run(env.exec("true"))

    def test_exec_result(self):
        from opensandbox.models.execd import Execution, ExecutionComplete, ExecutionError, OutputMessage

        done = Execution(complete=ExecutionComplete(timestamp=1, execution_time_in_millis=1), exit_code=3)
        done.logs.stdout.append(OutputMessage(text="out", timestamp=1))
        result = self.run_exec(done)
        self.assertEqual((result.return_code, result.stdout), (3, "out\n"))
        failed = Execution(error=ExecutionError(name="e", value="killed", timestamp=1))
        self.assertEqual(self.run_exec(failed).return_code, 1)

    def test_output_lines_are_joined(self):
        from opensandbox.models.execd import OutputMessage

        # As the server streams `printf 'a\nb\n\nc'`: a line per message, an empty line as "\n".
        messages = [OutputMessage(text=text, timestamp=1) for text in ("a", "b", "\n", "c")]
        self.assertEqual(osb_env.joined_lines(messages), "a\nb\n\nc\n")
        self.assertEqual(osb_env.joined_lines([]), "")

    def test_exec_without_a_result_is_not_a_success(self):
        from opensandbox.models.execd import Execution

        with self.assertRaisesRegex(RuntimeError, "without a result"):
            self.run_exec(Execution())

    def test_delete_run_sandboxes(self):
        def page(has_next, **states):
            items = [mock.Mock(id=name, status=mock.Mock(state=state)) for name, state in states.items()]
            return mock.Mock(sandbox_infos=items, pagination=mock.Mock(has_next_page=has_next))

        manager = mock.Mock(
            list_sandbox_infos=mock.AsyncMock(
                side_effect=[page(True, a="Running", b="Terminated"), page(False, c="Pending", d="Failed")]
            ),
            kill_sandbox=mock.AsyncMock(),
            close=mock.AsyncMock(),
        )
        with mock.patch("opensandbox.manager.SandboxManager.create", mock.AsyncMock(return_value=manager)) as create:
            self.assertEqual(asyncio.run(osb_env.delete_run_sandboxes("run-1", "d", "k")), 2)
        self.assertEqual(create.await_args.args[0].get_base_url(), "https://d/v1")
        self.assertEqual(sorted(call.args[0] for call in manager.kill_sandbox.await_args_list), ["a", "c"])
        second = manager.list_sandbox_infos.await_args.args[0]
        self.assertEqual((second.metadata, second.page), ({"sandbox-run": "run-1"}, 2))
        manager.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
