import importlib.util
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
    from scripts.harbor import k8s_environment as k8s_env


class RunArgsTest(unittest.TestCase):
    def test_model_or_gold_required(self):
        with self.assertRaises(SystemExit):
            run.parse_args(["--benchmark", "swe_bench", "--work-dir", "w", "--sandbox", "docker"])
        args = run.parse_args(["--benchmark", "swe_bench,terminal_bench_4", "--gold", "--work-dir", "w", "--sandbox", "k8s"])
        self.assertEqual(args.benchmarks, ["swe_bench", "terminal_bench_4"])

    def test_unknown_benchmark(self):
        with self.assertRaises(SystemExit):
            run.parse_args(["--benchmark", "swe_bench_v2", "--gold", "--work-dir", "w", "--sandbox", "docker"])

    def test_docker_only_benchmarks_refuse_k8s(self):
        for name in ("senior_swe_bench",):
            with self.assertRaises(SystemExit):
                run.parse_args(["--benchmark", name, "--gold", "--work-dir", "w", "--sandbox", "k8s"])
            run.parse_args(["--benchmark", name, "--gold", "--work-dir", "w", "--sandbox", "docker"])

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
        with mock.patch.object(run.subprocess, "run") as sub:
            cmd = run.harbor_command(args, "multi_swe_bench", Path("w/jobs"))
        generate = sub.call_args[0][0]
        self.assertEqual(generate[:2], ["uvx", "--from"])
        self.assertIn(f"@v{run.HARBOR_VERSION}#subdirectory=adapters/multi-swe-bench", generate[2])
        self.assertEqual(generate[-2:], ["--limit", "3"])
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


@unittest.skipUnless(HAS_HARBOR, "harbor not installed (requirements/harbor-runtime.txt)")
class DockerfileReplayTest(unittest.TestCase):
    def test_substitute(self):
        env = {"HOME": "/root", "EMPTY": ""}
        self.assertEqual(k8s_env.substitute("$HOME/x:${HOME}/y", env), "/root/x:/root/y")
        self.assertEqual(k8s_env.substitute("${EMPTY:-d} ${HOME:+set} ${NOPE:+set}", env), "d set ")
        self.assertEqual(k8s_env.substitute("\\$HOME", env), "$HOME")

    def test_parse_assignments(self):
        self.assertEqual(
            k8s_env.parse_assignments('A=1 B="two words" C=$A/x', {}), {"A": "1", "B": "two words", "C": "1/x"}
        )
        self.assertEqual(k8s_env.parse_assignments("PATH /opt/bin:$PATH", {"PATH": "/bin"}), {"PATH": "/opt/bin:/bin"})

    def test_dockerfile_stages(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d, "Dockerfile")
            path.write_text("ARG V=1\nFROM debian AS fetch\nRUN true\nFROM busybox:$V\nCOPY --from=fetch /x /x\n")
            stages = k8s_env.dockerfile_stages(path)
            self.assertEqual([s["name"] for s in stages], ["fetch", None])
            # Each stage sees the ARGs declared before the first FROM.
            self.assertEqual([s["instruction"] for s in stages[1]["steps"]], ["ARG", "FROM", "COPY"])
            path.write_text("FROM debian AS base\nRUN true\nFROM base\nRUN true\n")
            with self.assertRaisesRegex(RuntimeError, "earlier stage"):
                k8s_env.dockerfile_stages(path)
            path.write_text("# comment\nFROM swebench/x:latest\nWORKDIR /testbed\nRUN mkdir -p /logs\n")
            (stage,) = k8s_env.dockerfile_stages(path)
            self.assertEqual([s["instruction"] for s in stage["steps"]], ["FROM", "WORKDIR", "RUN"])

    def test_copies_into(self):
        self.assertFalse(k8s_env.copies_into("/opt/python", 1))
        for dest, n in (("/opt/", 1), ("/opt", 2), (".", 1), ("/", 1)):
            self.assertTrue(k8s_env.copies_into(dest, n), dest)

    def test_wrappers_run(self):
        # in_dir/as_user produce plain sh commands: run one here.
        with tempfile.TemporaryDirectory() as d:
            argv = k8s_env.in_dir(k8s_env.as_user(["sh", "-c", 'printf "%s %s" "$X" "$(pwd)"'], None), d, {"X": "a b"})
            out = subprocess.run(argv, capture_output=True, text=True, check=True).stdout
        self.assertIn(out, {f"a b {d}", f"a b {os.path.realpath(d)}"})  # pwd may resolve symlinks (macOS /var)
        self.assertIn("runuser -u", k8s_env.as_user(["true"], "agent")[2])

    def test_network_policy(self):
        from harbor.models.task.config import NetworkMode, NetworkPolicy

        deny = k8s_env.network_policy_manifest("p", "pod-1", NetworkPolicy(network_mode=NetworkMode.NO_NETWORK))
        self.assertEqual(deny["spec"]["podSelector"], {"matchLabels": {"sandbox-pod": "pod-1"}})
        self.assertEqual(deny["spec"]["egress"], [])
        allow = k8s_env.network_policy_manifest(
            "p", "pod-1", NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=["10.0.0.0/8"])
        )
        self.assertEqual(allow["spec"]["egress"][1], {"to": [{"ipBlock": {"cidr": "10.0.0.0/8"}}]})


if __name__ == "__main__":
    unittest.main()
