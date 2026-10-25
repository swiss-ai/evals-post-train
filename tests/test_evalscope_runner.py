import asyncio
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.evalscope_runner import k8s_sandbox, run

REPO_ROOT = Path(__file__).resolve().parents[1]
HAS_ENCLAVE = importlib.util.find_spec("ms_enclave") is not None
HAS_EVALSCOPE = importlib.util.find_spec("evalscope") is not None


class PodManifestTest(unittest.TestCase):
    CONFIG = {
        "image": "jefzda/sweap-images:nodebb.nodebb-abc",
        "working_dir": "/app",
        "env_vars": {"PAGER": "cat"},
        "memory_limit": "8g",
        "cpu_limit": 4,
        "platform": "linux/amd64",
    }

    def test_maps_docker_config(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            m = k8s_sandbox.pod_manifest("p", "ns", self.CONFIG, "run1")
        c = m["spec"]["containers"][0]
        self.assertEqual(c["image"], self.CONFIG["image"])
        self.assertEqual(c["workingDir"], "/app")
        self.assertEqual(c["env"], [{"name": "PAGER", "value": "cat"}])
        self.assertEqual(c["resources"]["limits"], {"cpu": "4", "memory": "8Gi"})
        self.assertEqual(c["resources"]["requests"], c["resources"]["limits"])
        self.assertEqual(m["spec"]["nodeSelector"], {"kubernetes.io/arch": "amd64"})
        self.assertFalse(m["spec"]["automountServiceAccountToken"])
        self.assertEqual(m["metadata"]["labels"]["evalscope-run"], "run1")
        self.assertNotIn("ownerReferences", m["metadata"])
        self.assertNotIn("tolerations", m["spec"])

    def test_env_overrides(self):
        env = {
            "EVALSCOPE_K8S_NODE_ARCH": "arm64",
            "EVALSCOPE_K8S_TOLERATIONS": "hpc=true",
            "EVALSCOPE_K8S_OWNER_POD_NAME": "runner",
            "EVALSCOPE_K8S_OWNER_POD_UID": "uid-1",
            "EVALSCOPE_K8S_IMAGE_PULL_SECRET": "dockerhub",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            m = k8s_sandbox.pod_manifest("p", "ns", self.CONFIG, "run1")
        self.assertEqual(m["spec"]["nodeSelector"], {"kubernetes.io/arch": "arm64"})
        self.assertEqual({t["effect"] for t in m["spec"]["tolerations"]}, {"NoSchedule", "NoExecute"})
        self.assertEqual(m["metadata"]["ownerReferences"][0]["uid"], "uid-1")
        self.assertEqual(m["spec"]["imagePullSecrets"], [{"name": "dockerhub"}])

    def test_memory_units(self):
        self.assertEqual(k8s_sandbox.k8s_memory("4g"), "4Gi")
        self.assertEqual(k8s_sandbox.k8s_memory("512m"), "512Mi")
        self.assertEqual(k8s_sandbox.k8s_memory("2GB"), "2Gi")
        self.assertEqual(k8s_sandbox.k8s_memory(1024), "1024")
        self.assertEqual(k8s_sandbox.k8s_memory(None), "4Gi")


class ExecHelpersTest(unittest.TestCase):
    def test_timeout_argv_matches_docker_box(self):
        self.assertEqual(
            k8s_sandbox.timeout_argv(["bash", "-lc", "ls"], 60),
            ["timeout", "-s", "TERM", "-k", "5s", "60.0s", "bash", "-lc", "ls"],
        )
        self.assertEqual(k8s_sandbox.timeout_argv("echo 'a b'", None), ["echo", "a b"])

    def test_returncode_from_status(self):
        self.assertEqual(k8s_sandbox.returncode_from_status(None), 0)
        self.assertEqual(k8s_sandbox.returncode_from_status('{"status": "Success"}'), 0)
        failure = {"status": "Failure", "details": {"causes": [{"reason": "ExitCode", "message": "3"}]}}
        self.assertEqual(k8s_sandbox.returncode_from_status(json.dumps(failure)), 3)
        self.assertEqual(k8s_sandbox.returncode_from_status("garbage"), 1)

    def test_hosts_localhost_ipv4_only(self):
        with tempfile.TemporaryDirectory() as d:
            hosts = Path(d, "hosts")
            hosts.write_text(
                "# Kubernetes-managed hosts file.\n127.0.0.1\tlocalhost\n"
                "::1\tlocalhost ip6-localhost ip6-loopback\nfe00::0\tip6-localnet\n10.42.0.5\tpod\n"
            )
            script = k8s_sandbox._HOSTS_LOCALHOST_IPV4_ONLY.replace("/etc/hosts", str(hosts))
            subprocess.run(["sh", "-c", script], check=True)
            self.assertEqual(
                hosts.read_text(),
                "# Kubernetes-managed hosts file.\n127.0.0.1\tlocalhost\n"
                "::1\tip6-localhost ip6-loopback\nfe00::0\tip6-localnet\n10.42.0.5\tpod\n",
            )

    def test_tar_round_trip(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            Path(src, "patch.diff").write_text("diff --git a/x b/x\n")
            Path(src, "sub").mkdir()
            Path(src, "sub", "f").write_text("x")
            b64 = k8s_sandbox.tar_dir_b64(src)
            # The pod side's `base64` wraps lines; untar must not care.
            wrapped = "\n".join(b64[i : i + 76] for i in range(0, len(b64), 76))
            k8s_sandbox.untar_b64(wrapped, dst)
            self.assertEqual(Path(dst, "patch.diff").read_text(), "diff --git a/x b/x\n")
            self.assertEqual(Path(dst, "sub", "f").read_text(), "x")


@unittest.skipUnless(HAS_ENCLAVE, "ms-enclave not installed (requirements/evalscope-runtime.txt)")
class ExecuteToolTest(unittest.TestCase):
    def _manager(self, exec_result, volumes=None):
        manager = k8s_sandbox.K8sPodSandboxManager()
        manager._sandboxes["s1"] = k8s_sandbox._Sandbox("pod-1", volumes or {})
        calls = []

        async def fake_exec(pod, argv, timeout=None):
            calls.append((pod, argv, timeout))
            if argv[:2] == ["sh", "-c"] and "tar czf" in argv[2]:
                return 0, self._b64, ""
            return exec_result

        manager._exec = fake_exec
        return manager, calls

    def _run(self, manager, command, timeout=30):
        return asyncio.run(manager.execute_tool("s1", "shell_executor", {"command": command, "timeout": timeout}))

    def test_success_failure_shapes(self):
        from ms_enclave.sandbox.model import ExecutionStatus

        self._b64 = ""
        manager, _ = self._manager((0, "out", ""))
        result = self._run(manager, ["bash", "-lc", "ls"])
        self.assertEqual((result.status, result.output, result.error), (ExecutionStatus.SUCCESS, "out", None))

        manager, _ = self._manager((2, "", ""))
        result = self._run(manager, ["false"])
        self.assertEqual(result.status, ExecutionStatus.ERROR)
        # EvalScope's EnclaveAgentEnvironment parses the exit code out of this message.
        self.assertEqual(result.error, "Command failed with exit code 2")

    def test_timeout_needs_elapsed_time(self):
        from ms_enclave.sandbox.model import ExecutionStatus

        self._b64 = ""
        # busybox timeout exits 143; reported as a timeout only once the limit has passed.
        manager, _ = self._manager((143, "", ""))
        self.assertEqual(self._run(manager, ["sleep", "10"], timeout=0.01).status, ExecutionStatus.TIMEOUT)
        manager, _ = self._manager((143, "", ""))
        self.assertEqual(self._run(manager, ["sh", "-c", "exit 143"], timeout=60).status, ExecutionStatus.ERROR)

    def test_unsupported_tool(self):
        from ms_enclave.sandbox.model import ExecutionStatus

        self._b64 = ""
        manager, _ = self._manager((0, "", ""))
        result = asyncio.run(manager.execute_tool("s1", "python_executor", {"code": "1"}))
        self.assertEqual(result.status, ExecutionStatus.ERROR)

    def test_rw_volume_synced_back(self):
        with tempfile.TemporaryDirectory() as pod_side, tempfile.TemporaryDirectory() as host:
            Path(pod_side, "output.json").write_text('{"tests": []}')
            self._b64 = k8s_sandbox.tar_dir_b64(pod_side)
            manager, calls = self._manager((0, "", ""), {host: {"bind": "/workspace", "mode": "rw"}})
            self._run(manager, ["bash", "/workspace/entryscript.sh"], timeout=3600)
            self.assertEqual(Path(host, "output.json").read_text(), '{"tests": []}')
            self.assertEqual(calls[0][2], 3600)


@unittest.skipUnless(HAS_EVALSCOPE, "evalscope not installed (requirements/evalscope-runtime.txt)")
class InstallTest(unittest.TestCase):
    def test_docker_engine_routed_to_pods(self):
        from evalscope.api.sandbox import SandboxEngine
        from evalscope.api.sandbox.service import SandboxService

        original = SandboxService._construct_manager
        try:
            k8s_sandbox.install()
            k8s_sandbox.install()  # idempotent
            manager = SandboxService._construct_manager(None, SandboxEngine.DOCKER, {})
            self.assertIsInstance(manager, k8s_sandbox.K8sPodSandboxManager)
        finally:
            SandboxService._construct_manager = original

    def test_swe_bench_scoring_routed_through_sandbox_service(self):
        import evalscope.benchmarks.swe_bench.utils as upstream

        from scripts.evalscope_runner import swe_bench_scoring

        original = upstream.eval_instance
        try:
            swe_bench_scoring.install({"memory_limit": "8g"})
            self.assertIs(upstream.eval_instance, swe_bench_scoring.eval_instance)
            self.assertEqual(swe_bench_scoring._SANDBOX_DEFAULTS, {"memory_limit": "8g"})
        finally:
            upstream.eval_instance = original


class TaskConfigTest(unittest.TestCase):
    def _cfg(self, benchmark, *extra):
        args = run.parse_args(
            ["--benchmark", benchmark, "--model", "m", "--api-base-url", "http://x/v1", "--work-dir", "/w", *extra]
        )
        names = run.selected_benchmarks(args.benchmark)
        return run.build_task_config(args, names, run.resolved_revisions(names, args.dataset_revision))

    def test_defaults_keep_benchmark_agent_settings(self):
        cfg = self._cfg("swe_bench,swe_bench_pro")
        self.assertNotIn("agent_config", cfg)
        self.assertEqual(cfg["datasets"], ["swe_bench_verified_agentic", "swe_bench_pro"])
        self.assertEqual(
            cfg["dataset_args"]["swe_bench_pro"],
            {"hf_dataset_revision": run.BENCHMARKS["swe_bench_pro"]["revision"], "extra_params": {}},
        )
        # Prebuilt x86_64 images, no local Docker builds.
        self.assertEqual(
            cfg["dataset_args"]["swe_bench_verified_agentic"]["extra_params"],
            {"build_docker_images": False, "force_arch": "x86_64"},
        )
        self.assertEqual(cfg["sandbox"], {"enabled": True, "engine": "docker", "default_config": {}})
        self.assertEqual((cfg["eval_type"], cfg["dataset_hub"], cfg["no_timestamp"]), ("openai_api", "huggingface", True))
        self.assertNotIn("limit", cfg)

    def test_pro_pinned_to_v1(self):
        # The last pre-V2 commit: v1 scoring against V2 rows fails even gold patches.
        self.assertEqual(run.BENCHMARKS["swe_bench_pro"]["revision"], "7ab5114912baf22bb098818e604c02fe7ad2c11f")

    def test_overrides(self):
        cfg = self._cfg(
            "swe_bench_pro",
            "--max-steps", "100", "--strategy", "swe_bench_backticks", "--limit", "2",
            "--extra-param", "eval_timeout=7200", "--sandbox-config", '{"memory_limit": "8g"}',
            "--dataset-revision", "swe_bench_pro=abc123",
        )
        self.assertEqual(cfg["agent_config"], {"mode": "native", "strategy": "swe_bench_backticks", "max_steps": 100})
        self.assertEqual(cfg["limit"], 2)
        self.assertEqual(
            cfg["dataset_args"]["swe_bench_pro"], {"hf_dataset_revision": "abc123", "extra_params": {"eval_timeout": 7200}}
        )
        self.assertEqual(cfg["sandbox"]["default_config"], {"memory_limit": "8g"})

    def test_generation_config_keeps_temperature_default(self):
        self.assertEqual(self._cfg("swe_bench")["generation_config"], {"temperature": 0.0})
        cfg = self._cfg("swe_bench", "--generation-config", '{"retries": 100}')
        self.assertEqual(cfg["generation_config"], {"temperature": 0.0, "retries": 100})
        cfg = self._cfg("swe_bench", "--generation-config", '{"temperature": 0.6}')
        self.assertEqual(cfg["generation_config"], {"temperature": 0.6})

    def test_only_the_two_benchmarks(self):
        for bad in ("gsm8k", "swe_bench_verified_agentic", ""):
            with self.assertRaises(SystemExit):
                run.selected_benchmarks(bad)
        with self.assertRaises(SystemExit):
            run.resolved_revisions(["swe_bench"], ["swe_bench_pro=abc"])

    def test_gold_needs_no_model(self):
        args = run.parse_args(["--benchmark", "swe_bench", "--work-dir", "/w", "--gold"])
        cfg = run.build_task_config(args, ["swe_bench"], run.resolved_revisions(["swe_bench"], []))
        self.assertEqual((cfg["model"], cfg["eval_type"]), ("gold", "mock_llm"))
        self.assertNotIn("api_url", cfg)
        self.assertNotIn("api_key", cfg)
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            run.parse_args(["--benchmark", "swe_bench", "--work-dir", "/w", "--model", "m"])


class ShellWrapperTest(unittest.TestCase):
    def _run(self, *arguments, env=None):
        with tempfile.TemporaryDirectory() as stub_dir:
            # SKIP_INSTALL=1 runs `python3 -m scripts.evalscope_runner.run`: stub it to echo its argv.
            stub = Path(stub_dir, "python3")
            stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$ARGV_OUT"\n')
            stub.chmod(0o755)
            argv_out = Path(stub_dir, "argv")
            completed = subprocess.run(
                ["bash", "scripts/run_evalscope_eval.sh", *arguments],
                cwd=REPO_ROOT,
                env=os.environ
                | {"PATH": f"{stub_dir}:{os.environ['PATH']}", "SKIP_INSTALL": "1", "ARGV_OUT": str(argv_out)}
                | (env or {}),
                text=True,
                capture_output=True,
            )
            argv = argv_out.read_text().splitlines() if argv_out.exists() else []
        return completed, argv

    def test_requires_benchmark_model_and_endpoint(self):
        completed, _ = self._run("--benchmark", "swe_bench", "--model", "m")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--api-base-url", completed.stderr)
        completed, _ = self._run("--model", "m", "--api-base-url", "http://x")
        self.assertIn("--benchmark", completed.stderr)
        completed, argv = self._run("--benchmark", "swe_bench,gsm8k", "--model", "m", "--api-base-url", "http://x")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("gsm8k", completed.stderr)
        self.assertEqual(argv, [])

    def test_forwards_flags(self):
        with tempfile.TemporaryDirectory() as work:
            completed, argv = self._run(
                "--benchmark", "swe_bench_pro",
                "--model", "org/m", "--api-base-url", "http://host:8000", "--sandbox", "k8s",
                "--limit", "2", "--extra-param", "eval_timeout=7200", "--work-dir", work,
                env={"TARGET_API_KEY": "k"},
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(argv[:2], ["-m", "scripts.evalscope_runner.run"])
        pairs = dict(zip(argv[2::2], argv[3::2]))
        self.assertEqual(pairs["--api-base-url"], "http://host:8000/v1")
        self.assertEqual(pairs["--benchmark"], "swe_bench_pro")
        self.assertEqual(pairs["--sandbox"], "k8s")
        self.assertEqual(pairs["--limit"], "2")
        self.assertEqual(pairs["--extra-param"], "eval_timeout=7200")

    def test_gold(self):
        with tempfile.TemporaryDirectory() as work:
            completed, argv = self._run("--benchmark", "swe_bench", "--gold", "--work-dir", work)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--gold", argv)
        self.assertNotIn("--model", argv)
        self.assertNotIn("--api-base-url", argv)
        completed, argv = self._run("--benchmark", "swe_bench", "--gold", "--model", "m")
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(argv, [])


if __name__ == "__main__":
    unittest.main()
