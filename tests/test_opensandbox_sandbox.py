import asyncio
import importlib.util
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from scripts.common import opensandbox_sandbox
from scripts.evalscope import run

HAS_ENCLAVE = importlib.util.find_spec("ms_enclave") is not None
HAS_OPENSANDBOX = importlib.util.find_spec("opensandbox") is not None


def _lines(*texts):
    return [SimpleNamespace(text=t) for t in texts]


class HelpersTest(unittest.TestCase):
    def test_resources_follow_the_sandbox_config(self):
        self.assertEqual(
            opensandbox_sandbox.resources({"memory_limit": "8g", "cpu_limit": 4}), {"cpu": "4", "memory": "8Gi"}
        )
        # ms-enclave's Docker defaults otherwise
        self.assertEqual(opensandbox_sandbox.resources({}), {"cpu": "2.0", "memory": "4Gi"})

    def test_output_is_rebuilt_from_its_lines(self):
        self.assertEqual(opensandbox_sandbox.joined_lines(_lines("a", "\n", "b")), "a\n\nb\n")
        self.assertEqual(opensandbox_sandbox.joined_lines([]), "")

    def test_domain_defaults_to_breithorns_kata_endpoint(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(opensandbox_sandbox.domain(), "opensandbox-kata.breithorn.svc.cscs.ch")
        with mock.patch.dict(os.environ, {"OPENSANDBOX_DOMAIN": "other"}, clear=True):
            self.assertEqual(opensandbox_sandbox.domain(), "other")

    @unittest.skipUnless(HAS_OPENSANDBOX, "opensandbox not installed (requirements/evalscope-runtime.lock)")
    def test_platform_becomes_a_platform_spec(self):
        spec = opensandbox_sandbox.platform_spec({"platform": "linux/amd64"})
        self.assertEqual((spec.os, spec.arch), ("linux", "amd64"))
        self.assertIsNone(opensandbox_sandbox.platform_spec({}))


class RunnerTest(unittest.TestCase):
    def test_opensandbox_needs_a_key(self):
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(SystemExit):
            run.parse_args(["--benchmark", "swe_bench", "--gold", "--work-dir", "w", "--sandbox", "opensandbox"])
        with mock.patch.dict(os.environ, {"OPENSANDBOX_API_KEY": "k"}, clear=True):
            args = run.parse_args(["--benchmark", "swe_bench", "--gold", "--work-dir", "w", "--sandbox", "opensandbox"])
        self.assertEqual(args.sandbox, "opensandbox")

    def test_opensandbox_sandboxes_get_8g_unless_given(self):
        args = SimpleNamespace(sandbox="opensandbox", sandbox_config="{}")
        self.assertEqual(run.sandbox_config(args), {"memory_limit": "8g"})
        args.sandbox_config = '{"memory_limit": "16g", "cpu_limit": 4}'
        self.assertEqual(run.sandbox_config(args), {"memory_limit": "16g", "cpu_limit": 4})
        self.assertEqual(run.sandbox_config(SimpleNamespace(sandbox="docker", sandbox_config="{}")), {})


@unittest.skipUnless(HAS_ENCLAVE and HAS_OPENSANDBOX, "ms-enclave/opensandbox not installed (evalscope-runtime.lock)")
class ExecuteToolTest(unittest.TestCase):
    def _manager(self, *results):
        """A manager with one sandbox whose commands return results in order: (exit code, stdout
        lines, stderr lines[, error])."""
        calls = []

        async def run_command(argv, opts=None):
            calls.append((argv, opts))
            code, out, err, *error = results[len(calls) - 1]
            return SimpleNamespace(
                exit_code=code, logs=SimpleNamespace(stdout=_lines(*out), stderr=_lines(*err)), error=error[0] if error else None
            )

        sandbox = SimpleNamespace(id="sb-1", commands=SimpleNamespace(run=run_command))
        manager = opensandbox_sandbox.OpenSandboxSandboxManager()
        manager._boxes["s1"] = opensandbox_sandbox._Box(sandbox, {"working_dir": "/testbed"})
        return manager, calls

    @staticmethod
    def _clock(*readings):
        # The module's own clock only: the event loop keeps the real time.monotonic.
        return SimpleNamespace(monotonic=mock.Mock(side_effect=list(readings)))

    def _tool(self, manager, command="echo hi", timeout=30):
        return asyncio.run(manager.execute_tool("s1", "shell_executor", {"command": command, "timeout": timeout}))

    def test_success_runs_under_gnu_timeout_in_the_working_dir(self):
        from ms_enclave.sandbox.model import ExecutionStatus

        manager, calls = self._manager((0, ["hi"], []))
        result = self._tool(manager, ["bash", "-lc", "echo hi"])
        self.assertEqual((result.status, result.output, result.error), (ExecutionStatus.SUCCESS, "hi\n", None))
        argv, opts = calls[0]
        self.assertEqual(argv, ["timeout", "-s", "TERM", "-k", "5s", "30.0s", "bash", "-lc", "echo hi"])
        self.assertEqual(opts.working_directory, "/testbed")
        # the request's own timeout is only a backstop above the command's
        self.assertEqual(opts.timeout.total_seconds(), 90)

    def test_a_failing_command_reports_its_exit_code(self):
        from ms_enclave.sandbox.model import ExecutionStatus

        manager, _ = self._manager((2, [], []))
        result = self._tool(manager)
        self.assertEqual((result.status, result.error), (ExecutionStatus.ERROR, "Command failed with exit code 2"))

    def test_gnu_timeout_is_a_timeout_only_after_the_limit(self):
        from ms_enclave.sandbox.model import ExecutionStatus

        manager, _ = self._manager((124, [], []))
        with mock.patch.object(opensandbox_sandbox, "time", self._clock(0.0, 31.0)):
            self.assertEqual(self._tool(manager).status, ExecutionStatus.TIMEOUT)
        manager, _ = self._manager((124, [], []))
        with mock.patch.object(opensandbox_sandbox, "time", self._clock(0.0, 1.0)):
            self.assertEqual(self._tool(manager).status, ExecutionStatus.ERROR)

    def test_a_dropped_stream_is_never_success(self):
        from ms_enclave.sandbox.model import ExecutionStatus

        manager, _ = self._manager((None, [], []))
        result = self._tool(manager)
        self.assertEqual(result.status, ExecutionStatus.ERROR)
        self.assertIn("stream ended without a result", result.error)

    def test_an_execution_error_without_exit_code_is_a_failure(self):
        from ms_enclave.sandbox.model import ExecutionStatus

        manager, _ = self._manager((None, [], [], "OOM"))
        result = self._tool(manager)
        self.assertEqual((result.status, result.error), (ExecutionStatus.ERROR, "OOM"))


@unittest.skipUnless(HAS_OPENSANDBOX, "opensandbox not installed (requirements/evalscope-runtime.lock)")
class CreateSandboxTest(unittest.TestCase):
    def test_the_working_dir_is_created_before_any_command_runs_in_it(self):
        # SWE-bench Pro's /sandbox is not in the image; a container runtime would create it.
        calls = []

        async def run_command(argv, opts=None):
            calls.append((argv, opts.working_directory))
            return SimpleNamespace(exit_code=0, logs=SimpleNamespace(stdout=[], stderr=[]), error=None)

        sandbox = SimpleNamespace(id="sb-1", commands=SimpleNamespace(run=run_command))
        created = {}

        async def create(image, **kwargs):
            created.update(kwargs, image=image)
            return sandbox

        config = {"image": "img", "working_dir": "/sandbox", "memory_limit": "8g", "platform": "linux/amd64"}
        with mock.patch.dict(os.environ, {"OPENSANDBOX_API_KEY": "k", "OPENSANDBOX_RUN_LABEL": "run1"}), mock.patch(
            "opensandbox.Sandbox.create", side_effect=create
        ):
            manager = opensandbox_sandbox.OpenSandboxSandboxManager()
            sandbox_id = asyncio.run(manager.create_sandbox(config=config))
        self.assertIn(sandbox_id, manager._boxes)
        self.assertEqual(calls[0], (["timeout", "-s", "TERM", "-k", "5s", "600.0s", "sh", "-c", "mkdir -p /sandbox"], "/"))
        self.assertEqual(calls[1][1], "/")
        self.assertTrue(created["secure_access"])
        self.assertEqual(created["metadata"]["sandbox-run"], "run1")
        self.assertEqual(created["resource"], {"cpu": "2.0", "memory": "8Gi"})
        self.assertEqual(created["network_policy"].default_action, "allow")


@unittest.skipUnless(importlib.util.find_spec("evalscope") is not None, "evalscope not installed")
class InstallTest(unittest.TestCase):
    def test_install_routes_the_docker_engine_to_opensandbox(self):
        from evalscope.api.sandbox import SandboxEngine
        from evalscope.api.sandbox.service import SandboxService

        original = SandboxService._construct_manager
        try:
            opensandbox_sandbox.install()
            manager = SandboxService._construct_manager(SandboxService.__new__(SandboxService), SandboxEngine.DOCKER, {})
            self.assertIsInstance(manager, opensandbox_sandbox.OpenSandboxSandboxManager)
        finally:
            SandboxService._construct_manager = original


if __name__ == "__main__":
    unittest.main()
