import asyncio
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
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
    def test_sigterm_becomes_a_normal_exit_so_the_cleanup_runs(self):
        import signal

        with self.assertRaises(SystemExit) as stopped:
            try:
                run.exit_on_sigterm(signal.SIGTERM, None)
            finally:
                cleaned = True
        self.assertTrue(cleaned)
        self.assertEqual(stopped.exception.code, 128 + signal.SIGTERM)

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


@unittest.skipUnless(HAS_ENCLAVE and HAS_OPENSANDBOX, "ms-enclave/opensandbox not installed (evalscope-runtime.lock)")
class PythonExecutorAndPoolTest(unittest.TestCase):
    """What LiveCodeBench's scorer (CodeExecutionSandboxMixin) needs: python_executor on a pool."""

    def _manager(self, ids=("s1",)):
        calls = []
        written = []

        async def run_command(argv, opts=None):
            calls.append(argv)
            return SimpleNamespace(exit_code=0, logs=SimpleNamespace(stdout=_lines("TEST_PASSED"), stderr=[]), error=None)

        async def write_files(entries):
            written.extend(entries)

        manager = opensandbox_sandbox.OpenSandboxSandboxManager()
        for sandbox_id in ids:
            sandbox = SimpleNamespace(
                id=sandbox_id, commands=SimpleNamespace(run=run_command), files=SimpleNamespace(write_files=write_files)
            )
            manager._boxes[sandbox_id] = opensandbox_sandbox._Box(sandbox, {})
        return manager, calls, written

    def test_python_code_goes_through_a_script_file_not_the_command_line(self):
        from ms_enclave.sandbox.model import ExecutionStatus

        manager, calls, written = self._manager()
        code = "print('x')\n" + "# padding\n" * 50000  # far above the 128 KiB limit of one argument
        result = asyncio.run(manager.execute_tool("s1", "python_executor", {"code": code, "timeout": 6}))
        self.assertEqual((result.status, result.output), (ExecutionStatus.SUCCESS, "TEST_PASSED\n"))
        self.assertEqual(written[0].data, code.encode())
        script = written[0].path
        self.assertRegex(script, r"^/tmp/exec_script_[0-9a-f]{32}\.py$")
        self.assertEqual(calls[0], ["timeout", "-s", "TERM", "-k", "5s", "6.0s", "python", script])
        self.assertEqual(calls[1][-3:], ["rm", "-f", script])

    def test_empty_python_code_is_an_error(self):
        from ms_enclave.sandbox.model import ExecutionStatus

        manager, calls, _ = self._manager()
        result = asyncio.run(manager.execute_tool("s1", "python_executor", {"code": "  "}))
        self.assertEqual((result.status, calls), (ExecutionStatus.ERROR, []))

    def test_the_pool_lends_each_sandbox_to_one_caller_at_a_time(self):
        manager, _, _ = self._manager(("s1", "s2"))
        manager._pool_initialized = True
        manager._pool = asyncio.Queue()
        in_use, peak = set(), [0]

        async def execute_tool(sandbox_id, tool_name, parameters):
            self.assertNotIn(sandbox_id, in_use)
            in_use.add(sandbox_id)
            peak[0] = max(peak[0], len(in_use))
            await asyncio.sleep(0.01)
            in_use.discard(sandbox_id)
            return sandbox_id

        async def go():
            for sandbox_id in ("s1", "s2"):
                manager._pool.put_nowait(sandbox_id)
            with mock.patch.object(manager, "execute_tool", side_effect=execute_tool):
                return await asyncio.gather(*(manager.execute_tool_in_pool("python_executor", {}) for _ in range(6)))

        results = asyncio.run(go())
        self.assertEqual(sorted(set(results)), ["s1", "s2"])
        self.assertEqual(peak[0], 2)
        self.assertEqual(manager._pool.qsize(), 2)  # everything returned

    def test_the_pool_must_be_initialized_first(self):
        manager, _, _ = self._manager()
        with self.assertRaises(ValueError):
            asyncio.run(manager.execute_tool_in_pool("python_executor", {}))

    def test_initialize_pool_creates_the_sandboxes(self):
        manager = opensandbox_sandbox.OpenSandboxSandboxManager()
        created = []

        async def create_sandbox(sandbox_type=None, config=None, sandbox_id=None):
            created.append(config)
            return f"s{len(created)}"

        async def go():
            with mock.patch.object(manager, "create_sandbox", side_effect=create_sandbox):
                return await manager.initialize_pool(pool_size=3, sandbox_type=None, config={"image": "python:3.11-slim"})

        ids = asyncio.run(go())
        self.assertEqual((ids, manager._pool_initialized, manager._pool.qsize()), (["s1", "s2", "s3"], True, 3))
        self.assertEqual(asyncio.run(opensandbox_sandbox.OpenSandboxSandboxManager().initialize_pool(pool_size=0)), [])


@unittest.skipUnless(HAS_OPENSANDBOX, "opensandbox not installed (requirements/evalscope-runtime.lock)")
class CreateSandboxTest(unittest.TestCase):
    def test_the_working_dir_is_created_before_any_command_runs_in_it(self):
        # SWE-bench Pro's /sandbox is not in the image; a container runtime would create it.
        calls = []

        async def run_command(argv, opts=None):
            calls.append((argv, opts.working_directory, opts.uid))
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
        self.assertEqual(calls[0][:2], (["timeout", "-s", "TERM", "-k", "5s", "600.0s", "sh", "-c", "mkdir -p /sandbox"], "/"))
        # The microVM's RAM is its memory limit: allocations behave as on a large node, set as root.
        self.assertEqual((calls[1][0][-1], calls[1][2]), ("echo 1 > /proc/sys/vm/overcommit_memory", 0))
        self.assertEqual(calls[2][1], "/")
        self.assertTrue(created["secure_access"])
        self.assertEqual(created["metadata"]["sandbox-run"], "run1")
        self.assertEqual(created["resource"], {"cpu": "2.0", "memory": "8Gi"})
        self.assertEqual(created["network_policy"].default_action, "allow")


@unittest.skipUnless(HAS_OPENSANDBOX, "opensandbox not installed (requirements/evalscope-runtime.lock)")
class AgentNetworkTest(unittest.TestCase):
    def tearDown(self):
        opensandbox_sandbox.AGENT_ALLOWED_HOSTS = None

    def _created(self, env_vars):
        async def run_command(argv, opts=None):
            return SimpleNamespace(exit_code=0, logs=SimpleNamespace(stdout=[], stderr=[]), error=None)

        created = {}

        async def create(image, **kwargs):
            created.update(kwargs)
            return SimpleNamespace(id="sb", commands=SimpleNamespace(run=run_command))

        with mock.patch.dict(os.environ, {"OPENSANDBOX_API_KEY": "k"}), mock.patch(
            "opensandbox.Sandbox.create", side_effect=create
        ):
            asyncio.run(opensandbox_sandbox.OpenSandboxSandboxManager().create_sandbox(
                config={"image": "img", "env_vars": env_vars}))
        return created

    def test_only_the_agents_sandbox_is_restricted(self):
        opensandbox_sandbox.AGENT_ALLOWED_HOSTS = ("pypi.org", "files.pythonhosted.org")
        agent = self._created({"PAGER": "cat", opensandbox_sandbox.AGENT_ROLE_ENV: "agent"})
        policy = agent["network_policy"]
        self.assertEqual(policy.default_action, "deny")
        self.assertEqual([(r.action, r.target) for r in policy.egress],
                         [("allow", "pypi.org"), ("allow", "files.pythonhosted.org")])
        # The marker never reaches the sandbox.
        self.assertEqual(agent["env"], {"PAGER": "cat"})
        # Scoring sandboxes keep their network.
        self.assertEqual(self._created({})["network_policy"].default_action, "allow")

    def test_the_runner_allows_package_indexes_and_the_model_endpoint(self):
        from scripts.common.swe_bench_egress import AGENT_ALLOWED_HOSTS

        def args(*extra):
            return run.parse_args(["--benchmark", "swe_bench", "--work-dir", "w", "--model", "m",
                                   "--api-base-url", "https://api.example.ch/v1", *extra])

        self.assertEqual(run.agent_allowed_hosts(args()), AGENT_ALLOWED_HOSTS + ("api.example.ch",))
        self.assertIsNone(run.agent_allowed_hosts(args("--agent-network", "public")))
        gold = run.parse_args(["--benchmark", "swe_bench", "--work-dir", "w", "--gold"])
        self.assertEqual(run.agent_allowed_hosts(gold), AGENT_ALLOWED_HOSTS)

    def test_public_without_an_allowlist(self):
        created = self._created({opensandbox_sandbox.AGENT_ROLE_ENV: "agent"})
        self.assertEqual(created["network_policy"].default_action, "allow")
        self.assertIsNone(created["env"])


@unittest.skipUnless(importlib.util.find_spec("evalscope") is not None, "evalscope not installed")
class InstallTest(unittest.TestCase):
    def test_agent_environments_are_marked_once_an_allowlist_is_installed(self):
        from evalscope.agent.environments.enclave import EnclaveAgentEnvironment
        from evalscope.api.sandbox.service import SandboxService

        original_init, original_manager = EnclaveAgentEnvironment.__init__, SandboxService._construct_manager
        try:
            opensandbox_sandbox.install(("pypi.org",))
            env = EnclaveAgentEnvironment(engine="docker", sandbox_config={"image": "img", "env_vars": {"A": "1"}})
            self.assertEqual(env._sandbox_config_dict["env_vars"], {"A": "1", opensandbox_sandbox.AGENT_ROLE_ENV: "agent"})
            self.assertEqual(opensandbox_sandbox.AGENT_ALLOWED_HOSTS, ("pypi.org",))
        finally:
            EnclaveAgentEnvironment.__init__, SandboxService._construct_manager = original_init, original_manager
            opensandbox_sandbox.AGENT_ALLOWED_HOSTS = None

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


class _Diagnostics:
    """A sandbox as save_sandbox_logs sees it: its id and the two diagnostics calls."""

    def __init__(self, sandbox_id="sbx-1", fail=False, url=None):
        self.id = sandbox_id
        self.fail = fail
        self.url = url
        self.calls = []

    async def get_diagnostic_events(self, scope):
        self.calls.append(("events", scope))
        if self.fail:
            raise RuntimeError("server down")
        return SimpleNamespace(content="pulled the image", content_url=None, truncated=False)

    async def get_diagnostic_logs(self, scope):
        self.calls.append(("logs", scope))
        if self.url:
            return SimpleNamespace(content=None, content_url=self.url, truncated=True)
        return SimpleNamespace(content="execd started", content_url=None, truncated=False)


class SaveSandboxLogsTest(unittest.TestCase):
    def test_events_and_logs_are_saved_by_sandbox_id(self):
        sandbox = _Diagnostics()
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {opensandbox_sandbox.LOG_DIR_ENV: f"{directory}/opensandbox"}):
                asyncio.run(opensandbox_sandbox.save_sandbox_logs(sandbox))
            saved = Path(directory, "opensandbox", "sbx-1.log").read_text()
        self.assertEqual(
            saved, "----- events -----\npulled the image\n\n----- logs -----\nexecd started\n"
        )
        self.assertEqual(sorted(sandbox.calls), [("events", "all"), ("logs", "all")])

    def test_a_link_of_its_own_is_fetched_and_a_cut_is_said(self):
        sandbox = _Diagnostics(url="https://elsewhere.example/x")
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {opensandbox_sandbox.LOG_DIR_ENV: directory}
        ), mock.patch.object(opensandbox_sandbox, "_fetch_text", return_value="from the link") as fetch:
            asyncio.run(opensandbox_sandbox.save_sandbox_logs(sandbox))
            saved = Path(directory, "sbx-1.log").read_text()
        fetch.assert_called_once_with("https://elsewhere.example/x")
        self.assertIn("from the link\n[the server cut this short]", saved)

    def test_off_without_the_folder_and_never_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ)
            env.pop(opensandbox_sandbox.LOG_DIR_ENV, None)
            with mock.patch.dict(os.environ, env, clear=True):
                sandbox = _Diagnostics()
                asyncio.run(opensandbox_sandbox.save_sandbox_logs(sandbox))
                self.assertEqual(sandbox.calls, [])
            with mock.patch.dict(os.environ, {opensandbox_sandbox.LOG_DIR_ENV: directory}):
                asyncio.run(opensandbox_sandbox.save_sandbox_logs(_Diagnostics(fail=True)))
                asyncio.run(opensandbox_sandbox.save_sandbox_logs(None))
            self.assertEqual(os.listdir(directory), [])

    def test_the_manager_saves_a_sandboxs_logs_before_it_kills_it(self):
        order = []

        class Sandbox:
            id = "sbx-2"

            async def get_diagnostic_events(self, scope):
                order.append("saved")
                return SimpleNamespace(content="", content_url=None, truncated=False)

            async def get_diagnostic_logs(self, scope):
                return SimpleNamespace(content="", content_url=None, truncated=False)

            async def kill(self):
                order.append("killed")

            async def close(self):
                order.append("closed")

        manager = opensandbox_sandbox.OpenSandboxSandboxManager.__new__(opensandbox_sandbox.OpenSandboxSandboxManager)
        manager._boxes = {"sbx-2": SimpleNamespace(sandbox=Sandbox())}
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {opensandbox_sandbox.LOG_DIR_ENV: directory}
        ):
            self.assertTrue(asyncio.run(manager.delete_sandbox("sbx-2")))
        self.assertEqual(order, ["saved", "killed", "closed"])
