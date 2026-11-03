"""Offline tests for Inspect's opensandbox provider (scripts/inspect_ai/opensandbox_sandbox.py).

No server: the module's _run/_kill are replaced, and the OpenSandbox SDK's WriteEntry is stubbed
when the SDK isn't installed (the default CI job has Inspect but not the SDK).
"""

import asyncio
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]

try:
    import inspect_ai  # noqa: F401
except ImportError:  # pragma: no cover
    inspect_ai = None


def _stub_sdk() -> None:
    """opensandbox.models.filesystem.WriteEntry, when the SDK isn't installed."""
    try:
        import opensandbox.models.filesystem  # noqa: F401
    except ImportError:
        class WriteEntry:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        for name in ("opensandbox", "opensandbox.models"):
            sys.modules.setdefault(name, types.ModuleType(name))
        filesystem = types.ModuleType("opensandbox.models.filesystem")
        filesystem.WriteEntry = WriteEntry
        sys.modules["opensandbox.models.filesystem"] = filesystem


@unittest.skipIf(inspect_ai is None, "inspect_ai is not installed")
class ServiceSpecTest(unittest.TestCase):
    def setUp(self):
        from scripts.inspect_ai import opensandbox_sandbox

        self.m = opensandbox_sandbox
        self.tmp = Path(tempfile.mkdtemp())

    def write(self, name, text):
        path = self.tmp / name
        path.write_text(text)
        return str(path)

    def test_no_config_is_inspect_s_default_image(self):
        (spec,) = self.m.service_specs(None)
        self.assertEqual(spec.image, self.m.DEFAULT_IMAGE)

    def test_a_dockerfile_is_replayed_from_its_directory(self):
        path = self.write("Dockerfile", "FROM python:3.11-slim\n")
        (spec,) = self.m.service_specs(path)
        self.assertEqual(spec.dockerfile, Path(path).resolve())
        self.assertEqual(spec.context, self.tmp.resolve())
        self.assertIsNone(spec.image)

    def test_a_compose_service_is_read_as_docker_would(self):
        path = self.write(
            "compose.yaml",
            "services:\n  default:\n    image: python:3.12-slim\n    working_dir: /app\n"
            "    environment:\n      - A=1\n    user: agent\n    network_mode: none\n"
            "    cpus: 1.0\n    mem_limit: 0.5gb\n",
        )
        (spec,) = self.m.service_specs(path)
        self.assertEqual(
            (spec.image, spec.working_dir, spec.env, spec.user, spec.network_none, spec.cpu, spec.memory),
            ("python:3.12-slim", "/app", {"A": "1"}, "agent", True, 1.0, "0.5gb"),
        )

    def test_a_compose_build_points_at_its_context_and_dockerfile(self):
        (self.tmp / "ctx").mkdir()
        path = self.write("compose.yaml", "services:\n  default:\n    build:\n      context: ctx\n      dockerfile: Dockerfile.eval\n")
        (spec,) = self.m.service_specs(path)
        self.assertEqual(spec.dockerfile, (self.tmp / "ctx" / "Dockerfile.eval").resolve())
        self.assertFalse(spec.network_none)
        self.assertEqual((spec.cpu, spec.memory), (self.m.DEFAULT_CPU, self.m.DEFAULT_MEMORY))

    def test_several_services_are_refused(self):
        path = self.write("compose.yaml", "services:\n  default:\n    image: a\n  victim:\n    image: b\n")
        with self.assertRaisesRegex(ValueError, "2 services"):
            self.m.service_specs(path)

    def test_the_task_s_own_file_is_carried_over_and_none_is_auto_discovered(self):
        provider = self.m.OpenSandboxEnvironment
        self.assertTrue(provider.is_docker_compatible())
        self.assertEqual(provider.config_files(), [])


class _Files:
    def __init__(self):
        self.written = {}
        self.content = b""

    async def write_files(self, entries):
        for entry in entries:
            self.written[entry.path] = entry.data

    async def read_bytes(self, path):
        return self.content


@unittest.skipIf(inspect_ai is None, "inspect_ai is not installed")
class ExecTest(unittest.TestCase):
    def setUp(self):
        _stub_sdk()
        from scripts.common.dockerfile_replay import ImageConfig
        from scripts.inspect_ai import opensandbox_sandbox

        self.m = opensandbox_sandbox
        self.sandbox = types.SimpleNamespace(id="sb-1", files=_Files())
        spec = self.m.ServiceSpec(name="default", image="x", working_dir="/work", env={"A": "1"})
        self.env = self.m.OpenSandboxEnvironment(self.sandbox, ImageConfig(env={"B": "2"}), spec, 1000, "/usr/bin/timeout")
        self.calls = []
        self.results = []

        async def fake_run(sandbox, argv, cwd=None, uid=None, env=None, timeout=None):
            self.calls.append({"argv": argv, "cwd": cwd, "uid": uid, "env": env, "timeout": timeout})
            return self.results.pop(0) if self.results else (0, "", "")

        patcher = mock.patch.object(self.m, "_run", fake_run)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_(self, coro):
        return asyncio.run(coro)

    def test_a_command_runs_as_argv_in_the_working_dir_with_the_image_env(self):
        self.results = [(0, "out\n", "")]
        result = self.run_(self.env.exec(["ls", "-l"]))
        self.assertTrue(result.success)
        self.assertEqual(result.stdout, "out\n")
        call = self.calls[0]
        self.assertEqual(call["argv"], ["ls", "-l"])
        self.assertEqual((call["cwd"], call["uid"], call["env"]), ("/work", 1000, {"A": "1", "B": "2"}))

    def test_cwd_env_and_timeout(self):
        self.run_(self.env.exec(["make"], cwd="sub", env={"C": "3"}, timeout=10))
        call = self.calls[0]
        self.assertEqual(call["cwd"], "/work/sub")
        self.assertEqual(call["env"], {"A": "1", "B": "2", "C": "3"})
        # The wrapper by absolute path, never through the image's PATH.
        self.assertEqual(call["argv"][:6], ["/usr/bin/timeout", "-s", "TERM", "-k", "5s", "10s"])

    def test_stdin_goes_through_a_file(self):
        self.run_(self.env.exec(["cat"], input="hello"))
        (staged,) = self.sandbox.files.written
        self.assertEqual(self.sandbox.files.written[staged], b"hello")
        self.assertEqual(self.calls[0]["argv"], [self.m.SH, "-c", 'exec "$@" < "$0"', staged, "cat"])
        self.assertEqual(self.calls[-1]["argv"], ["rm", "-f", staged])  # cleaned up

    def test_an_oversized_c_argument_goes_through_a_file(self):
        code = "x = 1\n" * 20000
        self.run_(self.env.exec(["python", "-c", code, "arg"]))
        (staged,) = self.sandbox.files.written
        self.assertEqual(self.sandbox.files.written[staged], code.encode())
        self.assertEqual(self.calls[0]["argv"], ["python", staged, "arg"])

    def test_a_small_c_argument_stays_inline(self):
        self.run_(self.env.exec(["python", "-c", "print(1)"]))
        self.assertEqual(self.calls[0]["argv"], ["python", "-c", "print(1)"])
        self.assertEqual(self.sandbox.files.written, {})

    def test_another_user_is_looked_up_once(self):
        self.results = [(0, "0\n", ""), (0, "", ""), (0, "", "")]
        self.run_(self.env.exec(["id"], user="root"))
        self.run_(self.env.exec(["id"], user="root"))
        self.assertIn("id -u root", self.calls[0]["argv"][-1])
        self.assertEqual([c["uid"] for c in self.calls[1:]], [0, 0])

    def test_a_timed_out_command_raises_timeout_error_after_its_retries(self):
        self.results = [(124, "", ""), (-1, "", ""), (137, "", "")]
        clock = iter(range(0, 1000, 100))
        with mock.patch.object(self.m, "_now", lambda: next(clock)):
            with self.assertRaises(TimeoutError):
                self.run_(self.env.exec(["sleep", "999"], timeout=90))
        self.assertEqual([c["argv"][5] for c in self.calls], ["90s", "60s", "30s"])

    def test_an_exit_code_is_not_a_timeout_when_the_time_was_not_up(self):
        self.results = [(124, "", "")]
        clock = iter([0, 1])
        with mock.patch.object(self.m, "_now", lambda: next(clock)):
            result = self.run_(self.env.exec(["false"], timeout=90))
        self.assertEqual(result.returncode, 124)

    def test_permission_denied_on_exec(self):
        self.results = [(126, "", "sh: ./x: Permission denied")]
        with self.assertRaises(PermissionError):
            self.run_(self.env.exec(["./x"]))

    def test_read_file_errors(self):
        for code, error in ((21, IsADirectoryError), (2, FileNotFoundError), (13, PermissionError)):
            self.results = [(code, "", "")]
            with self.assertRaises(error):
                self.run_(self.env.read_file("f"))

    def test_read_file_keeps_crlf(self):
        self.sandbox.files.content = b"a\r\nb"
        self.results = [(0, "4\n", "")]
        self.assertEqual(self.run_(self.env.read_file("f")), "a\r\nb")
        self.assertEqual(self.calls[0]["uid"], 1000)  # checked as the default user

    def test_write_file_as_the_default_user(self):
        self.run_(self.env.write_file("d/f.txt", "x"))
        (staged,) = self.sandbox.files.written
        call = self.calls[0]
        self.assertEqual(call["uid"], 1000)
        self.assertIn("mkdir -p /work/d", call["argv"][2])
        self.assertEqual(call["argv"][3], staged)

    def test_write_file_errors(self):
        self.results = [(21, "", "")]
        with self.assertRaises(IsADirectoryError):
            self.run_(self.env.write_file("d", "x"))
        self.results = [(1, "", "sh: cannot create /etc/x: Permission denied")]
        with self.assertRaises(PermissionError):
            self.run_(self.env.write_file("/etc/x", "x"))


@unittest.skipIf(inspect_ai is None, "inspect_ai is not installed")
class CleanupTest(unittest.TestCase):
    def test_a_task_only_kills_its_own_sandboxes(self):
        """Tasks of one eval-set share the run label: one ending must not end the others'."""
        from scripts.inspect_ai import opensandbox_sandbox as m

        killed = []

        async def fake_kill(sandbox):
            killed.append(sandbox)

        m._LIVE.clear()
        m._LIVE["task_a"] = {"1": "sandbox-a"}
        m._LIVE["task_b"] = {"2": "sandbox-b"}
        with mock.patch.object(m, "_kill", fake_kill), mock.patch.object(m, "delete_run_sandboxes") as by_label:
            asyncio.run(m.OpenSandboxEnvironment.task_cleanup("task_a", None, True))
        self.assertEqual(killed, ["sandbox-a"])
        by_label.assert_not_called()
        self.assertIn("task_b", m._LIVE)
        m._LIVE.clear()


@unittest.skipIf(inspect_ai is None, "inspect_ai is not installed")
class RegistrationTest(unittest.TestCase):
    def test_sitecustomize_registers_the_provider(self):
        """run_inspect_eval.sh puts sandbox_patches on PYTHONPATH; --sandbox opensandbox resolves."""
        env = {**os.environ, "PYTHONPATH": str(REPO / "scripts" / "inspect_ai" / "sandbox_patches")}
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                "from inspect_ai.util._sandbox.registry import registry_find_sandboxenv;"
                "print(registry_find_sandboxenv('opensandbox').__name__)",
            ],
            cwd=tempfile.gettempdir(),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(out.stdout.strip(), "OpenSandboxEnvironment")


if __name__ == "__main__":
    unittest.main()
