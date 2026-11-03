"""Offline tests for GDPval's OpenSandbox code backend (scripts/aaii/gdpval_opensandbox.py).

Needs Stirrup (requirements/gdpval-runtime.lock); skipped where it isn't installed, as in the
default CI job. No server: the provider's _exec is replaced.
"""

import asyncio
import shlex
import unittest
from unittest import mock

try:
    from stirrup.tools.code_backends.base import CommandResult
except ImportError:  # pragma: no cover
    CommandResult = None


@unittest.skipIf(CommandResult is None, "stirrup is not installed")
class GdpvalOpenSandboxTest(unittest.TestCase):
    def setUp(self):
        from scripts.aaii import gdpval_opensandbox

        self.m = gdpval_opensandbox
        self.provider = self.m.OpenSandboxCodeExecToolProvider()
        self.provider._sbx = object()  # started
        self.provider._uid = 1000
        self.calls = []
        self.results = []

        async def fake_exec(script, timeout, uid, cwd=self.m.WORKING_DIR):
            self.calls.append({"script": script, "timeout": timeout, "uid": uid, "cwd": cwd})
            return self.results.pop(0) if self.results else CommandResult(exit_code=0, stdout="", stderr="")

        self.provider._exec = fake_exec

    def test_a_command_runs_through_a_shell_as_user_under_timeout(self):
        self.results = [CommandResult(exit_code=0, stdout="hi\n", stderr="")]
        result = asyncio.run(self.provider.run_command("echo hi | cat", timeout=7))
        self.assertEqual(result.stdout, "hi\n")
        call = self.calls[0]
        self.assertEqual(call["script"], "exec /usr/bin/timeout -s TERM -k 5s 7s /bin/sh -c 'echo hi | cat'")
        self.assertEqual((call["uid"], call["cwd"]), (1000, self.m.WORKING_DIR))

    def test_a_timed_out_command_is_reported_as_one(self):
        for code in (124, 137, -1):
            self.results = [CommandResult(exit_code=code, stdout="", stderr="")]
            clock = iter([0, 10])
            with mock.patch.object(self.m, "_now", lambda: next(clock)):
                result = asyncio.run(self.provider.run_command("sleep 99", timeout=3))
            self.assertEqual(result.error_kind, "timeout", code)

    def test_an_allowlist_runs_the_parsed_argv_with_nothing_expanded(self):
        provider = self.m.OpenSandboxCodeExecToolProvider(allowed_commands=[r"^python"])
        provider._sbx, provider._uid, provider._exec = object(), 1000, self.provider._exec
        asyncio.run(provider.run_command("python -c 'print($HOME)'"))
        # The inner shell gets the argv re-quoted (as Stirrup's E2B backend does): $HOME stays literal.
        inner = shlex.split(self.calls[0]["script"])[-1]
        self.assertEqual(inner, self.m._quote_argv_for_shell(["python", "-c", "print($HOME)"]))
        self.assertEqual(shlex.split(inner), ["python", "-c", "print($HOME)"])
        rejected = asyncio.run(provider.run_command("rm -rf /"))
        self.assertNotEqual(rejected.exit_code, 0)
        self.assertEqual(len(self.calls), 1)

    def test_outputs_outside_the_working_dir_are_refused(self):
        self.results = [CommandResult(exit_code=0, stdout="/etc/passwd\n", stderr="")]
        with self.assertRaises(ValueError):
            asyncio.run(self.provider.resolve_output_source("/etc/passwd"))

    def test_list_files_is_relative(self):
        self.results = [
            CommandResult(exit_code=0, stdout="", stderr=""),  # test -d
            CommandResult(exit_code=0, stdout="/home/user/out/a.md\n/home/user/out/sub/b.csv\n", stderr=""),
        ]
        self.assertEqual(asyncio.run(self.provider.list_files("out")), ["a.md", "sub/b.csv"])

    def test_set_up_runs_as_root_from_slash(self):
        asyncio.run(self.provider._root("true", "x"))
        self.assertEqual((self.calls[0]["uid"], self.calls[0]["cwd"]), (0, "/"))


if __name__ == "__main__":
    unittest.main()
