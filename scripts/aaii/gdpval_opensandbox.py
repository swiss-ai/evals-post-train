"""GDPval's code-execution sandbox as a Kata microVM on an OpenSandbox server
(`run_gdpval.sh --sandbox-backend opensandbox`): a Stirrup CodeExecToolProvider in place of E2B's.

It mirrors Stirrup's E2BCodeExecToolProvider (stirrup/tools/code_backends/e2b.py, 0.2.1), so the
agent sees the same sandbox: commands run through a shell as the unprivileged `user` in
/home/user (E2B's home; Stirrup resolves output files under it), with internet access, as an E2B
sandbox has. The default image is E2B's own code-interpreter image (the public image behind the
code-interpreter-v1 template: Python 3.12 with its data, plotting and document libraries),
pinned by digest; --opensandbox-image picks another. That image runs as root and has no `user`,
so the provider creates one at start-up.

One sandbox per task, killed when the task ends; every sandbox of a run carries its run label
(OPENSANDBOX_RUN_LABEL), and run_gdpval.sh deletes leftovers of the label at the end. The server
is OPENSANDBOX_DOMAIN (default: Breithorn's Kata endpoint) with OPENSANDBOX_API_KEY.
"""

from __future__ import annotations

import os
import shlex
import time
import uuid
from datetime import timedelta
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError
from stirrup.core.models import ImageContentBlock, Tool, ToolUseCountMetadata
from stirrup.tools.code_backends.base import (
    SHELL_TIMEOUT,
    CodeExecToolProvider,
    CodeExecutionParams,
    CommandResult,
    OutputSourceRoot,
    UploadedFile,
    UploadFilesResult,
)

from scripts.common.opensandbox_sandbox import (
    save_sandbox_logs,
    HEALTH_POLL_SEC,
    OVERCOMMIT_COMMAND,
    READY_TIMEOUT_SEC,
    REQUEST_SLACK_SEC,
    RUN_LABEL_ENV,
    RUN_LABEL_KEY,
    connection_config,
    joined_lines,
)

# docker.io/e2bdev/code-interpreter:python-3.12.8 (amd64, ~1.7 GB), the image behind E2B's
# code-interpreter-v1 template.
DEFAULT_IMAGE = "e2bdev/code-interpreter@sha256:9d5f81fff62a6ada2fcb1b7fcf14d08f6f2fcfd4706476057af7c4905eba614f"
WORKING_DIR = "/home/user"
SANDBOX_USER = "user"
# A task's sandbox lives at most this long (250 turns of an agent loop).
SANDBOX_TIMEOUT_SEC = 6 * 3600
RESOURCES = {"cpu": "2", "memory": "4Gi"}
SH = "/bin/sh"
TIMEOUT_EXIT_CODES = (124, 137, -1)
# Creates E2B's `user` (home /home/user) when the image has none, and gives it the working dir.
_SETUP = (
    f"id -u {SANDBOX_USER} >/dev/null 2>&1 || useradd -m -d {WORKING_DIR} -s /bin/bash {SANDBOX_USER} "
    f"|| adduser -D -h {WORKING_DIR} {SANDBOX_USER}; "
    f"mkdir -p {WORKING_DIR} && chown {SANDBOX_USER}: {WORKING_DIR}"
)


_now = time.monotonic  # replaced in tests (asyncio's own clock must stay real)


def run_label() -> str:
    label = os.environ.get(RUN_LABEL_ENV)
    if not label:
        label = f"gdpval-{uuid.uuid4().hex[:12]}"
        os.environ[RUN_LABEL_ENV] = label
    return label


def _quote_argv_for_shell(argv: list[str]) -> str:
    """argv as a shell command that runs exactly those arguments (as Stirrup's E2B backend)."""
    head = "'" + argv[0].replace("'", "'\"'\"'") + "'"
    return f"{head} {shlex.join(argv[1:])}" if len(argv) > 1 else head


class OpenSandboxCodeExecToolProvider(CodeExecToolProvider):
    """Stirrup code execution in an OpenSandbox (Kata) sandbox. See the module docstring."""

    def __init__(
        self,
        *,
        image: str = DEFAULT_IMAGE,
        timeout: int = SANDBOX_TIMEOUT_SEC,
        allowed_commands: list[str] | None = None,
        shell_timeout: int = SHELL_TIMEOUT,
    ) -> None:
        super().__init__(allowed_commands=allowed_commands, shell_timeout=shell_timeout)
        self._image = image
        self._timeout = timeout
        self._sbx: Any = None
        self._uid: int | None = None

    # --- lifecycle -----------------------------------------------------------------------------

    async def __aenter__(self) -> Tool[CodeExecutionParams, ToolUseCountMetadata]:
        from opensandbox import Sandbox
        from opensandbox.models.sandboxes import NetworkPolicy

        if not os.environ.get("OPENSANDBOX_API_KEY"):
            raise RuntimeError("--sandbox-backend opensandbox needs OPENSANDBOX_API_KEY")
        self._sbx = await Sandbox.create(
            self._image,
            timeout=timedelta(seconds=self._timeout),
            ready_timeout=timedelta(seconds=READY_TIMEOUT_SEC),
            metadata={RUN_LABEL_KEY: run_label(), "benchmark": "gdpval"},
            resource=RESOURCES,
            # Open internet, as an E2B sandbox has.
            network_policy=NetworkPolicy(defaultAction="allow", egress=[]),
            secure_access=True,
            connection_config=connection_config(),
            health_check_polling_interval=timedelta(seconds=HEALTH_POLL_SEC),
        )
        try:
            await self._root(OVERCOMMIT_COMMAND, "set vm.overcommit_memory")
            await self._root(_SETUP, f"create {SANDBOX_USER} and {WORKING_DIR}")
            self._uid = int((await self._root(f"id -u {SANDBOX_USER}", f"look up {SANDBOX_USER}")).strip())
        except BaseException:
            await self._close()
            raise
        return self.get_code_exec_tool()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self._close()

    async def _close(self) -> None:
        if self._sbx is not None:
            sandbox, self._sbx = self._sbx, None
            try:
                await save_sandbox_logs(sandbox)
                await sandbox.kill()
            finally:
                await sandbox.close()

    def _require(self) -> Any:
        if self._sbx is None:
            raise RuntimeError(
                "ExecutionEnvironment not started. Ensure current Agent is equipped with a CodeExecToolProvider."
            )
        return self._sbx

    # --- commands ------------------------------------------------------------------------------

    async def _exec(self, script: str, timeout: float, uid: int | None, cwd: str = WORKING_DIR) -> CommandResult:
        from opensandbox.models.execd import RunCommandOpts

        sandbox = self._require()
        execution = await sandbox.commands.run(
            [SH, "-c", script],
            opts=RunCommandOpts(
                working_directory=cwd,
                timeout=timedelta(seconds=float(timeout) + REQUEST_SLACK_SEC),
                uid=uid,
                envs={"HOME": WORKING_DIR, "USER": SANDBOX_USER} if uid else None,
            ),
        )
        code = execution.exit_code
        stderr = joined_lines(execution.logs.stderr)
        if code is None:
            code = 1
            stderr = stderr or f"{execution.error or 'the command ended without a result'}"
        return CommandResult(exit_code=code, stdout=joined_lines(execution.logs.stdout), stderr=stderr)

    async def _root(self, script: str, what: str) -> str:
        # From /: the set-up creates the working dir.
        result = await self._exec(script, 600, uid=0, cwd="/")
        if result.exit_code != 0:
            raise RuntimeError(f"opensandbox: {what} failed (exit {result.exit_code}): {result.stderr.strip()[:500]}")
        return result.stdout

    async def run_command(self, cmd: str, *, timeout: int | None = None) -> CommandResult:
        self._require()
        if timeout is None:
            timeout = self._shell_timeout
        argv, rejection = self._prepare_command(cmd)
        if rejection is not None:
            return rejection
        if argv is not None:
            cmd = _quote_argv_for_shell(argv)
        # GNU timeout by absolute path (never through the image's PATH), as E2B's per-command
        # timeout: TERM, then KILL 5 s later.
        script = f"exec /usr/bin/timeout -s TERM -k 5s {int(timeout)}s {SH} -c {shlex.quote(cmd)}"
        started = _now()
        try:
            result = await self._exec(script, timeout, uid=self._uid)
        except Exception as exc:  # the request itself failed or ran out of time
            return CommandResult(exit_code=1, stdout="", stderr=str(exc), error_kind="timeout" if "time" in str(exc).lower() else None)
        # timeout's own 124/137, or -1: how the server reports a command ended by a signal.
        if result.exit_code in TIMEOUT_EXIT_CODES and _now() - started + 0.05 >= timeout:
            return CommandResult(
                exit_code=1,
                stdout=result.stdout,
                stderr=result.stderr or f"Command timed out after {timeout} seconds",
                error_kind="timeout",
            )
        return result

    # --- files -----------------------------------------------------------------------------------

    def _abs(self, path: str) -> str:
        return path if path.startswith("/") else f"{WORKING_DIR}/{path}"

    async def _test(self, flag: str, path: str) -> bool:
        result = await self._exec(f"test {flag} {shlex.quote(self._abs(path))}", 60, uid=self._uid)
        return result.exit_code == 0

    async def read_file_bytes(self, path: str) -> bytes:
        if not await self._test("-e", path):
            raise FileNotFoundError(f"File not found: {path}")
        return bytes(await self._require().files.read_bytes(self._abs(path)))

    async def write_file_bytes(self, path: str, content: bytes) -> None:
        from opensandbox.models.filesystem import WriteEntry

        target = self._abs(path)
        await self._exec(f"mkdir -p {shlex.quote(str(PurePosixPath(target).parent))}", 60, uid=self._uid)
        await self._require().files.write_files(
            [WriteEntry(path=target, data=content, mode=644, owner=SANDBOX_USER, group=SANDBOX_USER)]
        )

    async def file_exists(self, path: str) -> bool:
        return await self._test("-e", path)

    async def is_directory(self, path: str) -> bool:
        return await self._test("-d", path)

    async def list_files(self, path: str) -> list[str]:
        if not await self.is_directory(path):
            return []
        root = self._abs(path).rstrip("/")
        result = await self._exec(f"find {shlex.quote(root)} -type f", 120, uid=self._uid)
        if result.exit_code != 0:
            return []
        prefix = f"{root}/"
        return [line.removeprefix(prefix) for line in result.stdout.splitlines() if line.startswith(prefix)]

    # --- output paths (as E2B's: files under the working dir only) --------------------------------

    def output_source_roots(self) -> tuple[OutputSourceRoot, ...]:
        return (OutputSourceRoot(path=WORKING_DIR),)

    async def resolve_output_source(self, source_path: str) -> str:
        result = await self._exec(f"realpath -e -- {shlex.quote(self._abs(source_path))}", 60, uid=self._uid)
        resolved = result.stdout.strip()
        if result.exit_code != 0 or not resolved:
            raise ValueError(f"Could not resolve output source: {source_path}")
        try:
            PurePosixPath(resolved).relative_to(WORKING_DIR)
        except ValueError as exc:
            raise ValueError(f"Output source is outside the execution root: {source_path}") from exc
        if not await self._test("-f", resolved):
            raise ValueError(f"Path is not a file: {source_path}")
        return resolved

    async def output_destination_identity(self, destination: str, output_root: Path | str) -> str:
        quoted = f"{shlex.quote(str(output_root))} {shlex.quote(destination)}"
        result = await self._exec(f"realpath -m -- {quoted}", 60, uid=self._uid)
        paths = result.stdout.split()
        if result.exit_code != 0 or len(paths) != 2:
            raise ValueError(f"Could not resolve cross-environment output destination: {destination}")
        root, resolved = map(PurePosixPath, paths)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Output destination escapes output directory: {destination}") from exc
        return resolved.as_posix()

    async def upload_files(
        self,
        *paths: Path | str,
        source_env: CodeExecToolProvider | None = None,
        dest_dir: str | None = None,
    ) -> UploadFilesResult:
        if source_env is not None:
            return await super().upload_files(*paths, source_env=source_env, dest_dir=dest_dir)
        self._require()
        dest_base = dest_dir or WORKING_DIR
        result = UploadFilesResult()
        for source in paths:
            name = Path(source).name
            source = Path(source).resolve()
            if not source.exists():
                result.failed[str(source)] = "File or directory does not exist"
                continue
            try:
                files = [source] if source.is_file() else [p for p in source.rglob("*") if p.is_file()]
                for file_path in files:
                    if source.is_file():
                        dest = f"{dest_base}/{name}"
                    else:
                        relative = file_path.relative_to(source)
                        dest = f"{dest_base}/{relative}" if dest_dir else f"{dest_base}/{source.name}/{relative}"
                    content = file_path.read_bytes()
                    await self.write_file_bytes(dest, content)
                    result.uploaded.append(UploadedFile(source_path=file_path, dest_path=dest, size=len(content)))
            except Exception as exc:  # noqa: BLE001 -- reported per path, as E2B's
                result.failed[str(source)] = str(exc)
        return result

    async def view_image(self, path: str) -> ImageContentBlock:
        if not path.lower().endswith((".png", ".jpg", ".jpeg")):
            raise ValueError(f"Unsupported image type for `{path}`. Only .png, .jpg, or .jpeg are allowed.")
        try:
            return ImageContentBlock(data=await self.read_file_bytes(path))
        except ValidationError as e:
            raise ValueError("You submitted a corrupt/unsupported image file.") from e
