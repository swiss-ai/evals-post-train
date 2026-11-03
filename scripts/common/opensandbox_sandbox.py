"""Sandbox-per-sample Kata microVMs through an OpenSandbox server, for the EvalScope runner.

EvalScope runs every sample in its own container through ms-enclave's Docker manager.
install() swaps that manager for OpenSandboxSandboxManager (`--sandbox opensandbox`), so each
sandbox is a Kata microVM on the OpenSandbox server instead: no Docker daemon and no Kubernetes
access, only OPENSANDBOX_API_KEY. scripts/harbor/opensandbox_environment.py does the same for
Harbor on top of Harbor's own OpenSandbox environment; delete_run_sandboxes() here is shared.

What the Docker manager does and how it maps here (as in scripts/common/k8s_sandbox.py):

- create_sandbox: a sandbox from the DockerSandboxConfig's image, with its env_vars, cpu_limit
  and memory_limit, on its platform ("linux/amd64"). Created with secure_access, which
  Breithorn's server requires, an explicit public network policy (a sandbox created without one
  has open internet anyway; SWE-bench's tests may install packages), and the sandbox-run=<run
  label> metadata. vm.overcommit_memory is set to 1: the microVM's RAM is only the sandbox's
  memory limit, so the kernel's default would refuse any single allocation above it, which a
  container on a large node gets. "localhost" is made IPv4-only in /etc/hosts, as under Docker. Bind-mount
  volumes ({host_dir: {"bind": path, "mode": ...}}, SWE-bench_Pro's evaluation step) are
  emulated: the host dir is copied in once the sandbox is ready, and copied back out after
  every command for "rw" ones.
- execute_tool("shell_executor"): the same argv under GNU timeout, in the config's working_dir,
  with the same ToolResult status/output/error shapes ms-enclave's ShellExecutor returns. A
  command whose stream ended without a result is an error, never exit code 0.
- delete_sandbox: kills the sandbox. A runner that dies leaves its sandboxes until their timeout
  (SANDBOX_TIMEOUT_SEC); the runner deletes leftovers of its run label at the end.

The server is OPENSANDBOX_DOMAIN (default: Breithorn's Kata endpoint; a gVisor one rejects
network policies) with OPENSANDBOX_API_KEY. OPENSANDBOX_RUN_LABEL sets the run label (default:
random).
"""

from __future__ import annotations

import asyncio
import io
import os
import shlex
import tarfile
import time
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from scripts.common.k8s_sandbox import (
    _HOSTS_LOCALHOST_IPV4_ONLY,
    _TIMEOUT_EXIT_CODES,
    _cfg,
    _logger,
    k8s_memory,
    timeout_argv,
    untar_bytes,
)

# Breithorn's Kata endpoint; the default (gVisor) one cannot enforce network policies.
DEFAULT_DOMAIN = "opensandbox-kata.breithorn.svc.cscs.ch"
RUN_LABEL_ENV = "OPENSANDBOX_RUN_LABEL"
RUN_LABEL_KEY = "sandbox-run"
# Creating a sandbox can wait on a pull of a several-GB image (the SDK's default request is 30 s).
REQUEST_TIMEOUT_SEC = 300
READY_TIMEOUT_SEC = 1800
# How long a sandbox lives at most: a SWE-bench agent loop (250 steps) plus its evaluation.
SANDBOX_TIMEOUT_SEC = 24 * 3600
# Readiness polls go through the server: the SDK's 200 ms default hammers it under concurrency.
HEALTH_POLL_SEC = 2
# A command's own timeout (GNU timeout in the sandbox) is the real limit; the request's is a
# backstop above it, and covers commands given without one.
REQUEST_SLACK_SEC = 60
DEFAULT_COMMAND_TIMEOUT_SEC = 3600
# Run as root in every new sandbox (Harbor's and EvalScope's): see the module docstring.
OVERCOMMIT_COMMAND = "echo 1 > /proc/sys/vm/overcommit_memory"


def domain() -> str:
    return os.environ.get("OPENSANDBOX_DOMAIN") or DEFAULT_DOMAIN


def connection_config(request_timeout_sec: float = REQUEST_TIMEOUT_SEC) -> Any:
    from opensandbox.config import ConnectionConfig

    # The SDK's default protocol is http; the server uses https. Clients outside the cluster (a
    # Clariden job, a laptop) reach sandboxes through the server.
    return ConnectionConfig(
        domain=domain(),
        api_key=os.environ["OPENSANDBOX_API_KEY"],
        protocol="https",
        use_server_proxy=True,
        request_timeout=timedelta(seconds=request_timeout_sec),
    )


def resources(config: Any) -> Dict[str, str]:
    """cpu_limit / memory_limit as the server's resource limits (2 CPUs and 4g by default, as
    ms-enclave's Docker box)."""
    return {"cpu": str(_cfg(config, "cpu_limit", 2.0)), "memory": k8s_memory(_cfg(config, "memory_limit", "4g"))}


def platform_spec(config: Any) -> Any:
    """"linux/amd64" -> PlatformSpec(os="linux", arch="amd64"); None when the config has none."""
    platform = _cfg(config, "platform")
    if not platform or "/" not in platform:
        return None
    from opensandbox.models.sandboxes import PlatformSpec

    os_name, arch = platform.split("/", 1)
    return PlatformSpec(os=os_name, arch=arch.split("/")[0])


def joined_lines(messages: List[Any]) -> str:
    """A command's output from its stream messages: the server sends one per line, without the
    newline (an empty line as "\\n")."""
    lines = [message.text.rstrip("\n") for message in messages]
    return "\n".join(lines) + "\n" if lines else ""


def tar_dir_bytes(source_dir: Union[str, Path]) -> bytes:
    """A gzipped tar of source_dir's contents."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for item in sorted(Path(source_dir).iterdir()):
            tf.add(item, arcname=item.name)
    return buf.getvalue()


class _Box:
    def __init__(self, sandbox: Any, config: Any):
        self.sandbox = sandbox
        self.working_dir: Optional[str] = _cfg(config, "working_dir")
        self.volumes: Dict[str, Dict[str, str]] = dict(_cfg(config, "volumes") or {})


class OpenSandboxSandboxManager:
    """Duck-types the parts of ms-enclave's SandboxManager EvalScope's SandboxService calls for
    per-sample sandboxes (no pools: neither benchmark uses one)."""

    def __init__(self, **_: Any) -> None:
        self._run_label = os.environ.get(RUN_LABEL_ENV) or f"evalscope-{uuid.uuid4().hex[:12]}"
        self._boxes: Dict[str, _Box] = {}

    # --- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        if not os.environ.get("OPENSANDBOX_API_KEY"):
            raise RuntimeError("--sandbox opensandbox needs OPENSANDBOX_API_KEY")

    async def stop(self) -> None:
        await self.cleanup_all_sandboxes()

    async def cleanup_all_sandboxes(self) -> None:
        for sandbox_id in list(self._boxes):
            await self.delete_sandbox(sandbox_id)

    async def create_sandbox(self, sandbox_type: Any = None, config: Any = None, sandbox_id: Optional[str] = None) -> str:
        from opensandbox import Sandbox
        from opensandbox.models.sandboxes import NetworkPolicy

        sandbox_id = sandbox_id or uuid.uuid4().hex[:12]
        sandbox = await Sandbox.create(
            _cfg(config, "image"),
            timeout=timedelta(seconds=SANDBOX_TIMEOUT_SEC),
            ready_timeout=timedelta(seconds=READY_TIMEOUT_SEC),
            env={k: str(v) for k, v in (_cfg(config, "env_vars") or {}).items()} or None,
            metadata={RUN_LABEL_KEY: self._run_label, "sandbox-id": sandbox_id},
            resource=resources(config),
            platform=platform_spec(config),
            network_policy=NetworkPolicy(defaultAction="allow", egress=[]),
            secure_access=True,
            connection_config=connection_config(),
            health_check_polling_interval=timedelta(seconds=HEALTH_POLL_SEC),
        )
        box = _Box(sandbox, config)
        self._boxes[sandbox_id] = box
        try:
            if box.working_dir:
                # A container runtime creates its working dir (Docker's -w, a pod's workingDir);
                # the server refuses a command whose working directory doesn't exist (SWE-bench
                # Pro's /sandbox).
                await self._checked(box, f"mkdir -p {shlex.quote(box.working_dir)}", "create working dir", cwd="/")
            await self._checked(box, OVERCOMMIT_COMMAND, "set vm.overcommit_memory", cwd="/", uid=0)
            code, _, err = await self._run(box, ["sh", "-c", _HOSTS_LOCALHOST_IPV4_ONLY], 60, cwd="/")
            if code != 0:
                _logger().warning(
                    f"sandbox {sandbox.id}: could not make localhost IPv4-only in /etc/hosts "
                    f"(exit {code}: {err.strip()[:200]}); localhost may resolve to ::1 there, unlike under Docker"
                )
            for host_dir, mount in box.volumes.items():
                await self._upload_dir(box, host_dir, mount["bind"])
        except BaseException:
            await self.delete_sandbox(sandbox_id)
            raise
        return sandbox_id

    async def delete_sandbox(self, sandbox_id: str) -> bool:
        box = self._boxes.pop(sandbox_id, None)
        if box is None:
            return False
        try:
            await box.sandbox.kill()
        finally:
            await box.sandbox.close()
        return True

    # --- commands -----------------------------------------------------------

    async def _run(
        self,
        box: _Box,
        command: Union[str, List[str]],
        timeout: Optional[float],
        cwd: Optional[str] = None,
        uid: Optional[int] = None,
    ) -> tuple[int, str, str]:
        """The command under GNU timeout (argv, no shell), in cwd (default: the sandbox's working_dir),
        as uid (default: the image's user)."""
        from opensandbox.models.execd import RunCommandOpts

        limit = float(timeout) if timeout and float(timeout) > 0 else DEFAULT_COMMAND_TIMEOUT_SEC
        execution = await box.sandbox.commands.run(
            timeout_argv(command, timeout),
            opts=RunCommandOpts(
                working_directory=cwd or box.working_dir,
                timeout=timedelta(seconds=limit + REQUEST_SLACK_SEC),
                uid=uid,
            ),
        )
        code = execution.exit_code
        stderr = joined_lines(execution.logs.stderr)
        if code is None:
            if execution.error is None:
                raise RuntimeError(f"sandbox {box.sandbox.id}: the command's stream ended without a result")
            code = 1
            stderr = stderr or f"{execution.error}"
        return code, joined_lines(execution.logs.stdout), stderr

    async def _checked(self, box: _Box, script: str, what: str, cwd: Optional[str] = None, uid: Optional[int] = None) -> str:
        code, out, err = await self._run(box, ["sh", "-c", script], 600, cwd=cwd, uid=uid)
        if code != 0:
            raise RuntimeError(f"{what} in sandbox {box.sandbox.id} failed (exit {code}): {err.strip()[:500]}")
        return out

    async def execute_tool(self, sandbox_id: str, tool_name: str, parameters: Dict[str, Any]) -> Any:
        from ms_enclave.sandbox.model import ExecutionStatus, ToolResult

        box = self._boxes.get(sandbox_id)
        if box is None:
            raise ValueError(f"Sandbox {sandbox_id} not found")
        if tool_name != "shell_executor":
            return ToolResult(
                tool_name=tool_name,
                status=ExecutionStatus.ERROR,
                output="",
                error=f"{tool_name} is not supported by the OpenSandbox backend (only shell_executor)",
            )
        command = parameters.get("command")
        if not command or (isinstance(command, str) and not command.strip()):
            return ToolResult(tool_name=tool_name, status=ExecutionStatus.ERROR, output="", error="No command provided")
        timeout = parameters.get("timeout", 30)

        started = time.monotonic()
        try:
            code, out, err = await self._run(box, command, timeout)
            await self._sync_back(box)
        except Exception as exc:
            return ToolResult(tool_name=tool_name, status=ExecutionStatus.ERROR, output="", error=f"Execution failed: {exc}")
        elapsed = time.monotonic() - started

        # Same shapes as ms-enclave's ShellExecutor on top of its Docker box.
        if timeout and code in _TIMEOUT_EXIT_CODES and elapsed + 0.05 >= float(timeout):
            return ToolResult(
                tool_name=tool_name,
                status=ExecutionStatus.TIMEOUT,
                output=out,
                error=err or f"Command timed out after {timeout} seconds",
                execution_time=elapsed,
            )
        if code == 0:
            return ToolResult(
                tool_name=tool_name, status=ExecutionStatus.SUCCESS, output=out, error=err or None, execution_time=elapsed
            )
        return ToolResult(
            tool_name=tool_name,
            status=ExecutionStatus.ERROR,
            output=out,
            error=err or f"Command failed with exit code {code}",
            execution_time=elapsed,
        )

    # --- file transfer (a tar through the server's file API) -------------------

    async def _upload_dir(self, box: _Box, source_dir: Union[str, Path], target_dir: str) -> None:
        from opensandbox.models.filesystem import WriteEntry

        archive = f"/tmp/.sandbox-upload-{uuid.uuid4().hex[:8]}.tgz"
        await box.sandbox.files.write_files([WriteEntry(path=archive, data=tar_dir_bytes(source_dir), mode=644)])
        await self._checked(
            box,
            f"mkdir -p {shlex.quote(target_dir)} && tar xzf {archive} -C {shlex.quote(target_dir)} && rm -f {archive}",
            "upload: extract",
        )

    async def _download_dir(self, box: _Box, source_dir: str, host_dir: Union[str, Path]) -> None:
        archive = f"/tmp/.sandbox-download-{uuid.uuid4().hex[:8]}.tgz"
        # GNU tar exits 1 when a file changed while it was read (a test process still writing a
        # log): the archive is still complete up to that point.
        await self._checked(box, f"tar czf {archive} -C {shlex.quote(source_dir)} .; [ $? -le 1 ]", "download: pack")
        raw = await box.sandbox.files.read_bytes(archive)
        await self._checked(box, f"rm -f {archive}", "download: clean up")
        untar_bytes(raw, host_dir)

    async def _sync_back(self, box: _Box) -> None:
        for host_dir, mount in box.volumes.items():
            if mount.get("mode", "rw") == "rw":
                await self._download_dir(box, mount["bind"], host_dir)

    async def put_dir(self, sandbox_id: str, source_dir: Union[str, Path], target_dir: str) -> bool:
        box = self._boxes.get(sandbox_id)
        if box is None:
            raise ValueError(f"Sandbox {sandbox_id} not found")
        await self._upload_dir(box, source_dir, target_dir)
        return True

    async def initialize_pool(self, *_: Any, **__: Any) -> Any:
        raise NotImplementedError("the OpenSandbox backend has no pools (per-sample sandboxes only)")


def install() -> None:
    """Route EvalScope's Docker sandbox engine to OpenSandboxSandboxManager, process-wide (the
    same single patch point as k8s_sandbox.install())."""
    from evalscope.api.sandbox import SandboxEngine
    from evalscope.api.sandbox.service import SandboxService

    original = SandboxService._construct_manager
    if getattr(original, "_evals_opensandbox", False):
        return

    def _construct_manager(self, engine, manager_config):
        if engine is SandboxEngine.DOCKER:
            return OpenSandboxSandboxManager(**(manager_config or {}))
        return original(self, engine, manager_config)

    _construct_manager._evals_opensandbox = True
    SandboxService._construct_manager = _construct_manager


async def replace_egress_policy(sandbox: Any, policy: Any) -> None:
    """Replaces a running sandbox's whole egress policy (an SDK NetworkPolicy) at its sidecar,
    then reads it back: a policy that did not take effect raises rather than leaving the network
    as it was. Shared by Harbor (a task set up with network, then cut off) and Inspect
    (network_mode: none after a Dockerfile replay)."""
    import httpx
    from opensandbox.constants import DEFAULT_EGRESS_PORT

    wanted = policy.model_dump(by_alias=True, exclude_none=True)
    config = sandbox.connection_config
    endpoint = await sandbox.get_endpoint(DEFAULT_EGRESS_PORT)
    async with httpx.AsyncClient(
        headers=endpoint.build_request_headers(config), timeout=60, transport=config.transport
    ) as client:
        response = await client.put(f"{config.protocol}://{endpoint.endpoint}/policy", json=wanted)
    if response.is_error:
        raise RuntimeError(f"sandbox {sandbox.id}: replacing the egress policy failed: HTTP {response.status_code} {response.text[:300]}")
    applied = (await sandbox.get_egress_policy()).model_dump(by_alias=True, exclude_none=True)
    if applied != wanted:
        raise RuntimeError(f"sandbox {sandbox.id}: egress policy is {applied}, not {wanted}")


async def delete_run_sandboxes(run_label: str, server: Optional[str] = None, api_key: Optional[str] = None) -> int:
    """Kills every sandbox of the run; returns how many. Lists all pages before the first kill,
    which would shift them."""
    from opensandbox.config import ConnectionConfig
    from opensandbox.manager import SandboxManager
    from opensandbox.models.sandboxes import SandboxFilter

    config = ConnectionConfig(
        domain=server or domain(),
        api_key=api_key or os.environ["OPENSANDBOX_API_KEY"],
        protocol="https",
        request_timeout=timedelta(seconds=60),
    )
    manager = await SandboxManager.create(config)
    try:
        live: List[str] = []
        page_number = 1
        while True:
            page = await manager.list_sandbox_infos(
                SandboxFilter(metadata={RUN_LABEL_KEY: run_label}, page_size=100, page=page_number)
            )
            live += [info.id for info in page.sandbox_infos if info.status.state not in ("Failed", "Terminated")]
            if not page.pagination.has_next_page:
                break
            page_number += 1
        await asyncio.gather(*(manager.kill_sandbox(sandbox_id) for sandbox_id in live))
        return len(live)
    finally:
        await manager.close()
