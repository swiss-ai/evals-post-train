"""Reverse tunnel from an OpenSandbox microVM to the model server on this node (`--model-tunnel`).

An agent that runs inside the sandbox (OpenHands, mini-swe-agent, ...) calls the model from the
microVM, but Breithorn's sandboxes cannot open connections to Clariden. The node can reach the
sandbox through the OpenSandbox server's proxy, which carries WebSockets, so the node opens the
tunnel and the sandbox uses it:

    in the sandbox:  127.0.0.1:<pod_port>  ->  chisel server --reverse  (on <server_port>)
                         <= WebSocket opened by the node through the server's proxy =>
    on the node:     chisel client  ->  <target>  (the model server, host:port)

Adapted from the per-sandbox model tunnel of swiss-ai/rl-infra (PR 7). chisel
(https://github.com/jpillora/chisel) is not shipped: scripts/harbor/fetch_chisel.sh puts
chisel_<version>_linux_{amd64,arm64} in a directory, which the sandbox copy is uploaded from at every
start, so the task image needs no internet and no tools beyond sh and grep. Agents that run on the
node (terminus-2) call the model directly and need no tunnel.
"""

from __future__ import annotations

import asyncio
import os
import platform
import secrets
import shlex
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

CHISEL_VERSION = "1.12.0"
ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}
CONNECTED_MARKER = "Connected (Latency"
POD_BINARY = "/tmp/.chisel"

# The environment variables run.py sets for the environment (--model-tunnel).
TARGET_ENV = "HARBOR_MODEL_TUNNEL_TARGET"
CHISEL_DIR_ENV = "HARBOR_MODEL_TUNNEL_CHISEL_DIR"
POD_PORT_ENV = "HARBOR_MODEL_TUNNEL_POD_PORT"
DEFAULT_POD_PORT = 8000


@dataclass(frozen=True)
class ModelTunnelConfig:
    """target: host:port the node-side client dials (the model server). chisel_dir: where
    chisel_<version>_linux_{amd64,arm64} are. pod_port: loopback port in the sandbox that reaches the
    target. server_port: sandbox port of the chisel server."""

    target: str
    chisel_dir: str
    pod_port: int = DEFAULT_POD_PORT
    server_port: int = 8765
    connect_timeout_s: float = 60.0
    health_path: Optional[str] = "/health"
    max_retry_count: int = 10

    @classmethod
    def from_env(cls, env: Optional[dict[str, str]] = None) -> Optional["ModelTunnelConfig"]:
        env = os.environ if env is None else env
        if not env.get(TARGET_ENV):
            return None
        return cls(
            target=env[TARGET_ENV],
            chisel_dir=env[CHISEL_DIR_ENV],
            pod_port=int(env.get(POD_PORT_ENV) or DEFAULT_POD_PORT),
        )

    def binary(self, arch: str) -> Path:
        return Path(self.chisel_dir) / f"chisel_{CHISEL_VERSION}_linux_{arch}"


def server_start_script(cfg: ModelTunnelConfig, wait_s: float = 15.0) -> str:
    """sh: start the chisel server detached and return only once it listens, or fail with its log.

    The command must not return before the server has left the exec's process group (the exec
    service may signal that group when the command ends), and a server that dies at start (bad
    binary, port taken) should fail here, not as a client timeout later."""
    binary, log = shlex.quote(POD_BINARY), shlex.quote(POD_BINARY + ".log")
    tries = max(1, int(wait_s / 0.2))
    return (
        f"chmod +x {binary} || exit 1; (setsid nohup {binary} server --host 0.0.0.0 --port {cfg.server_port} "
        f"--reverse > {log} 2>&1 &); "
        f"i=0; while [ $i -lt {tries} ]; do "
        f"grep -q 'Listening on' {log} 2>/dev/null && exit 0; i=$((i+1)); sleep 0.2; done; "
        f"echo 'chisel server not listening after {wait_s:g}s:'; cat {log}; exit 1"
    )


def health_check_script(pod_port: int, path: str, attempts: int = 40) -> str:
    """bash: GET `path` on 127.0.0.1:pod_port through the tunnel (retried while the reverse listener
    comes up), print the status line, exit 0 on any answer."""
    request = f"GET {path} HTTP/1.0\\r\\nHost: 127.0.0.1\\r\\n\\r\\n"
    return (
        f"for i in $(seq 1 {attempts}); do "
        f'r=$( (exec 3<>/dev/tcp/127.0.0.1/{pod_port} && printf "{request}" >&3 && head -n1 <&3) 2>/dev/null ); '
        'if [ -n "$r" ]; then echo "$r"; exit 0; fi; sleep 0.5; done; '
        f"echo 'no answer on 127.0.0.1:{pod_port}'; exit 1"
    )


class ModelTunnel:
    """One tunnel for one sandbox: open() once it is running, close() before it is deleted.

    exec(command, env) -> (return_code, stdout), upload(source, target) and endpoint(port) ->
    (url, headers) are the sandbox's: the environment passes its own."""

    def __init__(
        self,
        cfg: ModelTunnelConfig,
        exec: Callable[..., Awaitable[tuple[int, str]]],
        upload: Callable[[Path, str], Awaitable[None]],
        endpoint: Callable[[int], Awaitable[tuple[str, dict[str, str]]]],
        logger: Any,
        label: str = "",
    ) -> None:
        self._cfg, self._exec, self._upload, self._endpoint = cfg, exec, upload, endpoint
        self._logger, self._label = logger, label
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader: Optional[asyncio.Task] = None
        self._connected = asyncio.Event()
        self._tail: deque[str] = deque(maxlen=40)

    async def open(self) -> None:
        """Both ends up, and a health check through the tunnel. On failure nothing keeps running on
        the node; the sandbox side goes with the sandbox."""
        t0 = time.monotonic()
        try:
            status = await self._open()
        except BaseException:
            await self.close()
            raise
        self._logger.info(
            "model_tunnel%s: sandbox 127.0.0.1:%s -> %s up in %.1fs (%s)",
            self._label, self._cfg.pod_port, self._cfg.target, time.monotonic() - t0, status,
        )

    async def _open(self) -> str:
        cfg = self._cfg
        node_arch = ARCH.get(platform.machine())
        code, out = await self._exec("uname -m", None)
        pod_arch = ARCH.get(out.strip())
        if code != 0 or node_arch is None or pod_arch is None:
            raise RuntimeError(f"model_tunnel: unsupported architecture (node {platform.machine()!r}, sandbox {out!r})")
        node_bin, pod_bin = cfg.binary(node_arch), cfg.binary(pod_arch)
        for path in {node_bin, pod_bin}:
            if not path.is_file():
                raise FileNotFoundError(f"model_tunnel: {path} not found (scripts/harbor/fetch_chisel.sh)")
        await self._upload(pod_bin, POD_BINARY)
        # One credential per sandbox: the endpoint is open to every holder of the server's API key. It
        # goes in AUTH (chisel reads it from the environment), not in argv.
        auth = f"harbor:{secrets.token_hex(16)}"
        code, out = await self._exec(server_start_script(cfg), {"AUTH": auth})
        if code != 0:
            raise RuntimeError(f"model_tunnel{self._label}: chisel server in the sandbox did not come up: {out!r}")
        url, headers = await self._endpoint(cfg.server_port)
        args = [str(node_bin), "client", "--max-retry-count", str(cfg.max_retry_count), "--max-retry-interval", "10s"]
        for key, value in headers.items():  # chisel takes handshake headers only as flags
            args += ["--header", f"{key}: {value}"]
        args += [url.rstrip("/"), f"R:127.0.0.1:{cfg.pod_port}:{cfg.target}"]
        self._proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "AUTH": auth},
            start_new_session=True,
        )
        self._reader = asyncio.create_task(self._read_client_output())
        exited = asyncio.ensure_future(self._proc.wait())
        connected = asyncio.ensure_future(self._connected.wait())
        try:
            await asyncio.wait({exited, connected}, timeout=cfg.connect_timeout_s, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for fut in (exited, connected):
                if not fut.done():
                    fut.cancel()
        if not self._connected.is_set():
            _, pod_log = await self._exec(f"tail -n 20 {shlex.quote(POD_BINARY + '.log')}", None)
            raise RuntimeError(
                f"model_tunnel{self._label}: chisel client did not connect within {cfg.connect_timeout_s:g}s; "
                f"client output: {list(self._tail)[-10:]}; sandbox side: {pod_log.strip()!r}"
            )
        if not cfg.health_path:
            return "no health check"
        code, out = await self._exec(f"bash -c {shlex.quote(health_check_script(cfg.pod_port, cfg.health_path))}", None)
        parts = out.strip().split()
        if code != 0 or len(parts) < 2 or not parts[1].startswith("2"):
            raise RuntimeError(
                f"model_tunnel{self._label}: GET {cfg.health_path} through the tunnel failed: rc={code} {out.strip()!r}"
            )
        return f"GET {cfg.health_path}: {out.strip()}"

    async def _read_client_output(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        async for raw in self._proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            self._tail.append(line)
            if CONNECTED_MARKER in line:
                self._connected.set()

    async def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        reader, self._reader = self._reader, None
        if reader is not None:
            try:
                await asyncio.wait_for(reader, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                reader.cancel()
