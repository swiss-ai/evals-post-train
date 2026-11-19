"""GAIA2's scenario containers as Kata microVMs, through an OpenSandbox server.

gaia2-runner (facebookresearch/meta-agents-research-environments, gaia2-cli/runner) starts one
container per scenario through a ContainerLauncher; its only implementation runs podman, which
Clariden nodes and our k8s jobs don't have. OpenSandboxLauncher implements the same interface on
an OpenSandbox server, as Harbor's runs do (scripts/harbor/opensandbox_environment.py): only
OPENSANDBOX_API_KEY, no Docker and no Kubernetes rights. install() puts it where gaia2-runner's
CLI creates its launchers, and running this module is gaia2-runner's CLI with it installed:

    python -m scripts.gaia2.opensandbox_launcher run-config --config <toml>

What podman does and how it maps here:

- launch: a sandbox from the image, kept alive by the SDK's default entrypoint (tail -f
  /dev/null); the scenario (podman's read-only bind mount at /var/gaia2/custom_scenario.json)
  and any seed directories are written in through the file API, then the image's own
  entrypoint (/opt/gaia2-init-entrypoint.sh) starts in the background as root, with the
  environment podman would have passed with -e. Created with secure_access (Breithorn's server
  requires it), the sandbox-run=<OPENSANDBOX_RUN_LABEL> metadata, vm.overcommit_memory=1 as for
  Harbor, and a network policy: deny, plus an allow rule for each of GAIA2_ALLOWED_HOSTS (the
  model's and the judge's hosts, set by run.py); GAIA2_ALLOWED_HOSTS unset is the open internet.
- The adapter port: the runner calls the in-container adapter at http://127.0.0.1:<port>
  (/status, /notify). Each sandbox gets a local HTTP forward on a free port that relays those
  requests to the adapter's port through the server's proxy, with the sandbox's access headers;
  get_host_adapter_port() returns it, and wait_for_adapter polls /health through it.
- exec: the argv in the sandbox, as root or the user named (a name is resolved to its uid); a
  non-zero exit raises CalledProcessError, as podman exec does under check=True. A command whose
  stream ended without a result raises too.
- copy_from: a tar of the path, read through the file API and unpacked as `podman cp` would
  (a file to a path or into an existing directory; a directory, or with a trailing "/." its
  contents).
- stop: saves the sandbox's diagnostics (OPENSANDBOX_LOG_DIR), kills it, closes the forward.
  run.py deletes the leftovers of its run label at the end.

The server is OPENSANDBOX_DOMAIN (default: Breithorn's Kata endpoint) with OPENSANDBOX_API_KEY.
"""

from __future__ import annotations

import http.server
import io
import logging
import os
import shlex
import socketserver
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from gaia2_runner.launcher import ContainerLauncher

from scripts.common.opensandbox_sandbox import (
    DEFAULT_DOMAIN,
    HEALTH_POLL_SEC,
    LOG_DIR_ENV,
    OVERCOMMIT_COMMAND,
    READY_TIMEOUT_SEC,
    REQUEST_TIMEOUT_SEC,
    RUN_LABEL_ENV,
    RUN_LABEL_KEY,
    joined_lines,
)

logger = logging.getLogger(__name__)

SCENARIO_PATH = "/var/gaia2/custom_scenario.json"
ENTRYPOINT = "/opt/gaia2-init-entrypoint.sh"
ENTRYPOINT_LOG = "/tmp/gaia2-entrypoint.out"
# Comma-separated hosts the sandbox may reach (run.py: the model's and the judge's); unset:
# everything.
ALLOWED_HOSTS_ENV = "GAIA2_ALLOWED_HOSTS"
# What a scenario container gets: OpenClaw's gateway (Node) and the adapter, the app daemon
# and the judge (Python). Overridable for heavier agents.
CPU_ENV, MEMORY_ENV = "GAIA2_SANDBOX_CPU", "GAIA2_SANDBOX_MEMORY"
DEFAULT_CPU, DEFAULT_MEMORY = "2", "4Gi"
# A scenario's own timeout ([run].timeout, or its duration + 60 s) is the real limit; the
# sandbox's lifetime is a backstop for a runner that died without stopping it.
SANDBOX_TIMEOUT_SEC = 4 * 3600
COMMAND_TIMEOUT_SEC = 600
# Request headers the forward does not pass on (hop-by-hop, or set by the client itself).
_HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "host", "content-length", "accept-encoding"}


def allowed_hosts() -> Optional[List[str]]:
    value = os.environ.get(ALLOWED_HOSTS_ENV)
    if value is None:
        return None
    return [host.strip() for host in value.split(",") if host.strip()]


def network_policy(hosts: Optional[List[str]]) -> Any:
    """Deny plus an allow rule per host, or the open internet for None."""
    from opensandbox.models.sandboxes import NetworkPolicy, NetworkRule

    if hosts is None:
        return NetworkPolicy(defaultAction="allow", egress=[])
    return NetworkPolicy(defaultAction="deny", egress=[NetworkRule(action="allow", target=host) for host in hosts])


def connection_config() -> Any:
    from opensandbox.config.connection_sync import ConnectionConfigSync

    # The runner is outside the cluster (a Clariden job, a laptop, a k8s job): sandboxes are
    # reached through the server.
    return ConnectionConfigSync(
        domain=os.environ.get("OPENSANDBOX_DOMAIN") or DEFAULT_DOMAIN,
        api_key=os.environ["OPENSANDBOX_API_KEY"],
        protocol="https",
        use_server_proxy=True,
        request_timeout=timedelta(seconds=REQUEST_TIMEOUT_SEC),
    )


def unpack_like_podman_cp(raw: bytes, src_path: str, dst_path: str) -> None:
    """Unpacks a tar of src_path (made with `tar -C <parent> <name>`, or of a directory's
    contents for a src_path ending in "/.") the way `podman cp <ctr>:<src> <dst>` lays it out."""
    contents_only = src_path.rstrip("/").endswith("/.")
    dst = Path(dst_path)
    with tempfile.TemporaryDirectory(prefix="gaia2-cp-", dir=dst.parent if dst.parent.is_dir() else None) as tmp:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tf:
            tf.extractall(tmp, filter="data")
        if contents_only:
            dst.mkdir(parents=True, exist_ok=True)
            for item in Path(tmp).iterdir():
                _move(item, dst / item.name)
            return
        (item,) = list(Path(tmp).iterdir())
        _move(item, dst / item.name if dst.is_dir() else dst)


def _move(src: Path, dst: Path) -> None:
    if src.is_dir() and dst.is_dir():
        for child in src.iterdir():
            _move(child, dst / child.name)
        return
    if dst.is_dir():
        raise IsADirectoryError(f"cannot overwrite directory {dst} with a file")
    os.replace(src, dst)


class AdapterForward:
    """A local HTTP server on 127.0.0.1:<free port> that relays each request to one sandbox port
    through the OpenSandbox server's proxy."""

    def __init__(self, base_url: str, headers: Mapping[str, str], client: Any = None):
        import httpx

        self.base_url = base_url.rstrip("/")
        self.headers = dict(headers)
        self._client = client or httpx.Client(timeout=60, follow_redirects=False)
        forward = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:
                logger.debug("adapter forward: " + fmt, *args)

            def _relay(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else None
                headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP_HEADERS}
                try:
                    resp = forward._client.request(
                        self.command, forward.base_url + self.path, content=body, headers={**headers, **forward.headers}
                    )
                    status, content, ctype = resp.status_code, resp.content, resp.headers.get("content-type")
                except Exception as exc:  # the server or the sandbox is unreachable
                    status, content, ctype = 502, f"adapter forward: {exc}".encode(), "text/plain"
                self.send_response(status)
                if ctype:
                    self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            do_GET = do_POST = do_PUT = do_DELETE = _relay

        class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True

        self._server = Server(("127.0.0.1", 0), Handler)
        self.port: int = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._client.close()


class _Box:
    def __init__(self, sandbox: Any, forward: Optional[AdapterForward]):
        self.sandbox = sandbox
        self.forward = forward
        self.uids: Dict[str, Tuple[int, int]] = {"root": (0, 0)}


class OpenSandboxLauncher(ContainerLauncher):
    """gaia2-runner's ContainerLauncher on an OpenSandbox server. One instance serves one runner,
    which runs one scenario at a time: get_host_adapter_port() is the latest sandbox's forward."""

    def __init__(
        self,
        runtime: str = "opensandbox",
        *,
        sandbox_factory: Optional[Callable[..., Any]] = None,
        forward_factory: Callable[..., AdapterForward] = AdapterForward,
    ) -> None:
        self.runtime = runtime
        self._create = sandbox_factory or self._sdk_create
        self._forward_factory = forward_factory
        self._boxes: Dict[str, _Box] = {}
        self._last_port: Optional[int] = None

    @staticmethod
    def _sdk_create(**kwargs: Any) -> Any:
        from opensandbox.sync import SandboxSync

        return SandboxSync.create(**kwargs)

    # --- lifecycle --------------------------------------------------------------------------

    def launch(
        self,
        image: str,
        scenario_json_path: str,
        *,
        env: dict[str, str] | None = None,
        network: str = "host",
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        extra_volumes: tuple[str, ...] | None = None,
        adapter_port: int | None = None,
        gateway_port: int | None = None,
        seed_dirs: Mapping[str, str] | None = None,
    ) -> str:
        from opensandbox.models.sandboxes import PlatformSpec

        if extra_volumes:
            raise ValueError("[agent].volumes are bind mounts of the local launcher; OpenSandbox has none")
        port = adapter_port or 8090
        container_env = self._container_env(image, env, provider, model, api_key, port, gateway_port)
        name = self._container_name(scenario_json_path)
        sandbox = self._create(
            image=image,
            timeout=timedelta(seconds=SANDBOX_TIMEOUT_SEC),
            ready_timeout=timedelta(seconds=READY_TIMEOUT_SEC),
            metadata={RUN_LABEL_KEY: os.environ.get(RUN_LABEL_ENV, "gaia2"), "gaia2-scenario": name[:63]},
            resource={
                "cpu": os.environ.get(CPU_ENV, DEFAULT_CPU),
                "memory": os.environ.get(MEMORY_ENV, DEFAULT_MEMORY),
            },
            platform=PlatformSpec(os="linux", arch="amd64"),
            network_policy=network_policy(allowed_hosts()),
            secure_access=True,
            connection_config=connection_config(),
            health_check_polling_interval=timedelta(seconds=HEALTH_POLL_SEC),
        )
        box = _Box(sandbox, None)
        self._boxes[sandbox.id] = box
        try:
            self._run(box, ["sh", "-c", OVERCOMMIT_COMMAND], check=True)
            self._write_file(box, SCENARIO_PATH, Path(scenario_json_path).read_bytes(), mode=644)
            for container_path, host_dir in (seed_dirs or {}).items():
                self._upload_dir(box, host_dir, container_path)
            self._start_entrypoint(box, container_env)
            endpoint = sandbox.get_endpoint(port)
            config = sandbox.connection_config
            box.forward = self._forward_factory(
                f"{config.protocol}://{endpoint.endpoint}", endpoint.build_request_headers(config)
            )
        except BaseException:
            self.stop(sandbox.id)
            raise
        self._last_port = box.forward.port
        logger.info("Started sandbox %s (%s) from %s; adapter at 127.0.0.1:%d", sandbox.id, name, image, self._last_port)
        return sandbox.id

    def _container_env(
        self,
        image: str,
        env: Mapping[str, str] | None,
        provider: str | None,
        model: str | None,
        api_key: str | None,
        adapter_port: int,
        gateway_port: int | None,
    ) -> Dict[str, str]:
        """What LocalLauncher passes with -e (no proxy relay or host CA bundle: the sandbox
        reaches its allowed hosts directly)."""
        out = {"GAIA2_ADAPTER_PORT": str(adapter_port)}
        if gateway_port is not None:
            out["OPENCLAW_GATEWAY_PORT"] = str(gateway_port)
            out["OPENCLAW_GATEWAY_URL"] = f"ws://127.0.0.1:{gateway_port}"
        env = dict(env or {})
        out.update(self._build_provider_env(image, provider=provider, model=model, api_key=api_key, env=env))
        for key in ("GAIA2_PROXY_RELAY_URL", "GAIA2_CA_BUNDLE"):
            env.pop(key, None)
        out.update({k: str(v) for k, v in env.items()})
        return out

    def _start_entrypoint(self, box: _Box, env: Dict[str, str]) -> None:
        from opensandbox.models.execd import RunCommandOpts

        # The environment goes in the request's envs, not the command line (which the server
        # keeps as the command's content): it holds the model's and the judge's keys.
        box.sandbox.commands.run(
            ["bash", "-c", f"exec bash {ENTRYPOINT} >{ENTRYPOINT_LOG} 2>&1"],
            opts=RunCommandOpts(background=True, uid=0, gid=0, working_directory="/", envs=env),
        )

    def stop(self, container_id: str) -> None:
        box = self._boxes.pop(container_id, None)
        if box is None:
            return
        try:
            if box.forward is not None:
                box.forward.close()
            _save_logs(box.sandbox)
            box.sandbox.kill()
        except Exception as exc:
            logger.warning("Could not kill sandbox %s: %s", container_id, exc)
        finally:
            try:
                box.sandbox.close()
            except Exception:
                pass
        logger.info("Removed sandbox %s", container_id)

    def get_host_adapter_port(self) -> int | None:
        return self._last_port

    def wait_for_adapter(self, container_id: str, port: int = 8090, timeout: int = 120, interval: float = 2.0) -> None:
        """Polls /health through the forward (which is what the runner then talks to) until the
        adapter reports connected."""
        import httpx

        box = self._box(container_id)
        url = f"http://127.0.0.1:{box.forward.port}/health"
        deadline = time.monotonic() + timeout
        last_error = ""
        while time.monotonic() < deadline:
            try:
                resp = httpx.get(url, timeout=10, trust_env=False)
                health = resp.json() if resp.status_code == 200 else {}
                if health.get("connected"):
                    logger.info("Adapter is ready in sandbox %s", container_id)
                    return
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as exc:
                last_error = str(exc)
            time.sleep(interval)
        try:
            tail = self.exec(container_id, ["tail", "-n", "40", ENTRYPOINT_LOG])
        except Exception:
            tail = ""
        raise TimeoutError(f"Adapter in sandbox {container_id} not ready after {timeout}s: {last_error}\n{tail}")

    # --- commands and files -------------------------------------------------------------------

    def exec(self, container_id: str, command: list[str], *, user: str | None = None) -> str:
        box = self._box(container_id)
        uid, gid = self._ids(box, user or "root")
        return self._run(box, command, uid=uid, gid=gid, check=True)

    def copy_from(self, container_id: str, src_path: str, dst_path: str) -> None:
        box = self._box(container_id)
        src = src_path.rstrip("/")
        if src.endswith("/."):
            tar_args = ["-C", src[:-2] or "/", "."]
        else:
            parent, name = os.path.split(src)
            tar_args = ["-C", parent or "/", name]
        archive = f"/tmp/.gaia2-copy-{uuid.uuid4().hex[:8]}.tgz"
        self._run(box, ["tar", "czf", archive, *tar_args], check=True)
        try:
            raw = box.sandbox.files.read_bytes(archive)
        finally:
            self._run(box, ["rm", "-f", archive])
        unpack_like_podman_cp(raw, src_path, dst_path)

    def _box(self, container_id: str) -> _Box:
        box = self._boxes.get(container_id)
        if box is None:
            raise RuntimeError(f"no running sandbox {container_id}")
        return box

    def _ids(self, box: _Box, user: str) -> Tuple[int, int]:
        """A user name or uid as (uid, gid) in the sandbox: commands there take numbers."""
        if user not in box.uids:
            if user.isdigit():
                box.uids[user] = (int(user), int(user))
            else:
                out = self._run(box, ["sh", "-c", f"id -u {shlex.quote(user)} && id -g {shlex.quote(user)}"], check=True)
                uid, gid = out.split()
                box.uids[user] = (int(uid), int(gid))
        return box.uids[user]

    def _run(
        self, box: _Box, argv: List[str], uid: int = 0, gid: Optional[int] = 0, check: bool = False
    ) -> str:
        from opensandbox.models.execd import RunCommandOpts

        execution = box.sandbox.commands.run(
            argv,
            opts=RunCommandOpts(
                uid=uid, gid=gid, working_directory="/", timeout=timedelta(seconds=COMMAND_TIMEOUT_SEC)
            ),
        )
        code = execution.exit_code
        stdout, stderr = joined_lines(execution.logs.stdout), joined_lines(execution.logs.stderr)
        if code is None:
            if execution.error is None:
                raise RuntimeError(f"sandbox {box.sandbox.id}: the command's stream ended without a result")
            code, stderr = 1, stderr or str(execution.error)
        if check and code != 0:
            raise subprocess.CalledProcessError(code, argv, output=stdout, stderr=stderr)
        return stdout

    def _write_file(self, box: _Box, path: str, data: bytes, mode: int = 644) -> None:
        from opensandbox.models.filesystem import WriteEntry

        self._run(box, ["mkdir", "-p", os.path.dirname(path)], check=True)
        box.sandbox.files.write_files([WriteEntry(path=path, data=data, mode=mode)])

    def _upload_dir(self, box: _Box, host_dir: str, container_path: str) -> None:
        from scripts.common.opensandbox_sandbox import tar_dir_bytes

        archive = f"/tmp/.gaia2-seed-{uuid.uuid4().hex[:8]}.tgz"
        self._write_file(box, archive, tar_dir_bytes(host_dir))
        self._run(
            box,
            ["sh", "-c", f"mkdir -p {shlex.quote(container_path)} && tar xzf {archive} -C {shlex.quote(container_path)} && rm -f {archive}"],
            check=True,
        )


def _save_logs(sandbox: Any) -> None:
    """The sandbox's diagnostics to $OPENSANDBOX_LOG_DIR/<id>.log before it is killed (as
    scripts/common/opensandbox_sandbox.save_sandbox_logs does for the async SDK)."""
    directory = os.environ.get(LOG_DIR_ENV)
    if not directory:
        return
    try:
        events = getattr(sandbox.get_diagnostic_events("all"), "content", None) or ""
        logs = getattr(sandbox.get_diagnostic_logs("all"), "content", None) or ""
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / f"{sandbox.id}.log").write_text(f"----- events -----\n{events}\n\n----- logs -----\n{logs}\n")
    except Exception as exc:  # noqa: BLE001 - saving logs must never stop the run
        logger.warning("Could not save the logs of sandbox %s: %s", getattr(sandbox, "id", "?"), exc)


def install() -> None:
    """gaia2-runner's CLI creates every launcher as LocalLauncher(runtime=...): make that ours."""
    import gaia2_runner.cli

    gaia2_runner.cli.LocalLauncher = OpenSandboxLauncher


def main() -> None:
    if not os.environ.get("OPENSANDBOX_API_KEY"):
        sys.exit("gaia2 on OpenSandbox needs OPENSANDBOX_API_KEY")
    install()
    import gaia2_runner.cli

    gaia2_runner.cli.main()


if __name__ == "__main__":
    main()
