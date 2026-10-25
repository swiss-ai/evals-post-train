"""Pod-per-sandbox Kubernetes backend for EvalScope's ms-enclave sandboxes.

EvalScope's agentic SWE-bench and SWE-bench_Pro run every sample in its own
container through ms-enclave's Docker manager. Our runners have no Docker
daemon (k8s Jobs are unprivileged pods, CSCS refuses privileged containers), so
install() swaps that manager for this one: each sandbox is a plain,
unprivileged pod running the same image -- the approach evals-svc's
harbor_envs/evals_k8s.py already takes for Terminal-Bench's trial containers.

What the Docker manager does and how it maps here:

- create_sandbox: a pod with the DockerSandboxConfig's image, working_dir,
  env_vars, cpu_limit and memory_limit (requests == limits), kept alive by a
  sleep loop; platform ("linux/amd64") becomes a kubernetes.io/arch node
  selector. Once Ready, "localhost" is made IPv4-only in /etc/hosts, as it
  effectively is under Docker (see _HOSTS_LOCALHOST_IPV4_ONLY). Bind-mount volumes ({host_dir: {"bind": path, "mode": ...}}, used
  by SWE-bench_Pro's evaluation step) are emulated: the host dir is copied in
  after the pod is Ready, and copied back out after every command for "rw"
  ones, so the caller reads its output files from the host dir as with Docker.
- execute_tool("shell_executor"): an exec of the same argv under GNU timeout,
  with the same ToolResult status/output/error shapes ms-enclave's own
  ShellExecutor returns, so EvalScope's exit-code parsing is unchanged.
- delete_sandbox: deletes the pod. Pods are also owned by the runner's own pod
  (EVALSCOPE_K8S_OWNER_POD_NAME/_UID), so Kubernetes garbage-collects them if
  the runner dies without cleaning up.

Sandbox pods never get a service-account token: the model under test runs
arbitrary commands in them.

Environment (all optional):
  EVALSCOPE_K8S_NAMESPACE          default: the in-cluster namespace, else the kubeconfig's
  EVALSCOPE_K8S_NODE_ARCH          node arch override (default: from the config's platform)
  EVALSCOPE_K8S_TOLERATIONS        "key=value,..." tolerated for NoSchedule and NoExecute
  EVALSCOPE_K8S_OWNER_POD_NAME/_UID  owner reference for garbage collection
  EVALSCOPE_K8S_IMAGE_PULL_SECRET  imagePullSecret name (DockerHub pull rate limits)
  EVALSCOPE_K8S_READY_TIMEOUT      seconds to wait for a pod to become Ready (default: 1800,
                                   SWE-bench images are several GB)
  EVALSCOPE_K8S_RUN_LABEL          value of the evalscope-run pod label (default: random)
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import shlex
import tarfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

_INCLUSTER_NS = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
_MANAGED_BY = "evals-post-train-evalscope"
# Stay under Linux's 128 KiB single-argument cap (MAX_ARG_STRLEN) when writing
# base64 chunks through `printf %s <chunk>`.
_B64_CHUNK = 100_000
# Exit codes `timeout` produces when it kills the command: GNU's 124 (and 137 after
# KILL), busybox's 143 (128 + TERM). Only counted as a timeout once the elapsed time
# reached the limit, so a command that itself exits 143 early is still an error.
_TIMEOUT_EXIT_CODES = (124, 137, 143)
# Waiting reasons that never resolve by themselves.
_FATAL_WAITING_REASONS = {"InvalidImageName", "ErrImageNeverPull", "CreateContainerConfigError"}
# Pods get an IPv6 link-local address on eth0, Docker containers (IPv6 off) do not, so
# getaddrinfo(AI_ADDRCONFIG) resolves "localhost" to ::1 in a pod but to 127.0.0.1 under
# Docker: a test server bound to 0.0.0.0 is then unreachable via "localhost" (SWE-bench
# Pro's NodeBB tests: ECONNREFUSED ::1). Dropping the ::1 localhost alias restores the
# Docker behaviour. /etc/hosts is a kubelet bind mount, so rewrite it in place (no sed -i).
_HOSTS_LOCALHOST_IPV4_ONLY = (
    "hosts=$(sed -E '/^::1[[:space:]]/s/([[:space:]])localhost([[:space:]]|$)/\\1/' /etc/hosts)"
    ' && printf "%s\\n" "$hosts" > /etc/hosts'
)


def k8s_memory(value: Union[str, int, float, None]) -> str:
    """Docker-style memory limit ("4g", "512m", bytes) -> Kubernetes quantity."""
    if value is None or value == "":
        return "4Gi"
    if isinstance(value, (int, float)):
        return str(int(value))
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?)i?b?\s*", str(value).lower())
    if not match:
        return str(value)
    number, unit = match.groups()
    return f"{number}{unit.upper()}i" if unit else number


def node_arch(platform: Optional[str]) -> Optional[str]:
    """ "linux/amd64" -> "amd64"; EVALSCOPE_K8S_NODE_ARCH wins."""
    override = os.environ.get("EVALSCOPE_K8S_NODE_ARCH")
    if override:
        return override
    if not platform:
        return None
    return platform.split("/")[1] if "/" in platform else platform


def timeout_argv(command: Union[str, List[str]], timeout: Optional[float]) -> List[str]:
    """The exec argv: the command under GNU timeout, as ms-enclave's Docker box runs it.

    A bare string is split without a shell, also as the Docker box does
    (EvalScope always passes an explicit [bash, -lc, ...] list anyway)."""
    argv = list(command) if isinstance(command, (list, tuple)) else shlex.split(command)
    if timeout is None or float(timeout) <= 0:
        return argv
    return ["timeout", "-s", "TERM", "-k", "5s", f"{float(timeout)}s", *argv]


def returncode_from_status(err_channel: Optional[str]) -> int:
    """Parse the exec ERROR_CHANNEL v1.Status JSON into an exit code."""
    if not err_channel:
        return 0
    try:
        status = json.loads(err_channel)
    except (json.JSONDecodeError, TypeError):
        return 1
    if status.get("status") == "Success":
        return 0
    for cause in (status.get("details") or {}).get("causes") or []:
        if cause.get("reason") == "ExitCode":
            try:
                return int(cause.get("message"))
            except (TypeError, ValueError):
                return 1
    return 1


def _cfg(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def pod_manifest(name: str, namespace: str, config: Any, run_label: str) -> Dict[str, Any]:
    """The sandbox pod for one ms-enclave DockerSandboxConfig (or its dict form)."""
    cpu = str(_cfg(config, "cpu_limit", 2.0))
    memory = k8s_memory(_cfg(config, "memory_limit", "4g"))
    resources = {"cpu": cpu, "memory": memory}
    container: Dict[str, Any] = {
        "name": "main",
        "image": _cfg(config, "image"),
        # No `sleep infinity`: not every image's sleep understands it.
        "command": ["sh", "-c", "trap 'exit 0' TERM; while :; do sleep 3600 & wait $!; done"],
        "env": [{"name": k, "value": str(v)} for k, v in (_cfg(config, "env_vars") or {}).items()],
        "resources": {"requests": dict(resources), "limits": dict(resources)},
        # Task images run as root and expect to; still no privilege escalation.
        "securityContext": {"allowPrivilegeEscalation": False},
    }
    working_dir = _cfg(config, "working_dir")
    if working_dir:
        container["workingDir"] = working_dir
    spec: Dict[str, Any] = {
        "restartPolicy": "Never",
        "automountServiceAccountToken": False,
        "terminationGracePeriodSeconds": 5,
        "containers": [container],
    }
    arch = node_arch(_cfg(config, "platform"))
    if arch:
        spec["nodeSelector"] = {"kubernetes.io/arch": arch}
    tolerations = [t.strip() for t in os.environ.get("EVALSCOPE_K8S_TOLERATIONS", "").split(",") if t.strip()]
    if tolerations:
        spec["tolerations"] = [
            {"key": k, "operator": "Equal", "value": v, "effect": effect}
            for k, v in (t.split("=", 1) for t in tolerations)
            for effect in ("NoSchedule", "NoExecute")
        ]
    pull_secret = os.environ.get("EVALSCOPE_K8S_IMAGE_PULL_SECRET")
    if pull_secret:
        spec["imagePullSecrets"] = [{"name": pull_secret}]
    metadata: Dict[str, Any] = {
        "name": name,
        "namespace": namespace,
        "labels": {"app.kubernetes.io/managed-by": _MANAGED_BY, "evalscope-run": run_label},
    }
    owner_name = os.environ.get("EVALSCOPE_K8S_OWNER_POD_NAME")
    owner_uid = os.environ.get("EVALSCOPE_K8S_OWNER_POD_UID")
    if owner_name and owner_uid:
        metadata["ownerReferences"] = [{"apiVersion": "v1", "kind": "Pod", "name": owner_name, "uid": owner_uid}]
    return {"apiVersion": "v1", "kind": "Pod", "metadata": metadata, "spec": spec}


def tar_dir_b64(source_dir: Union[str, Path]) -> str:
    """A gzipped tar of source_dir's contents, base64-encoded."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for item in sorted(Path(source_dir).iterdir()):
            tf.add(item, arcname=item.name)
    return base64.b64encode(buf.getvalue()).decode()


def untar_b64(data: str, target_dir: Union[str, Path]) -> None:
    """Inverse of `tar czf - . | base64` run in the pod (newlines tolerated)."""
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    raw = base64.b64decode("".join(data.split()))
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tf:
        tf.extractall(target_dir, filter="data")


def _load_kube_config() -> None:
    from kubernetes import config as k8s_config
    from kubernetes.config.config_exception import ConfigException

    try:
        k8s_config.load_incluster_config()
    except ConfigException:
        k8s_config.load_kube_config()


def _default_namespace() -> str:
    ns = os.environ.get("EVALSCOPE_K8S_NAMESPACE")
    if ns:
        return ns
    if _INCLUSTER_NS.is_file():
        ns = _INCLUSTER_NS.read_text().strip()
        if ns:
            return ns
    try:
        from kubernetes import config as k8s_config

        _, active = k8s_config.list_kube_config_contexts()
        ns = (active or {}).get("context", {}).get("namespace")
        if ns:
            return ns
    except Exception:
        pass
    return "default"


class _Sandbox:
    def __init__(self, pod: str, volumes: Dict[str, Dict[str, str]]):
        self.pod = pod
        self.volumes = volumes


class K8sPodSandboxManager:
    """Duck-types the parts of ms-enclave's SandboxManager EvalScope's SandboxService
    calls for per-sample sandboxes (no pools: neither benchmark uses one)."""

    def __init__(self, **_: Any) -> None:
        self._namespace = ""
        self._run_label = os.environ.get("EVALSCOPE_K8S_RUN_LABEL") or uuid.uuid4().hex[:8]
        self._ready_timeout = float(os.environ.get("EVALSCOPE_K8S_READY_TIMEOUT", "1800"))
        self._sandboxes: Dict[str, _Sandbox] = {}

    # --- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        await asyncio.to_thread(_load_kube_config)
        self._namespace = _default_namespace()

    async def stop(self) -> None:
        await self.cleanup_all_sandboxes()

    async def cleanup_all_sandboxes(self) -> None:
        for sandbox_id in list(self._sandboxes):
            await self.delete_sandbox(sandbox_id)

    @staticmethod
    def _core():
        # A fresh client per call: kubernetes.stream patches its ApiClient while an exec
        # runs, which breaks any concurrent call sharing it.
        from kubernetes import client

        return client.CoreV1Api(api_client=client.ApiClient())

    async def create_sandbox(self, sandbox_type: Any = None, config: Any = None, sandbox_id: Optional[str] = None) -> str:
        sandbox_id = sandbox_id or uuid.uuid4().hex[:12]
        pod = f"evalscope-{self._run_label}-{sandbox_id}"[:63].rstrip("-")
        manifest = pod_manifest(pod, self._namespace, config, self._run_label)
        await asyncio.to_thread(self._core().create_namespaced_pod, self._namespace, manifest)
        sandbox = _Sandbox(pod, dict(_cfg(config, "volumes") or {}))
        self._sandboxes[sandbox_id] = sandbox
        try:
            await self._wait_ready(pod)
            code, _, err = await self._exec(pod, ["sh", "-c", _HOSTS_LOCALHOST_IPV4_ONLY], timeout=60)
            if code != 0:
                from evalscope.utils.logger import get_logger

                get_logger().warning(
                    f"sandbox pod {pod}: could not make localhost IPv4-only in /etc/hosts "
                    f"(exit {code}: {err.strip()[:200]}); localhost resolves to ::1 there, unlike under Docker"
                )
            for host_dir, mount in sandbox.volumes.items():
                await self._upload_dir(pod, host_dir, mount["bind"])
        except BaseException:
            await self.delete_sandbox(sandbox_id)
            raise
        return sandbox_id

    async def delete_sandbox(self, sandbox_id: str) -> bool:
        sandbox = self._sandboxes.pop(sandbox_id, None)
        if sandbox is None:
            return False
        from kubernetes.client.rest import ApiException

        try:
            await asyncio.to_thread(
                self._core().delete_namespaced_pod, sandbox.pod, self._namespace, grace_period_seconds=0
            )
        except ApiException as exc:
            if exc.status != 404:
                raise
        return True

    async def _wait_ready(self, pod: str) -> None:
        deadline = time.monotonic() + self._ready_timeout
        last = ""
        while time.monotonic() < deadline:
            status = (await asyncio.to_thread(self._core().read_namespaced_pod, pod, self._namespace)).status
            conditions = {c.type: c.status for c in (status.conditions or [])}
            if conditions.get("Ready") == "True":
                return
            if status.phase in ("Failed", "Succeeded"):
                raise RuntimeError(f"sandbox pod {pod} entered phase {status.phase} before Ready")
            for cs in status.container_statuses or []:
                waiting = cs.state.waiting if cs.state else None
                if waiting and waiting.reason in _FATAL_WAITING_REASONS:
                    raise RuntimeError(f"sandbox pod {pod}: {waiting.reason}: {waiting.message}")
                if waiting:
                    last = f"{waiting.reason}: {waiting.message}"
            await asyncio.sleep(3)
        raise RuntimeError(f"sandbox pod {pod} not Ready after {self._ready_timeout:.0f}s ({last or 'no status'})")

    # --- exec -----------------------------------------------------------------

    def _exec_blocking(self, pod: str, argv: List[str], deadline_s: Optional[float]) -> tuple[int, str, str]:
        """Exec argv in the pod. A keepalive ping every 30 s keeps proxies from dropping
        long commands (an evaluation step can run for an hour); deadline_s is a hard
        backstop above the in-container GNU timeout."""
        from kubernetes.stream import stream
        from kubernetes.stream.ws_client import ERROR_CHANNEL

        resp = stream(
            self._core().connect_get_namespaced_pod_exec,
            pod,
            self._namespace,
            container="main",
            command=argv,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _preload_content=False,
        )
        stop = threading.Event()
        lock = threading.Lock()

        def keepalive() -> None:
            while not stop.wait(30):
                try:
                    with lock:
                        resp.sock.ping()
                except Exception:
                    return

        threading.Thread(target=keepalive, daemon=True).start()
        out: List[str] = []
        err: List[str] = []
        hard_deadline = time.monotonic() + deadline_s if deadline_s else None
        try:
            while resp.is_open():
                with lock:
                    resp.update(timeout=1)
                if resp.peek_stdout():
                    out.append(resp.read_stdout())
                if resp.peek_stderr():
                    err.append(resp.read_stderr())
                if hard_deadline and time.monotonic() > hard_deadline:
                    return 124, "".join(out), "".join(err)
            with lock:
                err_channel = resp.read_channel(ERROR_CHANNEL)
            return returncode_from_status(err_channel), "".join(out), "".join(err)
        finally:
            stop.set()
            with lock:
                try:
                    resp.close()
                except Exception:
                    pass

    async def _exec(self, pod: str, argv: List[str], timeout: Optional[float] = None) -> tuple[int, str, str]:
        deadline = float(timeout) + 30 if timeout else None
        return await asyncio.to_thread(self._exec_blocking, pod, timeout_argv(argv, timeout), deadline)

    async def _checked(self, pod: str, script: str, what: str) -> str:
        code, out, err = await self._exec(pod, ["sh", "-c", script], timeout=600)
        if code != 0:
            raise RuntimeError(f"{what} in sandbox pod {pod} failed (exit {code}): {err.strip()[:500]}")
        return out

    async def execute_tool(self, sandbox_id: str, tool_name: str, parameters: Dict[str, Any]) -> Any:
        from ms_enclave.sandbox.model import ExecutionStatus, ToolResult

        sandbox = self._sandboxes.get(sandbox_id)
        if sandbox is None:
            raise ValueError(f"Sandbox {sandbox_id} not found")
        if tool_name != "shell_executor":
            return ToolResult(
                tool_name=tool_name,
                status=ExecutionStatus.ERROR,
                output="",
                error=f"{tool_name} is not supported by the Kubernetes sandbox backend (only shell_executor)",
            )
        command = parameters.get("command")
        if not command or (isinstance(command, str) and not command.strip()):
            return ToolResult(tool_name=tool_name, status=ExecutionStatus.ERROR, output="", error="No command provided")
        timeout = parameters.get("timeout", 30)

        started = time.monotonic()
        try:
            code, out, err = await self._exec(sandbox.pod, command, timeout=timeout)
            await self._sync_back(sandbox)
        except Exception as exc:
            return ToolResult(
                tool_name=tool_name, status=ExecutionStatus.ERROR, output="", error=f"Execution failed: {exc}"
            )
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

    # --- file transfer (tar + base64 over exec) --------------------------------

    async def _upload_dir(self, pod: str, source_dir: Union[str, Path], target_dir: str) -> None:
        b64 = tar_dir_b64(source_dir)
        tmp = "/tmp/.evalscope-upload.b64"
        await self._checked(pod, f"mkdir -p {shlex.quote(target_dir)} && : > {tmp}", "upload: prepare")
        for offset in range(0, len(b64), _B64_CHUNK):
            chunk = b64[offset : offset + _B64_CHUNK]
            await self._checked(pod, f"printf %s {shlex.quote(chunk)} >> {tmp}", "upload: write chunk")
        await self._checked(
            pod, f"base64 -d {tmp} | tar xzf - -C {shlex.quote(target_dir)} && rm -f {tmp}", "upload: extract"
        )

    async def _sync_back(self, sandbox: _Sandbox) -> None:
        for host_dir, mount in sandbox.volumes.items():
            if mount.get("mode", "rw") != "rw":
                continue
            out = await self._checked(
                sandbox.pod, f"tar czf - -C {shlex.quote(mount['bind'])} . | base64", "download"
            )
            untar_b64(out, host_dir)

    async def put_dir(self, sandbox_id: str, source_dir: Union[str, Path], target_dir: str) -> bool:
        sandbox = self._sandboxes.get(sandbox_id)
        if sandbox is None:
            raise ValueError(f"Sandbox {sandbox_id} not found")
        await self._upload_dir(sandbox.pod, source_dir, target_dir)
        return True

    async def initialize_pool(self, *_: Any, **__: Any) -> Any:
        raise NotImplementedError("the Kubernetes sandbox backend has no pools (per-sample sandboxes only)")


def install() -> None:
    """Route EvalScope's Docker sandbox engine to K8sPodSandboxManager, process-wide.

    Every sandbox user -- the agent loop's EnclaveAgentEnvironment, SWE-bench_Pro's
    eval_instance, and swe_bench_scoring's -- gets its manager from
    SandboxService._construct_manager, so one patch covers all of them."""
    from evalscope.api.sandbox import SandboxEngine
    from evalscope.api.sandbox.service import SandboxService

    original = SandboxService._construct_manager
    if getattr(original, "_evals_k8s", False):
        return

    def _construct_manager(self, engine, manager_config):
        if engine is SandboxEngine.DOCKER:
            return K8sPodSandboxManager(**(manager_config or {}))
        return original(self, engine, manager_config)

    _construct_manager._evals_k8s = True
    SandboxService._construct_manager = _construct_manager
