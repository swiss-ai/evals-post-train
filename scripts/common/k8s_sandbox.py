"""Pod-per-sandbox Kubernetes backend, shared by the EvalScope and Harbor runners.

Both run every sample/trial in its own container, normally through a Docker daemon. Our
runners have none (k8s Jobs are unprivileged pods, CSCS refuses privileged containers, Clariden
has no Docker), so each sandbox is a plain, unprivileged pod running the same image.

- EvalScope: install() swaps ms-enclave's Docker manager for K8sPodSandboxManager.
- Harbor: scripts/harbor/k8s_environment.py is a Harbor environment built on
  K8sPodSandboxManager's pod, exec and file-transfer primitives.

What the Docker manager does and how it maps here:

- create_sandbox: a pod with the DockerSandboxConfig's image, working_dir,
  env_vars, cpu_limit and memory_limit (requests == limits), kept alive by a
  sleep loop (SANDBOX_K8S_MEMORY_REQUEST/_CPU_REQUEST lower the requests); platform ("linux/amd64") becomes a kubernetes.io/arch node
  selector. Once Ready, "localhost" is made IPv4-only in /etc/hosts, as it
  effectively is under Docker (see _HOSTS_LOCALHOST_IPV4_ONLY). Bind-mount volumes ({host_dir: {"bind": path, "mode": ...}}, used
  by SWE-bench_Pro's evaluation step) are emulated: the host dir is copied in
  after the pod is Ready, and copied back out after every command for "rw"
  ones, so the caller reads its output files from the host dir as with Docker.
- execute_tool("shell_executor"): the same argv under GNU timeout, run detached
  in the pod (output and exit code to files, waited for by short execs, so a
  dropped exec stream is retried instead of misread as success), with the same ToolResult status/output/error shapes ms-enclave's own
  ShellExecutor returns, so EvalScope's exit-code parsing is unchanged.
- delete_sandbox: deletes the pod. Pods are also owned by the runner's own pod
  (SANDBOX_K8S_OWNER_POD_NAME/_UID), so Kubernetes garbage-collects them if
  the runner dies without cleaning up.

Sandbox pods never get a service-account token: the model under test runs
arbitrary commands in them.

Environment (all optional; each SANDBOX_K8S_* is also read under its old name EVALSCOPE_K8S_*):
  SANDBOX_K8S_NAMESPACE          default: the in-cluster namespace, else the kubeconfig's
  SANDBOX_K8S_NODE_ARCH          node arch override (default: from the config's platform)
  SANDBOX_K8S_TOLERATIONS        "key=value,..." tolerated for NoSchedule and NoExecute
  SANDBOX_K8S_OWNER_POD_NAME/_UID  owner reference for garbage collection
  SANDBOX_K8S_IMAGE_PULL_SECRET  imagePullSecret name (DockerHub pull rate limits)
  SANDBOX_K8S_READY_TIMEOUT      seconds to wait for a pod to become Ready (default: 1800,
                                   SWE-bench images are several GB)
  SANDBOX_K8S_RUN_LABEL          value of the sandbox-run pod label (default: random)
  SANDBOX_K8S_MEMORY_REQUEST     memory request below the limit, e.g. "4g" (default: the limit)
  SANDBOX_K8S_CPU_REQUEST        CPU request below the limit, e.g. "1" (default: the limit)
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

def setting(name: str, default: Optional[str] = None) -> Optional[str]:
    """SANDBOX_K8S_<name>, else its old name EVALSCOPE_K8S_<name> (from when only the EvalScope
    runner used this module), else default."""
    return os.environ.get(f"SANDBOX_K8S_{name}") or os.environ.get(f"EVALSCOPE_K8S_{name}") or default


_INCLUSTER_NS = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
_MANAGED_BY = "evals-post-train-sandbox"
# Stay under Linux's 128 KiB single-argument cap (MAX_ARG_STRLEN) when writing
# base64 chunks through `printf %s <chunk>`.
_B64_CHUNK = 100_000
# Upload chunks written at once: one exec per chunk, so a large upload (a build stage's
# output, tens of MB) is mostly round trips.
_UPLOAD_PARALLEL = 8
# Downloads are read back in pieces of this many raw bytes: one exec streaming tens of
# MB of output loses frames now and then (silently, with a Success status), while every
# read up to 32 MB arrived intact.
_DOWNLOAD_CHUNK = 8 * 1024 * 1024
_DOWNLOAD_ATTEMPTS = 3
# Exit codes `timeout` produces when it kills the command: GNU's 124 (and 137 after
# KILL), busybox's 143 (128 + TERM). Only counted as a timeout once the elapsed time
# reached the limit, so a command that itself exits 143 early is still an error.
_TIMEOUT_EXIT_CODES = (124, 137, 143)
# A command runs detached in the pod and is waited for by short execs of at most this
# many seconds: a dropped exec stream (API server restart, laptop network) can end
# with a Success status while the command still runs, so no stream may be load-bearing.
_WAIT_SLICE_S = 60
# Pause before retrying a wait exec that failed, e.g. while the API server is down.
_WAIT_RETRY_S = 5
# How long a failed exec waits for the kubelet to report a killed container.
_TERMINATION_POLLS = 10
_TERMINATION_POLL_INTERVAL = 3
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


def quantity_value(quantity: str) -> float:
    """A Kubernetes quantity ("4Gi", "512Mi", "1500m", "2") as a number, for comparisons."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([a-zA-Z]*)", str(quantity).strip())
    if not match:
        raise ValueError(f"not a Kubernetes quantity: {quantity!r}")
    number, unit = float(match.group(1)), match.group(2)
    factors = {"": 1, "m": 1e-3, "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40}
    if unit not in factors:
        raise ValueError(f"unknown unit in Kubernetes quantity: {quantity!r}")
    return number * factors[unit]


def node_arch(platform: Optional[str]) -> Optional[str]:
    """ "linux/amd64" -> "amd64"; SANDBOX_K8S_NODE_ARCH wins."""
    override = setting("NODE_ARCH")
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


def exec_failure_message(err_channel: Optional[str]) -> str:
    """The Status message when the exec itself failed (no command exit code), e.g.
    "OCI runtime exec failed" once the container is gone; "" otherwise."""
    try:
        status = json.loads(err_channel or "")
    except (json.JSONDecodeError, TypeError):
        return (err_channel or "").strip()
    if not isinstance(status, dict) or status.get("status") == "Success":
        return ""
    causes = (status.get("details") or {}).get("causes") or []
    if any(cause.get("reason") == "ExitCode" for cause in causes):
        return ""
    return str(status.get("message") or "")


def describe_termination(pod_status: Any) -> str:
    """ "container main terminated: OOMKilled (exit 137)" for a dead sandbox container, else ""."""
    for cs in getattr(pod_status, "container_statuses", None) or []:
        terminated = cs.state.terminated if cs.state else None
        if cs.name == "main" and terminated:
            return f"container main terminated: {terminated.reason or 'unknown reason'} (exit {terminated.exit_code})"
    return ""


def _logger() -> Any:
    """EvalScope's logger when this runs under EvalScope, else the standard one (Harbor)."""
    try:
        from evalscope.utils.logger import get_logger
    except ImportError:
        import logging

        return logging.getLogger(__name__)
    return get_logger()


def _cfg(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def pod_manifest(name: str, namespace: str, config: Any, run_label: str) -> Dict[str, Any]:
    """The sandbox pod for one ms-enclave DockerSandboxConfig (or its dict form)."""
    cpu = str(_cfg(config, "cpu_limit", 2.0))
    memory = k8s_memory(_cfg(config, "memory_limit", "4g"))
    resources = {"cpu": cpu, "memory": memory}
    # Lower requests let a sandbox schedule on a node whose memory/CPU is mostly reserved (not
    # used) and still grow to the limit, at the risk of node-pressure eviction (memory) or
    # throttling (CPU). Capped at the limit: a request above it is rejected by the API server.
    requests = dict(resources)
    memory_request = setting("MEMORY_REQUEST")
    if memory_request:
        requests["memory"] = min(k8s_memory(memory_request), memory, key=quantity_value)
    cpu_request = setting("CPU_REQUEST")
    if cpu_request:
        requests["cpu"] = min(str(cpu_request), cpu, key=quantity_value)
    container: Dict[str, Any] = {
        "name": "main",
        "image": _cfg(config, "image"),
        # No `sleep infinity`: not every image's sleep understands it.
        "command": ["sh", "-c", "trap 'exit 0' TERM; while :; do sleep 3600 & wait $!; done"],
        "env": [{"name": k, "value": str(v)} for k, v in (_cfg(config, "env_vars") or {}).items()],
        "resources": {"requests": requests, "limits": dict(resources)},
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
    tolerations = [t.strip() for t in setting("TOLERATIONS", "").split(",") if t.strip()]
    if tolerations:
        spec["tolerations"] = [
            {"key": k, "operator": "Equal", "value": v, "effect": effect}
            for k, v in (t.split("=", 1) for t in tolerations)
            for effect in ("NoSchedule", "NoExecute")
        ]
    pull_secret = setting("IMAGE_PULL_SECRET")
    if pull_secret:
        spec["imagePullSecrets"] = [{"name": pull_secret}]
    metadata: Dict[str, Any] = {
        "name": name,
        "namespace": namespace,
        # sandbox-pod selects exactly this pod, e.g. for a per-sandbox NetworkPolicy.
        "labels": {"app.kubernetes.io/managed-by": _MANAGED_BY, "sandbox-run": run_label, "sandbox-pod": name},
    }
    owner_name = setting("OWNER_POD_NAME")
    owner_uid = setting("OWNER_POD_UID")
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


def b64_bytes(data: str) -> bytes:
    """Decode `base64` output from the pod (its line wrapping tolerated)."""
    return base64.b64decode("".join(data.split()))


def untar_bytes(raw: bytes, target_dir: Union[str, Path]) -> None:
    """Extract a (gzipped) tar archive into target_dir."""
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tf:
        tf.extractall(target_dir, filter="data")


def untar_b64(data: str, target_dir: Union[str, Path]) -> None:
    """Inverse of `tar czf - . | base64` run in the pod (newlines tolerated)."""
    untar_bytes(b64_bytes(data), target_dir)


def _load_kube_config() -> None:
    from kubernetes import config as k8s_config
    from kubernetes.config.config_exception import ConfigException

    try:
        k8s_config.load_incluster_config()
    except ConfigException:
        k8s_config.load_kube_config()


def _default_namespace() -> str:
    ns = setting("NAMESPACE")
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
        self._run_label = setting("RUN_LABEL") or uuid.uuid4().hex[:8]
        self._ready_timeout = float(setting("READY_TIMEOUT", "1800"))
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
        pod = f"sandbox-{self._run_label}-{sandbox_id}"[:63].rstrip("-")
        manifest = pod_manifest(pod, self._namespace, config, self._run_label)
        await asyncio.to_thread(self._core().create_namespaced_pod, self._namespace, manifest)
        sandbox = _Sandbox(pod, dict(_cfg(config, "volumes") or {}))
        self._sandboxes[sandbox_id] = sandbox
        try:
            await self._wait_ready(pod)
            code, _, err = await self._exec(pod, ["sh", "-c", _HOSTS_LOCALHOST_IPV4_ONLY], timeout=60)
            if code != 0:
                _logger().warning(
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
                # The last update() before the close can buffer output that the loop
                # never read: drain it, or large outputs come back silently truncated.
                out.append(resp.read_stdout(timeout=0) or "")
                err.append(resp.read_stderr(timeout=0) or "")
                err_channel = resp.read_channel(ERROR_CHANNEL)
            stderr = "".join(err) or exec_failure_message(err_channel)
            return returncode_from_status(err_channel), "".join(out), stderr
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

    async def _termination(self, pod: str, polls: int = _TERMINATION_POLLS) -> str:
        # The kubelet reports a killed container a few seconds after its execs fail.
        for attempt in range(polls):
            try:
                pod_obj = await asyncio.to_thread(self._core().read_namespaced_pod, pod, self._namespace)
            except Exception:
                return ""
            termination = describe_termination(pod_obj.status)
            if termination or attempt == polls - 1:
                return termination
            await asyncio.sleep(_TERMINATION_POLL_INTERVAL)
        return ""

    async def _checked(self, pod: str, script: str, what: str) -> str:
        code, out, err = await self._exec(pod, ["sh", "-c", script], timeout=600)
        if code != 0:
            raise RuntimeError(f"{what} in sandbox pod {pod} failed (exit {code}): {err.strip()[:500]}")
        return out

    async def _run_detached(self, pod: str, command: Union[str, List[str]], timeout: Optional[float]) -> tuple[int, str, str]:
        """Run command (under GNU timeout) detached in the pod and wait for its exit code
        file, so a dropped exec stream costs a retry instead of a wrong result."""
        job = f"/tmp/.sandbox-exec-{uuid.uuid4().hex[:12]}"
        argv = timeout_argv(list(command) if isinstance(command, (list, tuple)) else shlex.split(command), timeout)
        # mkdir doubles as the guard that a retried start never runs the command twice.
        start = (
            f"mkdir {job} || exit 0; ( {shlex.join(argv)} > {job}/out 2> {job}/err < /dev/null; "
            f"echo $? > {job}/rc.tmp && mv {job}/rc.tmp {job}/rc ) > /dev/null 2>&1 &"
        )
        wait = (
            f"i=0; while [ ! -f {job}/rc ] && [ $i -lt {_WAIT_SLICE_S * 10} ]; do "
            f"sleep 0.1 2>/dev/null || sleep 1; i=$((i+1)); done; cat {job}/rc 2>/dev/null"
        )
        deadline = time.monotonic() + float(timeout) + 60 if timeout else None
        await self._retrying(pod, start, "start command", deadline)
        while True:
            rc = (await self._retrying(pod, wait, "wait for command", deadline)).strip()
            if rc.isdigit():
                break
            if deadline and time.monotonic() > deadline:
                raise RuntimeError(f"command in sandbox pod {pod} did not finish within {timeout}s plus a minute")
        out = (await self._read_file(pod, f"{job}/out")).decode("utf-8", errors="replace")
        err = (await self._read_file(pod, f"{job}/err")).decode("utf-8", errors="replace")
        await self._checked(pod, f"rm -rf {job}", "clean up command")
        return int(rc), out, err

    async def _retrying(self, pod: str, script: str, what: str, deadline: Optional[float]) -> str:
        """_checked, retried while the exec fails and the container is still alive."""
        while True:
            try:
                code, out, err = await self._exec(pod, ["sh", "-c", script], timeout=_WAIT_SLICE_S + 30)
                if code == 0:
                    return out
                problem = f"exit {code}: {err.strip()[:300]}"
            except Exception as exc:
                problem = str(exc)[:300]
            termination = await self._termination(pod, polls=1)
            if termination:
                raise RuntimeError(f"{what} in sandbox pod {pod}: {termination}")
            if deadline and time.monotonic() > deadline:
                raise RuntimeError(f"{what} in sandbox pod {pod} kept failing ({problem})")
            await asyncio.sleep(_WAIT_RETRY_S)

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
            code, out, err = await self._run_detached(sandbox.pod, command, timeout)
            await self._sync_back(sandbox)
        except Exception as exc:
            # A killed container (OOMKilled at the memory limit) fails every later exec
            # with a bare runtime error: name the actual cause.
            termination = await self._termination(sandbox.pod)
            return ToolResult(
                tool_name=tool_name,
                status=ExecutionStatus.ERROR,
                output="",
                error=f"Execution failed: {exc}" + (f" ({termination})" if termination else ""),
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
        await self._upload_tar_b64(pod, tar_dir_b64(source_dir), target_dir)

    async def _upload_tar_b64(self, pod: str, b64: str, target_dir: str) -> None:
        """Extract a base64-encoded gzipped tar into target_dir in the pod: written as
        numbered chunk files, several at a time, then joined."""
        tmp = f"/tmp/.sandbox-upload-{uuid.uuid4().hex[:8]}"
        await self._checked(pod, f"mkdir -p {tmp} {shlex.quote(target_dir)}", "upload: prepare")
        gate = asyncio.Semaphore(_UPLOAD_PARALLEL)

        async def write(index: int, offset: int) -> None:
            async with gate:
                chunk = b64[offset : offset + _B64_CHUNK]
                await self._checked(pod, f"printf %s {shlex.quote(chunk)} > {tmp}/{index:08d}", "upload: write chunk")

        await asyncio.gather(*(write(i, offset) for i, offset in enumerate(range(0, len(b64), _B64_CHUNK))))
        await self._checked(
            pod, f"cat {tmp}/* | base64 -d | tar xzf - -C {shlex.quote(target_dir)} && rm -rf {tmp}", "upload: extract"
        )

    async def _download_dir(self, pod: str, source_dir: str, host_dir: Union[str, Path]) -> None:
        tmp = "/tmp/.sandbox-download.tgz"
        # GNU tar exits 1 when a file changed while it was read (a test process still
        # writing a log): the archive is still complete up to that point.
        await self._checked(pod, f"tar czf {tmp} -C {shlex.quote(source_dir)} .; [ $? -le 1 ]", "download: pack")
        raw = await self._read_file(pod, tmp)
        await self._checked(pod, f"rm -f {tmp}", "download: clean up")
        untar_bytes(raw, host_dir)

    async def _read_file(self, pod: str, path: str) -> bytes:
        """A file's bytes, read in verified chunks (see _DOWNLOAD_CHUNK)."""
        size = int((await self._checked(pod, f"wc -c < {path}", "download: size")).strip())
        raw = bytearray()
        for index, offset in enumerate(range(0, size, _DOWNLOAD_CHUNK)):
            expected = min(_DOWNLOAD_CHUNK, size - offset)
            for _ in range(_DOWNLOAD_ATTEMPTS):
                chunk = b64_bytes(
                    await self._checked(
                        pod,
                        f"dd if={path} bs={_DOWNLOAD_CHUNK} skip={index} count=1 2>/dev/null | base64",
                        "download: read chunk",
                    )
                )
                if len(chunk) == expected:
                    break
            else:
                raise RuntimeError(
                    f"download from sandbox pod {pod}: chunk {index} came back as {len(chunk)} of "
                    f"{expected} bytes in {_DOWNLOAD_ATTEMPTS} attempts"
                )
            raw += chunk
        return bytes(raw)

    async def _sync_back(self, sandbox: _Sandbox) -> None:
        for host_dir, mount in sandbox.volumes.items():
            if mount.get("mode", "rw") != "rw":
                continue
            await self._download_dir(sandbox.pod, mount["bind"], host_dir)

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
