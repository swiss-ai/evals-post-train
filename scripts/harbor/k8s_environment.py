"""Pod-per-trial Kubernetes environment for Harbor (`harbor run --env <this module>:K8sPodEnvironment`).

Harbor's built-in Kubernetes providers are cloud-specific (gke, ack, openshift); this one
only needs a kubeconfig (or an in-cluster service account) with create/get/delete on pods
and networkpolicies and create on pods/exec, e.g. Breithorn's rob-poc namespace. The pod
plumbing is scripts/common/k8s_sandbox.py's K8sPodSandboxManager, shared with the
EvalScope runner: an unprivileged pod per sandbox, commands run detached in the pod and
waited for by short execs (a dropped API stream is retried, not misread), files moved as
verified tar chunks, and a dead container reported with its termination reason (OOMKilled).
Its SANDBOX_K8S_* settings (namespace, node arch, tolerations, memory request, run label,
pull secret, ready timeout) apply here too.

The trial's image: [environment].docker_image runs as is (Terminal-Bench, DeepSWE); otherwise
the task's Dockerfile is replayed in the pod (scripts/common/dockerfile_replay.py), since pods
cannot build images.

Network policy (agent and verifier phases): no-network and allowlist are enforced with a
NetworkPolicy selecting the trial's pod (egress denied; for an allowlist, DNS plus the listed
IPs/CIDRs and the addresses the listed hostnames resolve to when the policy is applied).
The Dockerfile replay runs before any policy, as a Docker build has network.
"""

from __future__ import annotations

import asyncio
import re
import shlex
import shutil
import socket
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import NetworkMode, NetworkPolicy
from harbor.models.trial.paths import EnvironmentPaths

from scripts.common.k8s_sandbox import K8sPodSandboxManager
from scripts.common.dockerfile_replay import DockerfileReplay, ImageConfig, as_user, in_dir

ENVIRONMENT_TYPE = "evals-k8s"


def network_policy_manifest(name: str, pod: str, policy: NetworkPolicy, run_label: str) -> Dict[str, Any]:
    """Egress for the trial pod under a no-network or allowlist policy."""
    egress: List[Dict[str, Any]] = []
    if policy.network_mode == NetworkMode.ALLOWLIST:
        blocks = set()
        for entry in policy.allowed_hosts:
            host = str(entry)
            if "/" in host:
                blocks.add(host)
                continue
            try:
                infos = socket.getaddrinfo(host, None, family=socket.AF_INET)
            except socket.gaierror as exc:
                raise RuntimeError(f"allowlist host {host!r} does not resolve: {exc}") from exc
            blocks.update(f"{info[4][0]}/32" for info in infos)
        egress.append({"ports": [{"protocol": "UDP", "port": 53}, {"protocol": "TCP", "port": 53}]})
        egress.append({"to": [{"ipBlock": {"cidr": cidr}} for cidr in sorted(blocks)]})
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {
            "name": name,
            "labels": {"app.kubernetes.io/managed-by": "evals-post-train-sandbox", "sandbox-run": run_label},
        },
        "spec": {"podSelector": {"matchLabels": {"sandbox-pod": pod}}, "policyTypes": ["Egress"], "egress": egress},
    }


class K8sPodEnvironment(BaseEnvironment):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._manager = K8sPodSandboxManager()
        self._sandbox_id: Optional[str] = None
        self._pod = ""
        self._image = ImageConfig()
        self._has_network_policy = False
        super().__init__(*args, **kwargs)

    @staticmethod
    def type() -> str:
        return ENVIRONMENT_TYPE

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(
            disable_internet=True,
            network_allowlist=True,
            network_allowlist_hostnames=True,
            network_allowlist_ipv4_addresses=True,
            network_allowlist_ipv4_cidrs=True,
            dynamic_network_policy=True,
        )

    @property
    def _dockerfile(self) -> Path:
        return self.environment_dir / "Dockerfile"

    def _validate_definition(self) -> None:
        if not self.task_env_config.docker_image and not self._dockerfile.is_file():
            raise FileNotFoundError(f"{self.environment_dir}: neither [environment].docker_image nor a Dockerfile")

    # --- lifecycle ------------------------------------------------------------

    async def start(self, force_build: bool) -> None:
        await self._manager.start()
        replay = DockerfileReplay(self, force_build)
        exports = await replay.export_stages()
        self._sandbox_id = await self._manager.create_sandbox(None, self._pod_config(replay.image), self.session_id_suffix())
        self._pod = self._manager._sandboxes[self._sandbox_id].pod
        self._image = await replay.replay(exports)
        # Harbor's Docker environment bind-mounts these from the trial dir, so they exist by
        # construction; in a pod nobody makes them, and the verifier's redirect into
        # /logs/verifier fails before the tests run. World-writable for non-root agent users.
        paths = EnvironmentPaths()
        dirs = " ".join(str(d) for d in (paths.agent_dir, paths.user_agent_dir, paths.verifier_dir, paths.artifacts_dir))
        result = await self.exec(f"mkdir -p {dirs} && chmod 777 {dirs}", cwd="/", user="root", timeout_sec=60)
        if result.return_code != 0:
            raise RuntimeError(f"pod {self._pod}: creating the log directories failed: {result.stderr.strip()[-500:]}")
        if self.network_policy.network_mode != NetworkMode.PUBLIC:
            await self._apply_network_policy(self.network_policy)

    def _pod_config(self, image: str) -> Dict[str, Any]:
        memory_mb = self._effective_memory_mb
        return {
            "image": image,
            "platform": "linux/amd64",
            "cpu_limit": self._effective_cpus or 1,
            "memory_limit": f"{memory_mb}m" if memory_mb else None,
            "env_vars": self._startup_env(),
        }

    def session_id_suffix(self) -> str:
        # Pod names: sandbox-<run label>-<this>; keep the trial recognisable, cut to fit 63 chars.
        name = re.sub(r"[^a-z0-9-]+", "-", self.session_id.lower()).strip("-")
        return name[-40:].lstrip("-") or "trial"

    async def stop(self, delete: bool) -> None:
        if self._sandbox_id is None:
            return
        if self._has_network_policy:
            await self._delete_network_policy()
        if delete:
            await self._manager.delete_sandbox(self._sandbox_id)
        self._sandbox_id = None

    # --- exec and files -------------------------------------------------------

    async def exec(
        self,
        command: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_sec: Optional[int] = None,
        user: Optional[Any] = None,
    ) -> ExecResult:
        merged = {**self._image.env, **(self._merge_env(env) or {})}
        who = self._resolve_user(user)
        if who is None:
            who = self._image.user
        workdir = cwd or self.task_env_config.workdir or self._image.workdir
        # As Harbor's Docker environment runs it: bash -c, falling back to sh where there is no bash.
        shell = ["sh", "-c", f"if command -v bash >/dev/null 2>&1; then exec bash -c {shlex.quote(command)}; else exec sh -c {shlex.quote(command)}; fi"]
        try:
            code, out, err = await self._manager._run_detached(self._pod, in_dir(as_user(shell, who), workdir, merged), timeout_sec)
        except Exception as exc:
            termination = await self._manager._termination(self._pod)
            raise RuntimeError(f"exec in pod {self._pod} failed: {exc}" + (f" ({termination})" if termination else "")) from exc
        return ExecResult(stdout=out, stderr=err, return_code=code)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        with tempfile.TemporaryDirectory() as staging:
            shutil.copy2(source_path, Path(staging) / PurePosixPath(target_path).name)
            await self._manager._upload_dir(self._pod, staging, str(PurePosixPath(target_path).parent))

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self._manager._upload_dir(self._pod, source_dir, target_dir)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        data = await self._manager._read_file(self._pod, shlex.quote(source_path))
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        Path(target_path).write_bytes(data)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        await self._manager._download_dir(self._pod, source_dir, target_dir)

    # --- network policy ---------------------------------------------------------

    @property
    def _policy_name(self) -> str:
        return self._pod

    async def _apply_network_policy(self, network_policy: NetworkPolicy) -> None:
        if network_policy.network_mode == NetworkMode.PUBLIC:
            if self._has_network_policy:
                await self._delete_network_policy()
            return
        from kubernetes import client

        api = client.NetworkingV1Api(api_client=client.ApiClient())
        manifest = network_policy_manifest(
            self._policy_name, self._pod, network_policy, self._manager._run_label
        )
        namespace = self._manager._namespace
        if self._has_network_policy:
            await asyncio.to_thread(api.replace_namespaced_network_policy, self._policy_name, namespace, manifest)
        else:
            await asyncio.to_thread(api.create_namespaced_network_policy, namespace, manifest)
            self._has_network_policy = True

    async def _delete_network_policy(self) -> None:
        from kubernetes import client
        from kubernetes.client.rest import ApiException

        api = client.NetworkingV1Api(api_client=client.ApiClient())
        try:
            await asyncio.to_thread(api.delete_namespaced_network_policy, self._policy_name, self._manager._namespace)
        except ApiException as exc:
            if exc.status != 404:
                raise
        self._has_network_policy = False
