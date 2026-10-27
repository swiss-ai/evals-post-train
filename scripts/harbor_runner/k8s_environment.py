"""Pod-per-trial Kubernetes environment for Harbor (`harbor run --env <this module>:K8sPodEnvironment`).

Harbor's built-in Kubernetes providers are cloud-specific (gke, ack, openshift); this one
only needs a kubeconfig (or an in-cluster service account) with create/get/delete on pods
and networkpolicies and create on pods/exec, e.g. Breithorn's rob-poc namespace. The pod
plumbing is scripts/k8s_sandbox.py's K8sPodSandboxManager, shared with the
EvalScope runner: an unprivileged pod per sandbox, commands run detached in the pod and
waited for by short execs (a dropped API stream is retried, not misread), files moved as
verified tar chunks, and a dead container reported with its termination reason (OOMKilled).
Its SANDBOX_K8S_* settings (namespace, node arch, tolerations, memory request, run label,
pull secret, ready timeout) apply here too.

The trial's image:

- [environment].docker_image (Terminal-Bench, DeepSWE): the pod runs it directly.
- Otherwise the task's environment/Dockerfile is replayed in the pod: the pod starts from
  its FROM image and runs its RUN steps in order, with ENV, ARG, WORKDIR, SHELL and USER
  applied as a build would and COPY/ADD sources uploaded from the task's environment dir.
  Pods cannot build images, and for these tasks (SWE-bench's FROM <prebuilt image> plus a few
  setup lines, or a from-scratch toolchain) the result is the same filesystem a build makes.
  Multi-stage Dockerfiles (COPY --from) are rejected: run those with --sandbox docker.

Network policy (agent and verifier phases): no-network and allowlist are enforced with a
NetworkPolicy selecting the trial's pod (egress denied; for an allowlist, DNS plus the listed
IPs/CIDRs and the addresses the listed hostnames resolve to when the policy is applied).
The build replay runs before any policy, as a Docker build has network.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import socket
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import NetworkMode, NetworkPolicy
from harbor.models.trial.paths import EnvironmentPaths

from scripts.k8s_sandbox import K8sPodSandboxManager

ENVIRONMENT_TYPE = "evals-k8s"
# Instructions that only describe the image (ports, labels, the default process): a pod
# replay has nothing to do for them.
_IGNORED_INSTRUCTIONS = {"CMD", "ENTRYPOINT", "EXPOSE", "HEALTHCHECK", "LABEL", "MAINTAINER", "ONBUILD", "STOPSIGNAL", "VOLUME"}
_ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")
# The image's environment, as the container's first process sees it (image ENV plus pod env).
_READ_IMAGE_ENV = "cat /proc/1/environ"


def substitute(value: str, env: Dict[str, str]) -> str:
    """Dockerfile variable substitution: $VAR, ${VAR}, ${VAR:-default}, ${VAR:+alt}, \\$ escapes."""

    def braced(match: re.Match) -> str:
        name, op, word = match.group(1), match.group(2), match.group(3)
        current = env.get(name, "")
        if op == ":-":
            return current or word
        if op == ":+":
            return word if current else ""
        return current

    out = []
    i = 0
    while i < len(value):
        if value.startswith("\\$", i):
            out.append("$")
            i += 2
            continue
        if value[i] == "$":
            m = re.match(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:(:-|:\+)([^}]*))?\}", value[i:])
            if m:
                out.append(braced(m))
                i += m.end()
                continue
            m = re.match(r"\$([A-Za-z_][A-Za-z0-9_]*)", value[i:])
            if m:
                out.append(env.get(m.group(1), ""))
                i += m.end()
                continue
        out.append(value[i])
        i += 1
    return "".join(out)


def parse_assignments(value: str, env: Dict[str, str]) -> Dict[str, str]:
    """ENV/ARG arguments: `K=V K2="V 2"`, or the legacy `K V...` form; values substituted."""
    tokens = shlex.split(value, posix=True)
    if not tokens:
        return {}
    if "=" not in tokens[0]:
        key, _, rest = value.strip().partition(" ")
        return {key: substitute(rest.strip(), env)}
    out: Dict[str, str] = {}
    for token in tokens:
        key, eq, raw = token.partition("=")
        out[key] = substitute(raw, {**env, **out}) if eq else env.get(key, "")
    return out


def dockerfile_steps(dockerfile: Path) -> List[Dict[str, str]]:
    """The Dockerfile's instructions as [{"instruction", "value"}], refusing what a pod
    replay cannot reproduce."""
    from dockerfile_parse import DockerfileParser

    steps = [
        {"instruction": item["instruction"].upper(), "value": item["value"]}
        for item in DockerfileParser(path=str(dockerfile)).structure
        if item["instruction"].upper() != "COMMENT"
    ]
    if sum(step["instruction"] == "FROM" for step in steps) != 1:
        raise RuntimeError(
            f"{dockerfile}: multi-stage Dockerfile; the Kubernetes environment replays a single "
            "stage in a pod and cannot build this one: use --sandbox docker"
        )
    for step in steps:
        if step["instruction"] in ("COPY", "ADD") and "--from" in step["value"]:
            raise RuntimeError(f"{dockerfile}: COPY --from needs a real build: use --sandbox docker")
    return steps


def as_user(argv: List[str], user: Optional[str]) -> List[str]:
    """argv run as user (name or uid) by a root shell, without resetting the environment."""
    if user is None or str(user) in ("", "0", "root"):
        return argv
    command = shlex.join(argv)
    who = shlex.quote(str(user))
    lookup = f'$(getent passwd {who} | cut -d: -f1 || echo {who})' if str(user).isdigit() else who
    return [
        "sh",
        "-c",
        f"u={lookup}; if command -v runuser >/dev/null 2>&1; then exec runuser -u \"$u\" -- {command}; "
        f"else exec su -s /bin/sh \"$u\" -c {shlex.quote(command)}; fi",
    ]


def in_dir(argv: List[str], cwd: Optional[str], env: Dict[str, str]) -> List[str]:
    """argv with env added and cwd as its working directory (created if missing, as WORKDIR is)."""
    assignments = [f"{k}={v}" for k, v in env.items()]
    run = shlex.join(["env", *assignments, *argv]) if assignments else shlex.join(argv)
    if not cwd:
        return ["sh", "-c", f"exec {run}"]
    return ["sh", "-c", f"cd {shlex.quote(cwd)} && exec {run}"]


def network_policy_manifest(name: str, pod: str, policy: NetworkPolicy) -> Dict[str, Any]:
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
        "metadata": {"name": name, "labels": {"app.kubernetes.io/managed-by": "evals-post-train-sandbox"}},
        "spec": {"podSelector": {"matchLabels": {"sandbox-pod": pod}}, "policyTypes": ["Egress"], "egress": egress},
    }


class K8sPodEnvironment(BaseEnvironment):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._manager = K8sPodSandboxManager()
        self._sandbox_id: Optional[str] = None
        self._pod = ""
        self._image_env: Dict[str, str] = {}
        self._image_workdir: Optional[str] = None
        self._image_user: Optional[str] = None
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
        steps: List[Dict[str, str]] = []
        image = self.task_env_config.docker_image
        if not image or (force_build and self._dockerfile.is_file()):
            steps = dockerfile_steps(self._dockerfile)
            image = self._from_image(steps)
        memory_mb = self._effective_memory_mb
        config = {
            "image": image,
            "platform": "linux/amd64",
            "cpu_limit": self._effective_cpus or 1,
            "memory_limit": f"{memory_mb}m" if memory_mb else None,
            "env_vars": self._startup_env(),
        }
        self._sandbox_id = await self._manager.create_sandbox(None, config, self.session_id_suffix())
        self._pod = self._manager._sandboxes[self._sandbox_id].pod
        if steps:
            await self._replay(steps)
        # Harbor's Docker environment bind-mounts these from the trial dir, so they exist by
        # construction; in a pod nobody makes them, and the verifier's redirect into
        # /logs/verifier fails before the tests run. World-writable for non-root agent users.
        paths = EnvironmentPaths()
        dirs = [str(d) for d in (paths.agent_dir, paths.user_agent_dir, paths.verifier_dir, paths.artifacts_dir)]
        await self._build_step(["sh", "-c", f"mkdir -p {' '.join(dirs)} && chmod 777 {' '.join(dirs)}"], None, {}, None, 60, {"instruction": "setup", "value": "/logs"})
        if self.network_policy.network_mode != NetworkMode.PUBLIC:
            await self._apply_network_policy(self.network_policy)

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

    # --- Dockerfile replay ------------------------------------------------------

    def _from_image(self, steps: List[Dict[str, str]]) -> str:
        args: Dict[str, str] = {}
        for step in steps:
            if step["instruction"] == "ARG":
                args.update(parse_assignments(step["value"], args))
            elif step["instruction"] == "FROM":
                tokens = [t for t in step["value"].split() if not t.startswith("--")]
                return substitute(tokens[0], args)
        raise RuntimeError(f"{self._dockerfile}: no FROM")

    async def _replay(self, steps: List[Dict[str, str]]) -> None:
        code, out, err = await self._manager._run_detached(self._pod, ["sh", "-c", _READ_IMAGE_ENV], 60)
        base_env = dict(item.split("=", 1) for item in out.split("\0") if "=" in item) if code == 0 else {}
        env: Dict[str, str] = {}
        args: Dict[str, str] = {}
        shell = ["/bin/sh", "-c"]
        workdir: Optional[str] = None
        user: Optional[str] = None
        timeout = int(self.task_env_config.build_timeout_sec or 1800)
        before_from = True
        for step in steps:
            kind, value = step["instruction"], step["value"]
            scope = {**base_env, **args, **env}
            if kind == "FROM":
                before_from = False
            elif kind == "ARG":
                # ARGs declared before FROM are only visible to FROM (and to a later bare ARG).
                declared = parse_assignments(value, scope)
                args.update({k: v for k, v in declared.items() if not before_from})
            elif kind == "ENV":
                env.update(parse_assignments(value, scope))
            elif kind == "WORKDIR":
                path = substitute(value.strip(), scope)
                workdir = str(PurePosixPath(workdir or "/") / path)
                await self._build_step(["mkdir", "-p", workdir], None, {}, None, 60, step)
            elif kind == "SHELL":
                shell = json.loads(value)
            elif kind == "USER":
                user = substitute(value.strip(), scope).split(":", 1)[0]
            elif kind == "RUN":
                argv = json.loads(value) if value.lstrip().startswith("[") else [*shell, value]
                await self._build_step(argv, workdir, {**args, **env}, user, timeout, step)
            elif kind in ("COPY", "ADD"):
                await self._copy(value, workdir or "/", scope, kind == "ADD", step)
            elif kind not in _IGNORED_INSTRUCTIONS:
                raise RuntimeError(f"{self._dockerfile}: {kind} is not supported by the pod replay")
        self._image_env, self._image_workdir, self._image_user = env, workdir, user

    async def _build_step(
        self, argv: List[str], cwd: Optional[str], env: Dict[str, str], user: Optional[str], timeout: int, step: Dict
    ) -> None:
        code, out, err = await self._manager._run_detached(self._pod, in_dir(as_user(argv, user), cwd, env), timeout)
        if code != 0:
            tail = (out + err).strip()[-1500:]
            raise RuntimeError(f"pod {self._pod}: {step['instruction']} {step['value'][:200]} failed (exit {code}): {tail}")

    async def _copy(self, value: str, workdir: str, scope: Dict[str, str], is_add: bool, step: Dict) -> None:
        tokens = shlex.split(value) if not value.lstrip().startswith("[") else json.loads(value)
        chown = next((t.split("=", 1)[1] for t in tokens if t.startswith("--chown=")), None)
        chmod = next((t.split("=", 1)[1] for t in tokens if t.startswith("--chmod=")), None)
        paths = [substitute(t, scope) for t in tokens if not t.startswith("--")]
        *sources, dest = paths
        dest_abs = str(PurePosixPath(workdir) / dest)
        into_dir = dest.endswith("/") or len(sources) > 1
        with tempfile.TemporaryDirectory() as staging:
            root = Path(staging)
            for source in sources:
                if is_add and re.match(r"https?://", source):
                    target = f"{dest_abs.rstrip('/')}/{PurePosixPath(source).name}" if into_dir else dest_abs
                    await self._build_step(
                        ["sh", "-c", f"mkdir -p \"$(dirname {shlex.quote(target)})\" && curl -fsSL -o {shlex.quote(target)} {shlex.quote(source)}"],
                        None, {}, None, 600, step,
                    )
                    continue
                matches = sorted(self.environment_dir.glob(source)) or [self.environment_dir / source]
                for match in matches:
                    if not match.exists():
                        raise RuntimeError(f"{self._dockerfile}: {step['instruction']} source {source} not found")
                    if match.is_dir():
                        target = root / dest_abs.lstrip("/")
                        shutil.copytree(match, target, dirs_exist_ok=True, symlinks=True)
                    elif is_add and match.name.endswith(_ARCHIVE_SUFFIXES):
                        target = root / dest_abs.lstrip("/")
                        target.mkdir(parents=True, exist_ok=True)
                        with tarfile.open(match) as archive:
                            archive.extractall(target, filter="tar")
                    else:
                        name = dest_abs.lstrip("/")
                        target = root / name / match.name if (into_dir or dest_abs.endswith("/")) else root / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(match, target)
            if any(root.iterdir()):
                await self._manager._upload_dir(self._pod, root, "/")
        for flag, tool in ((chown, "chown"), (chmod, "chmod")):
            if flag:
                await self._build_step([tool, "-R", flag, dest_abs], None, {}, None, 120, step)

    # --- exec and files -------------------------------------------------------

    async def exec(
        self,
        command: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_sec: Optional[int] = None,
        user: Optional[Any] = None,
    ) -> ExecResult:
        merged = {**self._image_env, **(self._merge_env(env) or {})}
        who = self._resolve_user(user)
        if who is None:
            who = self._image_user
        workdir = cwd or self.task_env_config.workdir or self._image_workdir
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
        manifest = network_policy_manifest(self._policy_name, self._pod, network_policy)
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
