"""Sandbox-per-trial Kata microVMs for Harbor, through an OpenSandbox server (`--sandbox opensandbox`).

Harbor's OpenSandboxEnvironment does the sandbox lifecycle, exec and file transfer. On top of it:

- Every sandbox gets an explicit network policy, because a sandbox created without one has open
  internet whatever the task asked for. The task's policy is what is sent: public is
  defaultAction allow, no-network is deny, and an allowlist is deny plus an allow rule per entry.
  Hostnames and wildcard hostnames are enforced by the egress sidecar, which a NetworkPolicy
  cannot do. The policy is fixed at creation (the SDK can patch rules, not defaultAction), so
  Harbor refuses a task that is set up with public networking and then narrowed for its agent.
- vm.overcommit_memory is 1 in the microVM, whose RAM is only the task's memory limit: memory
  allocations then behave as in a container on a large node (see _allow_overcommit).
- A command's output is rebuilt from its lines (the server streams one message per line, without
  the newline).
- A command whose stream ended without a result raises: Harbor's exec would report exit code 0, so
  a dropped connection during verification could look like a passing test run.
- Sandboxes are created with secure_access (per-sandbox access credentials, which the SDK sends
  with each call): Breithorn's server refuses sandboxes without it.
- Sandboxes carry the sandbox-run=<OPENSANDBOX_RUN_LABEL> metadata (set per run by run.py), and
  delete_run_sandboxes removes what a run left behind.

OpenSandbox cannot build images: a task without [environment].docker_image has its Dockerfile
replayed in the sandbox (scripts/harbor/dockerfile_replay.py, as for --sandbox k8s), and later
commands get the replayed ENV, WORKDIR and USER. The replay needs network, so it only runs for
tasks whose policy is public.

The server is OPENSANDBOX_DOMAIN with OPENSANDBOX_API_KEY. It decides the container runtime, so
this must be a Kata endpoint: a gVisor one rejects every network policy with HTTP 400.
"""

from __future__ import annotations

import asyncio
import os
import shlex
from datetime import timedelta
from typing import Any

from harbor.environments.base import ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.definition import effective_exec_cwd
from harbor.environments.opensandbox import OpenSandboxEnvironment
from harbor.models.task.config import NetworkMode
from opensandbox import Sandbox
from opensandbox.config import ConnectionConfig
from opensandbox.manager import SandboxManager
from opensandbox.models.sandboxes import NetworkRule, SandboxFilter

from scripts.harbor.dockerfile_replay import DockerfileReplay, ImageConfig

RUN_LABEL_ENV = "OPENSANDBOX_RUN_LABEL"
RUN_LABEL_KEY = "sandbox-run"
# Clients outside the cluster (a Clariden job, a laptop) reach sandboxes through the server.
USE_SERVER_PROXY = True
# Creating a sandbox can wait on a pull of a several-GB image (the SDK's default request is 30 s).
REQUEST_TIMEOUT_SEC = 300
READY_TIMEOUT_SEC = 1800


def joined_lines(messages: list[Any]) -> str:
    """A command's output from its stream messages: the server sends one per line, without the
    newline (an empty line as "\n"), and Harbor's own exec joins them with nothing."""
    lines = [message.text.rstrip("\n") for message in messages]
    return "\n".join(lines) + "\n" if lines else ""


class SecureSandbox(Sandbox):
    """Sandbox.create with secure_access, which Harbor's environment doesn't pass."""

    @classmethod
    async def create(cls, *args: Any, **kwargs: Any) -> Sandbox:
        return await super().create(*args, secure_access=True, **kwargs)


class KataSandboxEnvironment(OpenSandboxEnvironment):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("use_server_proxy", USE_SERVER_PROXY)
        kwargs.setdefault("request_timeout_sec", REQUEST_TIMEOUT_SEC)
        kwargs.setdefault("ready_timeout_sec", READY_TIMEOUT_SEC)
        kwargs.setdefault("metadata", {RUN_LABEL_KEY: os.environ[RUN_LABEL_ENV]})
        self._image_ref: str | None = None
        self._image = ImageConfig()
        self._image_uid: int | None = None
        super().__init__(*args, **kwargs)

    def _load_opensandbox(self) -> dict[str, Any]:
        return {**super()._load_opensandbox(), "Sandbox": SecureSandbox}

    def _validate_definition(self) -> None:
        compose = any((self.environment_dir / name).exists() for name in ("docker-compose.yaml", "docker-compose.yml"))
        if self.task_env_config.docker_image or compose or not self._dockerfile_path.is_file():
            super()._validate_definition()

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return super().capabilities.model_copy(
            update={
                "network_allowlist": True,
                "network_allowlist_hostnames": True,
                "network_allowlist_wildcard_hostnames": True,
                "network_allowlist_ipv4_addresses": True,
                "network_allowlist_ipv4_cidrs": True,
            }
        )

    async def start(self, force_build: bool) -> None:
        replay = DockerfileReplay(self, force_build)
        if replay.stages and self.network_policy.network_mode != NetworkMode.PUBLIC:
            # The policy is fixed at creation and the replay installs packages.
            raise RuntimeError(f"{self.session_id}: a Dockerfile replay needs network, but the task's policy is not public")
        exports = await replay.export_stages()
        self._image_ref = replay.image
        await super().start(force_build)
        await self._allow_overcommit()
        self._image = await replay.replay(exports)
        if self._image.user:
            self._image_uid = await self._uid(self._image.user)

    async def _allow_overcommit(self) -> None:
        # The microVM's RAM is the task's memory limit, so Linux's default overcommit heuristic
        # refuses any single allocation above it (malloc returns NULL), where a container on a
        # large node gets it. gpt2-codegolf's reference solution mallocs several GB and touches
        # far less. Always overcommitting, as such a node effectively does, keeps them alike.
        result = await self.exec("echo 1 > /proc/sys/vm/overcommit_memory", cwd="/", user="root", timeout_sec=60)
        if result.return_code != 0:
            raise RuntimeError(f"sandbox {self.session_id}: setting vm.overcommit_memory failed: {result.stderr.strip()}")

    def _resolve_image_spec(self, sdk: dict[str, Any]) -> Any:
        # The replay's FROM image, or the task's docker_image (no registry credentials here).
        return self._image_ref

    async def _uid(self, user: str) -> int:
        """The uid of the Dockerfile's USER: commands here take a uid, not a name."""
        if user.isdigit():
            return int(user)
        result = await self.exec(f"id -u {shlex.quote(user)}", cwd="/", user="root", timeout_sec=60)
        if result.return_code != 0 or not result.stdout.strip().isdigit():
            raise RuntimeError(f"sandbox {self.session_id}: no uid for the Dockerfile's USER {user}: {result.stderr.strip()}")
        return int(result.stdout.strip())

    def _resolve_user(self, user: str | int | None) -> str | int | None:
        resolved = super()._resolve_user(user)
        return self._image_uid if resolved is None else resolved

    def _effective_cwd(self, cwd: str | None) -> str | None:
        return effective_exec_cwd(cwd, self.task_env_config.workdir, self._image.workdir)

    def _build_network_policy(self, sdk: dict[str, Any]) -> Any:
        policy = self.network_policy
        if policy.network_mode == NetworkMode.PUBLIC:
            return sdk["NetworkPolicy"](defaultAction="allow", egress=[])
        rules = [NetworkRule(action="allow", target=str(host)) for host in policy.allowed_hosts]
        return sdk["NetworkPolicy"](defaultAction="deny", egress=rules)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        execution = await self._sandbox.commands.run(
            command,
            opts=self._build_run_command_opts(
                self._load_opensandbox(), cwd, {**self._image.env, **(self._merge_env(env) or {})}, timeout_sec, user
            ),
        )
        return_code = execution.exit_code
        if return_code is None:
            if execution.error is None:
                raise RuntimeError(f"sandbox {self._sandbox.id}: the command's stream ended without a result")
            return_code = 1
        return ExecResult(
            stdout=joined_lines(execution.logs.stdout),
            stderr=joined_lines(execution.logs.stderr),
            return_code=return_code,
        )


async def delete_run_sandboxes(run_label: str, domain: str, api_key: str) -> int:
    """Kills every sandbox of the run; returns how many. Lists all pages before the first kill,
    which would shift them."""
    # The SDK's default protocol is http; Harbor's environment (and the server) use https.
    config = ConnectionConfig(domain=domain, api_key=api_key, protocol="https", request_timeout=timedelta(seconds=60))
    manager = await SandboxManager.create(config)
    try:
        live: list[str] = []
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
