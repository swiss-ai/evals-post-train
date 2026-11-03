"""Inspect sandbox provider "opensandbox": each sample's sandbox is a Kata microVM on an
OpenSandbox server (`run_inspect_eval.sh --sandbox opensandbox`), instead of a Docker container.
No Docker daemon and no Kubernetes access are needed, only OPENSANDBOX_API_KEY.

Registered by scripts/inspect_ai/sandbox_patches/sitecustomize.py (imported at interpreter
start-up), so `inspect eval ... --sandbox opensandbox` finds it.

The provider is "docker compatible": overriding a task's own ("docker", <compose file or
Dockerfile>) sandbox with --sandbox opensandbox keeps the task's file. It auto-discovers none, so
the file is always the one the task names. That file is read as Docker
would, within what one microVM per sample can do:

- One service. A task's compose file with several services (a CTF's victim containers) is
  refused: sandboxes here cannot reach each other by service name.
- `image:` runs as is. `build:` (or a bare Dockerfile) cannot be built here: the sandbox starts
  from the Dockerfile's FROM image and its steps are replayed in it (scripts/common/
  dockerfile_replay.py, as for Harbor), multi-stage COPY --from included (each helper stage in a
  short-lived sandbox of its own). The replay runs with network, as a build does.
- `working_dir`, `environment`, `user` and the replayed ENV/WORKDIR/USER apply to every command.
- `network_mode: none` cuts the sandbox off once it is set up (after the replay); otherwise it
  has open internet, as under Docker.
- `cpus`/`mem_limit` (or deploy.resources.limits) size the microVM (default 2 CPUs, 4g).
  `x-local`, ports, healthchecks, command/entrypoint and volumes have no meaning here and are
  ignored.

Commands go through the server's exec API as argv (no shell), under GNU timeout when Inspect
gives a timeout; stdin, oversized `-c` payloads (SciCode's composed solutions are far above
Linux's 128 KiB limit for one argument) and files go through the file API. Files are written
and read as the sample's default user, so permissions behave as in a container.

A sample's sandbox is killed when the sample ends, and a task kills what is left of its own when
it ends. Every sandbox of a run also carries its run label (OPENSANDBOX_RUN_LABEL, default:
random) and lives at most SANDBOX_TIMEOUT_SEC; run_inspect_eval.sh deletes leftovers of its label
once all tasks are done (never a task itself: tasks of one eval-set share the label).

Command output comes back line by line from the server, so non-empty output always ends with a
newline (as with Harbor's and EvalScope's OpenSandbox backends).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import posixpath
import shlex
import time
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path, PurePosixPath
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from inspect_ai.util import (
    ComposeConfig,
    ExecResult,
    OutputLimitExceededError,
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
    SandboxEnvironmentLimits,
    parse_compose_yaml,
    sandboxenv,
)
from inspect_ai.util._sandbox.compose import is_compose_yaml, is_dockerfile

from scripts.common.dockerfile_replay import DockerfileReplay, ImageConfig, dockerfile_stages, from_image
from scripts.common.k8s_sandbox import _HOSTS_LOCALHOST_IPV4_ONLY, k8s_memory
from scripts.common.opensandbox_sandbox import (
    HEALTH_POLL_SEC,
    OVERCOMMIT_COMMAND,
    READY_TIMEOUT_SEC,
    REQUEST_SLACK_SEC,
    RUN_LABEL_ENV,
    RUN_LABEL_KEY,
    connection_config,
    delete_run_sandboxes,
    joined_lines,
    replace_egress_policy,
)

PROVIDER = "opensandbox"
# Inspect's docker provider's image when a task names no config.
DEFAULT_IMAGE = "aisiuk/inspect-tool-support"
DEFAULT_CPU = 2.0
DEFAULT_MEMORY = "4g"
# A sample's sandbox lives at most this long (a long agent loop plus scoring).
SANDBOX_TIMEOUT_SEC = 12 * 3600
# Commands Inspect gives no timeout still end at the request's.
DEFAULT_COMMAND_TIMEOUT_SEC = 3600
DEFAULT_BUILD_TIMEOUT_SEC = 1800
# Above this many bytes, an argv element of `python -c`/`sh -c` goes through a file: Linux refuses
# one argument above 128 KiB (MAX_ARG_STRLEN) however large ARG_MAX is.
INLINE_ARG_LIMIT = 64 * 1024
# The wrappers this provider puts in front of a command run with the user's authority, so they are
# looked up by absolute path, never through the image's PATH (SandboxEnvironment.exec's contract).
SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
SH = "/bin/sh"
# A command GNU timeout stopped: its own 124 (TERM) or 137 (KILL after -k), or -1, how the server
# reports a command that ended by a signal.
TIMEOUT_EXIT_CODES = (124, 137, -1)


# The sandboxes each task (by name) has running in this process: what its task_cleanup kills. Not
# the run label: tasks of one `inspect eval-set` share it, and one task ending must not take the
# others' sandboxes with it (the runner deletes leftovers of the label once everything is done).
_LIVE: Dict[str, Dict[str, Any]] = {}


_now = time.monotonic  # replaced in tests (asyncio's own clock must stay real)


def run_label() -> str:
    """The label every sandbox of this run carries (set once per process)."""
    label = os.environ.get(RUN_LABEL_ENV)
    if not label:
        label = f"inspect-{uuid.uuid4().hex[:12]}"
        os.environ[RUN_LABEL_ENV] = label
    return label


# --- reading the task's sandbox config ----------------------------------------------------------


@dataclass
class ServiceSpec:
    """What one compose service (or a bare Dockerfile) asks for, as this provider applies it."""

    name: str
    image: Optional[str] = None
    dockerfile: Optional[Path] = None
    context: Optional[Path] = None
    working_dir: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    user: Optional[str] = None
    network_none: bool = False
    cpu: float = DEFAULT_CPU
    memory: str = DEFAULT_MEMORY
    platform: Optional[str] = None


def _environment(value: Any) -> Dict[str, str]:
    if not value:
        return {}
    if isinstance(value, dict):
        return {k: "" if v is None else str(v) for k, v in value.items()}
    pairs = (item.split("=", 1) for item in value)
    return {pair[0]: pair[1] if len(pair) > 1 else os.environ.get(pair[0], "") for pair in pairs}


def _service_spec(name: str, service: Any, base_dir: Path) -> ServiceSpec:
    spec = ServiceSpec(name=name, image=service.image)
    build = service.build
    if build is not None:
        context = build if isinstance(build, str) else (build.context or ".")
        spec.context = (base_dir / context).resolve()
        dockerfile = None if isinstance(build, str) else build.dockerfile
        spec.dockerfile = spec.context / (dockerfile or "Dockerfile")
    if not spec.image and spec.dockerfile is None:
        raise ValueError(f"opensandbox: service {name!r} has neither image nor build")
    spec.working_dir = service.working_dir
    spec.env = _environment(service.environment)
    spec.user = service.user
    spec.network_none = service.network_mode == "none"
    limits = service.deploy.resources.limits if service.deploy and service.deploy.resources else None
    cpus = service.cpus or (limits.cpus if limits else None)
    memory = service.mem_limit or (limits.memory if limits else None)
    spec.cpu = float(cpus) if cpus else DEFAULT_CPU
    spec.memory = str(memory) if memory else DEFAULT_MEMORY
    spec.platform = getattr(service, "platform", None)
    return spec


def service_specs(config: SandboxEnvironmentConfigType | None) -> List[ServiceSpec]:
    """The task's sandbox config as service specs, the default service first."""
    if config is None:
        return [ServiceSpec(name="default", image=DEFAULT_IMAGE)]
    if isinstance(config, str) and is_dockerfile(config):
        dockerfile = Path(config).resolve()
        return [ServiceSpec(name="default", dockerfile=dockerfile, context=dockerfile.parent)]
    if isinstance(config, str) and is_compose_yaml(config):
        compose = parse_compose_yaml(config)
        base_dir = Path(config).resolve().parent
    elif isinstance(config, ComposeConfig):
        compose, base_dir = config, Path.cwd()
    else:
        raise ValueError(f"opensandbox: unsupported sandbox config {config!r} (a compose file, a Dockerfile or none)")
    services = list(compose.services.items())
    if len(services) != 1:
        raise ValueError(
            f"opensandbox: the task's compose config has {len(services)} services "
            f"({', '.join(name for name, _ in services)}); this provider runs one sandbox per sample "
            "and sandboxes cannot reach each other by service name. Use --sandbox docker."
        )
    name, service = services[0]
    return [_service_spec("default", service, base_dir)]


# --- the replay, through this provider's sandboxes -----------------------------------------------


@dataclass
class _ReplayExec:
    return_code: int
    stdout: str
    stderr: str


class _ReplayTarget:
    """The exec/upload_file/download_file surface dockerfile_replay uses (Harbor's), on a raw
    OpenSandbox sandbox: as root, from /, with the replay's own env."""

    def __init__(self, sandbox: Any, context: Path):
        self.sandbox = sandbox
        self.environment_dir = context
        self.session_id = sandbox.id

    async def exec(self, command: str, cwd: str = "/", user: Any = None, timeout_sec: Optional[int] = None) -> _ReplayExec:
        code, out, err = await _run(self.sandbox, [SH, "-c", command], cwd=cwd, uid=0, timeout=timeout_sec)
        return _ReplayExec(code, out, err)

    async def upload_file(self, local: Union[str, Path], remote: str) -> None:
        from opensandbox.models.filesystem import WriteEntry

        await self.sandbox.files.write_files([WriteEntry(path=remote, data=Path(local).read_bytes(), mode=644)])

    async def download_file(self, remote: str, local: Union[str, Path]) -> None:
        Path(local).write_bytes(await self.sandbox.files.read_bytes(remote))


class _Replay(DockerfileReplay):
    """DockerfileReplay for one Inspect service: its Dockerfile and build context, helper stages
    in short-lived sandboxes of this provider."""

    def __init__(self, spec: ServiceSpec):
        self.environment = None  # set by bind() once the sample's sandbox exists
        self.spec = spec
        self.dockerfile = spec.dockerfile
        self.stages = dockerfile_stages(spec.dockerfile)
        self.image = from_image(self.stages[-1]["steps"], spec.dockerfile)
        self.timeout = DEFAULT_BUILD_TIMEOUT_SEC
        self._context = spec.context or spec.dockerfile.parent

    def bind(self, sandbox: Any) -> None:
        self.environment = _ReplayTarget(sandbox, self._context)

    def stage_environment(self, image: str) -> contextlib.AbstractAsyncContextManager:
        return self._stage(image)

    @contextlib.asynccontextmanager
    async def _stage(self, image: str) -> AsyncIterator[_ReplayTarget]:
        stage_spec = ServiceSpec(name="stage", image=image, cpu=self.spec.cpu, memory=self.spec.memory)
        sandbox = await _create_sandbox(stage_spec, image, {"sandbox-role": "build-stage"})
        try:
            yield _ReplayTarget(sandbox, self._context)
        finally:
            await _kill(sandbox)


# --- talking to the server -------------------------------------------------------------------------


async def _create_sandbox(spec: ServiceSpec, image: str, metadata: Optional[Dict[str, str]] = None) -> Any:
    from opensandbox import Sandbox
    from opensandbox.models.sandboxes import NetworkPolicy

    from scripts.common.opensandbox_sandbox import platform_spec

    return await Sandbox.create(
        image,
        timeout=timedelta(seconds=SANDBOX_TIMEOUT_SEC),
        ready_timeout=timedelta(seconds=READY_TIMEOUT_SEC),
        env=spec.env or None,
        metadata={RUN_LABEL_KEY: run_label(), **(metadata or {})},
        resource={"cpu": str(spec.cpu), "memory": k8s_memory(spec.memory)},
        platform=platform_spec({"platform": spec.platform}) if spec.platform else None,
        # Open, as a Docker container is; network_mode: none is applied after set-up.
        network_policy=NetworkPolicy(defaultAction="allow", egress=[]),
        secure_access=True,
        connection_config=connection_config(),
        health_check_polling_interval=timedelta(seconds=HEALTH_POLL_SEC),
    )


async def _kill(sandbox: Any) -> None:
    """Kills the sandbox; one that is already gone (expired, or failed on its own) is no error,
    and nothing here hides the error that led to the kill."""
    try:
        await sandbox.kill()
    except Exception as exc:  # noqa: BLE001
        print(f"opensandbox: could not kill sandbox {sandbox.id}: {exc}", flush=True)
    finally:
        with contextlib.suppress(Exception):
            await sandbox.close()


async def _run(
    sandbox: Any,
    argv: List[str],
    cwd: Optional[str] = None,
    uid: Optional[int] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
) -> tuple[int, str, str]:
    """argv as is (the caller adds any timeout wrapper); the request ends a minute after timeout."""
    from opensandbox.models.execd import RunCommandOpts

    limit = float(timeout) if timeout and float(timeout) > 0 else DEFAULT_COMMAND_TIMEOUT_SEC
    execution = await sandbox.commands.run(
        argv,
        opts=RunCommandOpts(
            working_directory=cwd,
            timeout=timedelta(seconds=limit + REQUEST_SLACK_SEC),
            uid=uid,
            envs=env or None,
        ),
    )
    code = execution.exit_code
    stderr = joined_lines(execution.logs.stderr)
    if code is None:
        if execution.error is None:
            raise RuntimeError(f"sandbox {sandbox.id}: the command's stream ended without a result")
        code = 1
        stderr = stderr or f"{execution.error}"
    return code, joined_lines(execution.logs.stdout), stderr


async def _checked(sandbox: Any, script: str, what: str, uid: Optional[int] = 0) -> str:
    code, out, err = await _run(sandbox, [SH, "-c", script], cwd="/", uid=uid, timeout=600)
    if code != 0:
        raise RuntimeError(f"opensandbox: {what} in sandbox {sandbox.id} failed (exit {code}): {err.strip()[:500]}")
    return out


def _limit_output(stream: str) -> str:
    limit = SandboxEnvironmentLimits.MAX_EXEC_OUTPUT_SIZE
    if len(stream.encode("utf-8", errors="replace")) > limit:
        raise OutputLimitExceededError(
            limit_str=SandboxEnvironmentLimits.MAX_EXEC_OUTPUT_SIZE_STR,
            truncated_output=stream.encode("utf-8", errors="replace")[-limit:].decode("utf-8", errors="replace"),
        )
    return stream


# --- the provider ------------------------------------------------------------------------------------


@sandboxenv(name=PROVIDER)
class OpenSandboxEnvironment(SandboxEnvironment):
    def __init__(self, sandbox: Any, image: ImageConfig, spec: ServiceSpec, default_uid: Optional[int], timeout_bin: Optional[str]):
        super().__init__()
        self.sandbox = sandbox
        self.working_dir = spec.working_dir or image.workdir or "/"
        self.env = {**spec.env, **image.env}
        self.default_uid = default_uid
        self.timeout_bin = timeout_bin
        self._uids: Dict[str, int] = {}

    # --- lifecycle -------------------------------------------------------------------------------

    @classmethod
    def is_docker_compatible(cls) -> bool:
        # A task's Dockerfile or compose file is carried over when --sandbox opensandbox
        # overrides its docker sandbox (and deserialized as a ComposeConfig).
        return True

    @classmethod
    def config_files(cls) -> list[str]:
        # None to auto-discover: Inspect would otherwise pick a compose.yaml lying in the task's
        # directory over the file the task itself names (a task choosing between several).
        return []

    @classmethod
    def default_concurrency(cls) -> int | None:
        return int(os.environ.get("OPENSANDBOX_MAX_SANDBOXES", "16"))

    @classmethod
    async def task_init(cls, task_name: str, config: SandboxEnvironmentConfigType | None) -> None:
        if not os.environ.get("OPENSANDBOX_API_KEY"):
            raise RuntimeError("--sandbox opensandbox needs OPENSANDBOX_API_KEY")
        service_specs(config)  # refuse an unusable config before any sample starts
        run_label()

    @classmethod
    async def sample_init(
        cls, task_name: str, config: SandboxEnvironmentConfigType | None, metadata: dict[str, str]
    ) -> dict[str, SandboxEnvironment]:
        environments: Dict[str, SandboxEnvironment] = {}
        try:
            for spec in service_specs(config):
                environment = await cls._start(spec)
                environments[spec.name] = environment
                _LIVE.setdefault(task_name, {})[environment.sandbox.id] = environment.sandbox
        except BaseException:
            await cls.sample_cleanup(task_name, config, environments, True)
            raise
        return environments

    @classmethod
    async def _start(cls, spec: ServiceSpec) -> "OpenSandboxEnvironment":
        replay = _Replay(spec) if spec.dockerfile is not None and spec.dockerfile.is_file() else None
        if spec.dockerfile is not None and replay is None and not spec.image:
            raise FileNotFoundError(f"opensandbox: {spec.dockerfile} not found")
        exports = await replay.export_stages() if replay else {}
        sandbox = await _create_sandbox(spec, replay.image if replay else spec.image)
        try:
            await _checked(sandbox, OVERCOMMIT_COMMAND, "set vm.overcommit_memory")
            await _run(sandbox, [SH, "-c", _HOSTS_LOCALHOST_IPV4_ONLY], cwd="/", uid=0, timeout=60)
            image = ImageConfig()
            if replay:
                replay.bind(sandbox)
                image = await replay.replay(exports)
            user = spec.user or image.user
            default_uid = await cls._resolve_uid(sandbox, user) if user else None
            workdir = spec.working_dir or image.workdir
            if workdir:
                await _checked(sandbox, f"mkdir -p {shlex.quote(workdir)}", "create the working dir")
            else:
                # The image's own WORKDIR: where a command given no working directory starts.
                code, out, _ = await _run(sandbox, [SH, "-c", "pwd"], timeout=60)
                workdir = out.strip() if code == 0 and out.strip() else "/"
            spec = ServiceSpec(**{**spec.__dict__, "working_dir": workdir})
            timeout_bin = (await _checked(sandbox, f"PATH={SYSTEM_PATH} command -v timeout || true", "find timeout")).strip() or None
            if spec.network_none:
                from opensandbox.models.sandboxes import NetworkPolicy

                await replace_egress_policy(sandbox, NetworkPolicy(defaultAction="deny", egress=[]))
            return cls(sandbox, image, spec, default_uid, timeout_bin)
        except BaseException:
            await _kill(sandbox)
            raise

    @staticmethod
    async def _resolve_uid(sandbox: Any, user: str) -> int:
        name = user.split(":", 1)[0]
        if name.isdigit():
            return int(name)
        out = await _checked(sandbox, f"id -u {shlex.quote(name)}", f"look up user {name}")
        return int(out.strip())

    @classmethod
    async def sample_cleanup(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        environments: dict[str, SandboxEnvironment],
        interrupted: bool,
    ) -> None:
        for environment in environments.values():
            if isinstance(environment, OpenSandboxEnvironment):
                _LIVE.get(task_name, {}).pop(environment.sandbox.id, None)
                await _kill(environment.sandbox)

    @classmethod
    async def task_cleanup(cls, task_name: str, config: SandboxEnvironmentConfigType | None, cleanup: bool) -> None:
        live = _LIVE.pop(task_name, {})
        if cleanup:
            await asyncio.gather(*(_kill(sandbox) for sandbox in live.values()))

    @classmethod
    async def cli_cleanup(cls, id: str | None) -> None:
        # `inspect sandbox cleanup opensandbox <run label>`
        if id:
            await delete_run_sandboxes(id)

    # --- commands --------------------------------------------------------------------------------

    def _path(self, file: str) -> str:
        return file if file.startswith("/") else posixpath.join(self.working_dir, file)

    async def _uid(self, user: Optional[str]) -> Optional[int]:
        if user is None:
            return self.default_uid
        if user not in self._uids:
            self._uids[user] = await self._resolve_uid(self.sandbox, user)
        return self._uids[user]

    async def _upload(self, data: bytes) -> str:
        """data in a fresh file only its reader needs (removed by the caller's command)."""
        from opensandbox.models.filesystem import WriteEntry

        path = f"/tmp/.inspect-{uuid.uuid4().hex}"
        await self.sandbox.files.write_files([WriteEntry(path=path, data=data, mode=644)])
        return path

    async def _shrink(self, argv: List[str], cleanup: List[str]) -> List[str]:
        """`python -c <code>` / `sh -c <script>` with an argument too large for execve: the code
        goes through a file instead (`python <file>`, `sh <file>`), with the same arguments after it."""
        if len(argv) >= 3 and argv[1] == "-c" and len(argv[2].encode()) > INLINE_ARG_LIMIT:
            path = await self._upload(argv[2].encode())
            cleanup.append(path)
            return [argv[0], path, *argv[3:]]
        return argv

    async def exec(
        self,
        cmd: list[str],
        input: str | bytes | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: int | None = None,
        timeout_retry: bool = True,
        concurrency: bool = True,
    ) -> ExecResult[str]:
        cleanup: List[str] = []
        try:
            argv = await self._shrink(list(cmd), cleanup)
            if input is not None:
                path = await self._upload(input.encode() if isinstance(input, str) else input)
                cleanup.append(path)
                argv = [SH, "-c", 'exec "$@" < "$0"', path, *argv]
            uid = await self._uid(user)
            workdir = self._path(cwd) if cwd else self.working_dir
            envs = {**self.env, **(env or {})}
            attempts = [timeout]
            if timeout and timeout_retry:
                attempts += [min(timeout, 60), min(timeout, 30)]
            for attempt in attempts:
                result = await self._exec_once(argv, workdir, uid, envs, attempt)
                if result is not None:
                    return result
            raise TimeoutError(f"command timed out after {timeout}s: {shlex.join(list(cmd))[:200]}")
        finally:
            if cleanup:
                with contextlib.suppress(Exception):
                    await _run(self.sandbox, ["rm", "-f", *cleanup], cwd="/", uid=0, timeout=60)

    async def _exec_once(
        self, argv: List[str], workdir: str, uid: Optional[int], envs: Dict[str, str], timeout: Optional[int]
    ) -> Optional[ExecResult[str]]:
        """The result, or None when the command timed out."""
        wrapped = argv
        if timeout and self.timeout_bin:
            wrapped = [self.timeout_bin, "-s", "TERM", "-k", "5s", f"{timeout}s", *argv]
        started = _now()
        try:
            code, out, err = await _run(self.sandbox, wrapped, cwd=workdir, uid=uid, env=envs, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        except Exception as exc:
            if timeout and "timed out" in str(exc).lower() and _now() - started >= timeout:
                return None
            raise
        if timeout and code in TIMEOUT_EXIT_CODES and _now() - started + 0.05 >= timeout:
            return None
        if code == 126 and "permission denied" in (out + err).lower():
            raise PermissionError(err.strip() or out.strip())
        return ExecResult(success=code == 0, returncode=code, stdout=_limit_output(out), stderr=_limit_output(err))

    # --- files -----------------------------------------------------------------------------------

    async def write_file(self, file: str, contents: str | bytes) -> None:
        path = self._path(file)
        staged = await self._upload(contents.encode("utf-8") if isinstance(contents, str) else contents)
        quoted, parent = shlex.quote(path), shlex.quote(str(PurePosixPath(path).parent))
        # As the default user: the file gets their ownership, and their permissions decide.
        script = f'if [ -d {quoted} ]; then exit 21; fi; mkdir -p {parent} && cat "$0" > {quoted}'
        try:
            code, out, err = await _run(
                self.sandbox, [SH, "-c", script, staged], cwd="/", uid=self.default_uid, env=self.env, timeout=600
            )
        finally:
            with contextlib.suppress(Exception):
                await _run(self.sandbox, ["rm", "-f", staged], cwd="/", uid=0, timeout=60)
        if code == 21:
            raise IsADirectoryError(f"{path} is a directory")
        if code != 0:
            if "permission denied" in (out + err).lower():
                raise PermissionError(f"writing {path}: {err.strip()}")
            raise RuntimeError(f"writing {path} failed (exit {code}): {err.strip()[:500]}")

    async def read_file(self, file: str, text: bool = True) -> Union[str | bytes]:
        path = self._path(file)
        quoted = shlex.quote(path)
        # Checked as the default user, then read through the file API.
        script = (
            f"if [ -d {quoted} ]; then exit 21; elif [ ! -e {quoted} ]; then exit 2; "
            f"elif [ ! -r {quoted} ]; then exit 13; fi; wc -c < {quoted}"
        )
        code, out, err = await _run(self.sandbox, [SH, "-c", script], cwd="/", uid=self.default_uid, env=self.env, timeout=120)
        if code == 21:
            raise IsADirectoryError(f"{path} is a directory")
        if code == 2:
            raise FileNotFoundError(f"{path} not found")
        if code == 13:
            raise PermissionError(f"{path} is not readable")
        if code != 0:
            raise RuntimeError(f"reading {path} failed (exit {code}): {err.strip()[:500]}")
        if int(out.strip() or 0) > SandboxEnvironmentLimits.MAX_READ_FILE_SIZE:
            raise OutputLimitExceededError(limit_str=SandboxEnvironmentLimits.MAX_READ_FILE_SIZE_STR, truncated_output=None)
        data = await self.sandbox.files.read_bytes(path)
        if not text:
            return bytes(data)
        # newline="" semantics: bytes decoded as they are, \r\n kept.
        return bytes(data).decode("utf-8")
