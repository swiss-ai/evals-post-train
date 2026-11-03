"""Dockerfile replay for Harbor environments that cannot build images (--sandbox k8s, opensandbox).

A task with only environment/Dockerfile (SWE-bench's FROM <prebuilt image> plus a few setup
lines, or a from-scratch toolchain) is started from its FROM image, and the Dockerfile's steps are
replayed in the running sandbox: RUN steps in order, with ENV, ARG, WORKDIR, SHELL and USER applied
as a build would, and COPY/ADD sources uploaded from the task's environment dir (as root, like a
build's). The result is the filesystem a build makes. With [environment].docker_image the image
runs as is, unless the trial forces a build and there is a Dockerfile.

Multi-stage Dockerfiles (Multi-SWE-bench: a helper stage fetches a standalone Python that the last
stage copies in): first, each stage that a COPY --from names is replayed in a sandbox of its own
(COPY --from=<image> runs the image as is), the paths it copies are packed and read out, and that
sandbox is deleted; then the last stage is replayed in the trial's sandbox with those paths unpacked
in place. A trial so holds one sandbox at a time: on a full cluster, trials never sit on a sandbox
while waiting for another. COPY --from paths are substituted with the Dockerfile's own ARG/ENV
values (not the base image's ENV). A stage built FROM an earlier stage is rejected: use
--sandbox docker.

The replay only uses the environment's own exec (as root), upload_file and download_file, and runs
each helper stage in a second environment of the same type, so every such environment shares it.
It runs before the trial's network policy applies, as a Docker build has network.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shlex
import shutil
import tarfile
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from harbor.environments.base import BaseEnvironment

# Instructions that only describe the image (ports, labels, the default process): a replay has
# nothing to do for them.
_IGNORED_INSTRUCTIONS = {"CMD", "ENTRYPOINT", "EXPOSE", "HEALTHCHECK", "LABEL", "MAINTAINER", "ONBUILD", "STOPSIGNAL", "VOLUME"}
_ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")
# The environment a command in the sandbox starts with: the image's ENV plus the sandbox's own.
_READ_ENV = "cat /proc/self/environ"


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


def dockerfile_stages(dockerfile: Path) -> List[Dict[str, Any]]:
    """The Dockerfile's build stages in order, as {"name", "steps"}: a stage's instructions
    from its FROM on, after the ARGs declared before the first FROM (only FROM lines see
    those). The last stage is the image. Refuses what a replay cannot reproduce."""
    from dockerfile_parse import DockerfileParser

    steps = [
        {"instruction": item["instruction"].upper(), "value": item["value"]}
        for item in DockerfileParser(path=str(dockerfile)).structure
        if item["instruction"].upper() != "COMMENT"
    ]
    first = next((i for i, step in enumerate(steps) if step["instruction"] == "FROM"), None)
    if first is None:
        raise RuntimeError(f"{dockerfile}: no FROM")
    stages: List[Dict[str, Any]] = []
    for step in steps[first:]:
        if step["instruction"] != "FROM":
            stages[-1]["steps"].append(step)
            continue
        words = [t for t in step["value"].split() if not t.startswith("--")]
        name = words[2].lower() if len(words) == 3 and words[1].lower() == "as" else None
        if words[0].lower() in {stage["name"] for stage in stages}:
            raise RuntimeError(
                f"{dockerfile}: FROM {words[0]} builds on an earlier stage, which the replay "
                "cannot reproduce: use --sandbox docker"
            )
        stages.append({"name": name, "steps": [*steps[:first], step]})
    return stages


def from_image(steps: List[Dict[str, str]], dockerfile: Path) -> str:
    """The image a stage starts FROM, with the ARGs declared before it substituted."""
    args: Dict[str, str] = {}
    for step in steps:
        if step["instruction"] == "ARG":
            args.update(parse_assignments(step["value"], args))
        elif step["instruction"] == "FROM":
            tokens = [t for t in step["value"].split() if not t.startswith("--")]
            return substitute(tokens[0], args)
    raise RuntimeError(f"{dockerfile}: no FROM")


def copies_into(dest: str, n_sources: int) -> bool:
    """Whether COPY puts its sources into dest (a directory) rather than at dest itself."""
    return dest.endswith("/") or n_sources > 1 or PurePosixPath(dest).name in ("", ".", "..")


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


def pack_tree(root: Path, keep: str) -> bytes:
    """root's contents as a tar.gz owned by root (as COPY makes them), without entries for keep
    and its parent directories: those may exist in the sandbox already (/, /tmp, /usr/local), and
    unpacking an entry for one would reset its owner and mode."""
    kept = {str(PurePosixPath(keep)), *(str(parent) for parent in PurePosixPath(keep).parents)}

    def as_root(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = "root"
        return info

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        for directory, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for name in [*dirnames, *sorted(filenames)]:
                path = Path(directory, name)
                arcname = path.relative_to(root).as_posix()
                if path.is_dir() and not path.is_symlink() and "/" + arcname in kept:
                    continue
                archive.add(path, arcname=arcname, recursive=False, filter=as_root)
    return buf.getvalue()


@dataclass
class ImageConfig:
    """What a build records in the image besides its files: ENV, WORKDIR and USER."""

    env: Dict[str, str] = field(default_factory=dict)
    workdir: Optional[str] = None
    user: Optional[str] = None


@contextlib.asynccontextmanager
async def stage_environment(environment: BaseEnvironment, image: str) -> AsyncIterator[BaseEnvironment]:
    """A started environment of the same type running image as is, deleted on exit."""
    stage = type(environment)(
        environment.environment_dir,
        environment.environment_name,
        f"{environment.session_id}-s{uuid.uuid4().hex[:4]}",
        environment.trial_paths,
        environment.task_env_config.model_copy(update={"docker_image": image}),
        logger=environment.logger,
    )
    try:
        await stage.start(force_build=False)
        yield stage
    finally:
        await stage.stop(delete=True)


class DockerfileReplay:
    """The replay for one environment's task; nothing to do when it runs a prebuilt image.

    Use: export_stages() before the trial's sandbox exists, start it from .image, then replay()."""

    def __init__(self, environment: BaseEnvironment, force_build: bool) -> None:
        self.environment = environment
        self.dockerfile = environment.environment_dir / "Dockerfile"
        image = environment.task_env_config.docker_image
        build = not image or (force_build and self.dockerfile.is_file())
        self.stages = dockerfile_stages(self.dockerfile) if build else []
        self.image = from_image(self.stages[-1]["steps"], self.dockerfile) if self.stages else image
        self.timeout = int(environment.task_env_config.build_timeout_sec or 1800)

    async def export_stages(self) -> Dict[int, bytes]:
        """The last stage's COPY --from paths, packed by step index."""
        return await self._export(self.stages[-1]["steps"], self.stages[:-1]) if self.stages else {}

    async def replay(self, exports: Dict[int, bytes]) -> ImageConfig:
        """Replays the last stage in the environment's sandbox, started from .image."""
        if not self.stages:
            return ImageConfig()
        return await self._replay(self.environment, self.stages[-1]["steps"], exports)

    async def _replay(self, target: BaseEnvironment, steps: List[Dict[str, str]], exports: Dict[int, bytes]) -> ImageConfig:
        result = await target.exec(_READ_ENV, cwd="/", user="root", timeout_sec=60)
        base_env = dict(item.split("=", 1) for item in result.stdout.split("\0") if "=" in item) if result.return_code == 0 else {}
        image = ImageConfig()
        args: Dict[str, str] = {}
        shell = ["/bin/sh", "-c"]
        before_from = True
        for index, step in enumerate(steps):
            kind, value = step["instruction"], step["value"]
            scope = {**base_env, **args, **image.env}
            if kind == "FROM":
                before_from = False
            elif kind == "ARG":
                # ARGs declared before FROM are only visible to FROM (and to a later bare ARG).
                declared = parse_assignments(value, scope)
                args.update({k: v for k, v in declared.items() if not before_from})
            elif kind == "ENV":
                image.env.update(parse_assignments(value, scope))
            elif kind == "WORKDIR":
                path = substitute(value.strip(), scope)
                image.workdir = str(PurePosixPath(image.workdir or "/") / path)
                await self._run(target, ["mkdir", "-p", image.workdir], 60, step)
            elif kind == "SHELL":
                shell = json.loads(value)
            elif kind == "USER":
                image.user = substitute(value.strip(), scope).split(":", 1)[0]
            elif kind == "RUN":
                argv = json.loads(value) if value.lstrip().startswith("[") else [*shell, value]
                await self._run(target, argv, self.timeout, step, image.workdir, {**args, **image.env}, image.user)
            elif kind in ("COPY", "ADD"):
                await self._copy(target, value, image.workdir or "/", scope, kind == "ADD", step, exports.get(index))
            elif kind not in _IGNORED_INSTRUCTIONS:
                raise RuntimeError(f"{self.dockerfile}: {kind} is not supported by the replay")
        return image

    async def _run(
        self,
        target: BaseEnvironment,
        argv: List[str],
        timeout: int,
        step: Dict[str, str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
    ) -> None:
        command = shlex.join(in_dir(as_user(argv, user), cwd, env or {}))
        result = await target.exec(command, cwd="/", user="root", timeout_sec=timeout)
        if result.return_code != 0:
            tail = ((result.stdout or "") + (result.stderr or "")).strip()[-1500:]
            raise RuntimeError(
                f"{target.session_id}: {step['instruction']} {step['value'][:200]} failed (exit {result.return_code}): {tail}"
            )

    async def _put_tar(self, target: BaseEnvironment, data: bytes, target_dir: str, step: Dict[str, str]) -> None:
        """Unpacks a tar.gz into target_dir in the sandbox, as root (owners and symlinks kept)."""
        # At / and not in /tmp: an upload creates the file's parent directory, which could
        # change /tmp's mode.
        remote = f"/.sandbox-replay-{uuid.uuid4().hex[:12]}.tgz"
        with tempfile.TemporaryDirectory() as staging:
            local = Path(staging, "upload.tgz")
            local.write_bytes(data)
            await target.upload_file(local, remote)
        directory = shlex.quote(target_dir)
        await self._run(target, ["sh", "-c", f"mkdir -p {directory} && tar xzf {remote} -C {directory} && rm -f {remote}"], 600, step)

    async def _copy(
        self,
        target: BaseEnvironment,
        value: str,
        workdir: str,
        scope: Dict[str, str],
        is_add: bool,
        step: Dict[str, str],
        packed: Optional[bytes],
    ) -> None:
        tokens = shlex.split(value) if not value.lstrip().startswith("[") else json.loads(value)
        chown = next((t.split("=", 1)[1] for t in tokens if t.startswith("--chown=")), None)
        chmod = next((t.split("=", 1)[1] for t in tokens if t.startswith("--chmod=")), None)
        paths = [substitute(t, scope) for t in tokens if not t.startswith("--")]
        *sources, dest = paths
        dest_abs = str(PurePosixPath(workdir) / dest)
        into_dir = dest.endswith("/") or len(sources) > 1
        if packed is not None:
            # COPY --from, packed by _export; unpacked at the destination (not at /, so a
            # symlinked directory such as /bin stays a symlink).
            target_path = PurePosixPath(dest_abs)
            await self._put_tar(target, packed, str(target_path if copies_into(dest, len(sources)) else target_path.parent), step)
            sources = []
        with tempfile.TemporaryDirectory() as staging:
            root = Path(staging)
            for source in sources:
                if is_add and re.match(r"https?://", source):
                    url_target = f"{dest_abs.rstrip('/')}/{PurePosixPath(source).name}" if into_dir else dest_abs
                    fetch = f"mkdir -p \"$(dirname {shlex.quote(url_target)})\" && curl -fsSL -o {shlex.quote(url_target)} {shlex.quote(source)}"
                    await self._run(target, ["sh", "-c", fetch], 600, step)
                    continue
                environment_dir = self.environment.environment_dir
                matches = sorted(environment_dir.glob(source)) or [environment_dir / source]
                for match in matches:
                    if not match.exists():
                        raise RuntimeError(f"{self.dockerfile}: {step['instruction']} source {source} not found")
                    if match.is_dir():
                        shutil.copytree(match, root / dest_abs.lstrip("/"), dirs_exist_ok=True, symlinks=True)
                    elif is_add and match.name.endswith(_ARCHIVE_SUFFIXES):
                        extracted = root / dest_abs.lstrip("/")
                        extracted.mkdir(parents=True, exist_ok=True)
                        with tarfile.open(match) as archive:
                            archive.extractall(extracted, filter="tar")
                    else:
                        name = dest_abs.lstrip("/")
                        file_target = root / name / match.name if (into_dir or dest_abs.endswith("/")) else root / name
                        file_target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(match, file_target)
            if any(root.iterdir()):
                await self._put_tar(target, pack_tree(root, dest_abs), "/", step)
        for flag, tool in ((chown, "chown"), (chmod, "chmod")):
            if flag:
                await self._run(target, [tool, "-R", flag, dest_abs], 120, step)

    async def _export(self, steps: List[Dict[str, str]], earlier: List[Dict[str, Any]]) -> Dict[int, bytes]:
        """For each COPY --from in steps (by index): the paths it copies, packed in the stage
        it names (one of earlier, by name or number) or in that image, laid out as they land
        under the destination. Each source runs in a sandbox of its own, deleted before the next."""
        copies: Dict[str, List[Tuple[int, List[str], str]]] = {}
        scope: Dict[str, str] = {}
        before_from = True
        for index, step in enumerate(steps):
            kind, value = step["instruction"], step["value"]
            if kind == "FROM":
                before_from = False
            elif kind == "ARG":
                scope.update({k: v for k, v in parse_assignments(value, scope).items() if not before_from})
            elif kind == "ENV":
                scope.update(parse_assignments(value, scope))
            elif kind in ("COPY", "ADD"):
                tokens = shlex.split(value) if not value.lstrip().startswith("[") else json.loads(value)
                ref = next((substitute(t.split("=", 1)[1], scope) for t in tokens if t.startswith("--from=")), None)
                if ref:
                    *sources, dest = [substitute(t, scope) for t in tokens if not t.startswith("--")]
                    copies.setdefault(ref.lower(), []).append((index, sources, dest))
        exports: Dict[int, bytes] = {}
        names = [stage["name"] for stage in earlier]
        for ref, items in copies.items():
            number = int(ref) if ref.isdigit() else (names.index(ref) if ref in names else None)
            if number is not None and number >= len(earlier):
                raise RuntimeError(f"{self.dockerfile}: COPY --from={ref}: no such earlier stage")
            stage_steps = earlier[number]["steps"] if number is not None else []
            inner = await self._export(stage_steps, earlier[:number]) if stage_steps else {}
            image = from_image(stage_steps, self.dockerfile) if stage_steps else ref
            async with stage_environment(self.environment, image) as stage:
                workdir = (await self._replay(stage, stage_steps, inner)).workdir if stage_steps else None
                for index, sources, dest in items:
                    exports[index] = await self._pack(stage, workdir, sources, dest, steps[index])
        return exports

    async def _pack(
        self, target: BaseEnvironment, workdir: Optional[str], sources: List[str], dest: str, step: Dict[str, str]
    ) -> bytes:
        """sources (relative to workdir) as a tar.gz laid out as they land: in the destination
        directory, or, for a single source to a path, under the destination's name."""
        staging = f"/tmp/.sandbox-stage-{uuid.uuid4().hex[:8]}"
        into = staging if copies_into(dest, len(sources)) else f"{staging}/{shlex.quote(PurePosixPath(dest).name)}"
        script = [f"mkdir -p {staging}"]
        for source in sources:
            src = shlex.quote(str(PurePosixPath(workdir or "/") / source))
            # A directory's contents are copied, a file itself.
            script.append(f"if [ -d {src} ]; then mkdir -p {into} && cp -a {src}/. {into}/; else cp -a {src} {into}; fi")
        script.append(f"tar czf {staging}.tgz -C {staging} .")
        await self._run(target, ["sh", "-c", " && ".join(script)], 600, step)
        with tempfile.TemporaryDirectory() as local:
            path = Path(local, "stage.tgz")
            await target.download_file(f"{staging}.tgz", path)
            return path.read_bytes()
