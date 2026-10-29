"""Harbor's Docker environment, with its egress-control sidecar built for DOCKER_DEFAULT_PLATFORM.

Harbor builds the sidecar that enforces network allowlists (Senior SWE-bench) for the Docker
daemon's native platform, but `docker compose` starts every service of the trial for
DOCKER_DEFAULT_PLATFORM when that is set. With the linux/amd64 the runner sets on arm64 hosts
(the task images are amd64), compose finds only an arm64 sidecar image and the trial fails
("does not provide the specified platform"). Build the sidecar for that platform too; it then
runs emulated like the task container. Without DOCKER_DEFAULT_PLATFORM nothing changes.
"""

from __future__ import annotations

import os

from harbor.environments.docker.docker import DockerEnvironment
from harbor.environments.docker.utils import default_docker_platform, ensure_docker_image_built


class PlatformDockerEnvironment(DockerEnvironment):
    async def _ensure_egress_control_sidecar_image_built(self) -> None:
        self._env_vars.egress_control_sidecar_image_name = await ensure_docker_image_built(
            docker_name=self._EGRESS_CONTROL_SIDECAR_DOCKER_NAME,
            docker_build_context=self._EGRESS_CONTROL_SIDECAR_CONTEXT_PATH,
            dockerfile_path=self._egress_control_sidecar_dockerfile_path(),
            build_args={},
            platform=os.environ.get("DOCKER_DEFAULT_PLATFORM") or await default_docker_platform(),
            logger=self.logger,
        )
