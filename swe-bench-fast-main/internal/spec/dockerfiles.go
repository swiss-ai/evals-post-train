package spec

import (
	"fmt"
	"strings"
)

// Upstream Miniconda version used by SWE-bench / Epoch AI images.
const condaVersion = "py311_23.11.0-2"

// GenerateBaseDockerfile creates the Dockerfile for the base layer:
// Ubuntu 22.04 + Miniconda + system dependencies.
// Matches upstream swebench _DOCKERFILE_BASE_PY exactly.
func GenerateBaseDockerfile(arch string) string {
	condaArch := "x86_64"
	if arch == "aarch64" || arch == "arm64" {
		condaArch = "aarch64"
	}

	platform := "linux/amd64"
	if condaArch == "aarch64" {
		platform = "linux/arm64"
	}

	var b strings.Builder

	b.WriteString(fmt.Sprintf("FROM --platform=%s ubuntu:22.04\n\n", platform))

	// System dependencies — matches upstream exactly
	b.WriteString("ARG DEBIAN_FRONTEND=noninteractive\n")
	b.WriteString("ENV TZ=Etc/UTC\n\n")
	b.WriteString("RUN apt update && apt install -y \\\n")
	b.WriteString("wget \\\n")
	b.WriteString("git \\\n")
	b.WriteString("build-essential \\\n")
	b.WriteString("libffi-dev \\\n")
	b.WriteString("libtiff-dev \\\n")
	b.WriteString("python3 \\\n")
	b.WriteString("python3-pip \\\n")
	b.WriteString("python-is-python3 \\\n")
	b.WriteString("jq \\\n")
	b.WriteString("curl \\\n")
	b.WriteString("locales \\\n")
	b.WriteString("locales-all \\\n")
	b.WriteString("tzdata \\\n")
	b.WriteString("&& rm -rf /var/lib/apt/lists/*\n\n")

	// Download and install conda — matches upstream Miniconda version
	b.WriteString(fmt.Sprintf("RUN wget 'https://repo.anaconda.com/miniconda/Miniconda3-%s-Linux-%s.sh' -O miniconda.sh \\\n", condaVersion, condaArch))
	b.WriteString("    && bash miniconda.sh -b -p /opt/miniconda3\n")
	b.WriteString("ENV PATH=/opt/miniconda3/bin:$PATH\n")
	b.WriteString("RUN conda init --all\n")
	b.WriteString("RUN conda config --set channel_priority flexible \\\n")
	b.WriteString("    && conda config --prepend channels defaults \\\n")
	b.WriteString("    && conda config --append channels conda-forge\n\n")

	b.WriteString("RUN adduser --disabled-password --gecos 'dog' nonroot\n\n")

	b.WriteString("WORKDIR /testbed\n")
	b.WriteString("CMD [\"/bin/bash\"]\n")

	return b.String()
}

// GenerateEnvDockerfile creates the Dockerfile for the env layer:
// FROM base, install conda env + pip packages.
// Matches upstream _DOCKERFILE_ENV_PY.
func GenerateEnvDockerfile(baseTag string) string {
	var b strings.Builder

	b.WriteString(fmt.Sprintf("FROM %s\n\n", baseTag))
	b.WriteString("COPY setup_env.sh /root/\n")
	b.WriteString("RUN sed -i -e 's/\\r$//' /root/setup_env.sh\n")
	b.WriteString("RUN chmod +x /root/setup_env.sh\n")
	b.WriteString("RUN /bin/bash -c \"source ~/.bashrc && /root/setup_env.sh\"\n\n")
	b.WriteString("WORKDIR /testbed/\n\n")
	// Automatically activate the testbed environment
	b.WriteString("RUN echo \"source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed\" > /root/.bashrc\n")

	return b.String()
}

// GenerateInstanceDockerfile creates the Dockerfile for the instance layer:
// FROM env, clone repo, checkout commit, run install.
// Matches upstream _DOCKERFILE_INSTANCE_PY.
func GenerateInstanceDockerfile(envTag string) string {
	var b strings.Builder

	b.WriteString(fmt.Sprintf("FROM %s\n\n", envTag))
	b.WriteString("COPY setup_repo.sh /root/\n")
	b.WriteString("RUN sed -i -e 's/\\r$//' /root/setup_repo.sh\n")
	b.WriteString("RUN /bin/bash /root/setup_repo.sh\n\n")
	b.WriteString("WORKDIR /testbed/\n")

	return b.String()
}
