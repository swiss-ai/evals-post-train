"""What a SWE-bench agent's sandbox may reach: package indexes, not the code hosts with the fix.

SWE-bench's leaderboard checklist asks that a system either has no web access or "has taken
steps to prevent lookup of SWE-bench solutions via web-browsing" (SWE-bench/experiments,
checklist.md). Every task is a merged pull request of a public repository, so an agent with open
internet can fetch the fix from GitHub. During the agent phase the sandbox may only reach the
hosts below (the egress sidecar denies the rest, GitHub, GitLab and the Go module proxy
included): it can still install a package or a system tool, as a developer without the upstream
history would. Setting up the environment and scoring keep their network.

Not closed: the package indexes also serve later releases of the packages under test (Django on
PyPI, NodeBB on npm, ...), whose source contains the fix. The Go module proxy is left out for
that reason (it serves any GitHub repository at any commit); PyPI and npm are kept because
agents need them routinely.

Harbor: scripts/harbor/task_patches.py (patch_swe_bench_agent_egress). EvalScope:
scripts/common/opensandbox_sandbox.py (install(agent_allowed_hosts=...)).
"""

AGENT_ALLOWED_HOSTS = (
    # Python
    "pypi.org",
    "files.pythonhosted.org",
    "conda.anaconda.org",
    "repo.anaconda.com",
    # System packages (the SWE-bench images are Debian and Ubuntu based)
    "deb.debian.org",
    "security.debian.org",
    "archive.ubuntu.com",
    "security.ubuntu.com",
    # JavaScript (SWE-bench Pro)
    "registry.npmjs.org",
    "registry.yarnpkg.com",
)
