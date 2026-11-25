"""The hosts an agent's sandbox may reach during the agent phase, by allowlist name.

configs/agent_allowlists.json maps a name to groups of hosts (hostnames, no wildcards: a
Kubernetes NetworkPolicy can't express them). A benchmark names its list in its runner entry
("agent_allowlist" in BENCHMARKS of scripts/harbor/run.py); a new list is a new key in that file.
The rationale of each list is in docs/harbor.md.

Harbor: scripts/harbor/task_patches.py (agent_egress_patch). EvalScope:
scripts/common/opensandbox_sandbox.py (install(agent_allowed_hosts=...)).
"""

import json
from pathlib import Path
from typing import Tuple

ALLOWLISTS_FILE = Path(__file__).resolve().parents[2] / "configs" / "agent_allowlists.json"


def allowed_hosts(name: str) -> Tuple[str, ...]:
    """The hosts of every group of the allowlist `name`, in file order."""
    lists = json.loads(ALLOWLISTS_FILE.read_text())
    if name not in lists:
        raise KeyError(f"no allowlist {name!r} in {ALLOWLISTS_FILE} (has: {', '.join(lists)})")
    return tuple(host for hosts in lists[name].values() for host in hosts)
