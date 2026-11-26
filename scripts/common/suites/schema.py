"""Reading a suite file: its shape and what it may hold. Knows no framework's command line."""

import json
import re
import shlex
from pathlib import Path

from scripts.common import sampling, thinking

SUITES_DIR = Path(__file__).resolve().parents[3] / "configs" / "suites"
FRAMEWORKS = ("lm_eval", "inspect", "evalscope", "harbor", "tau2", "bfcl")
VIA = ("flag", "task")
LAUNCH_KEYS = ("model_path", "environment", "nodes", "time", "framework_args")
LAUNCH_OPTIONAL = ("env", "install")
# What scripts/direct_serving/job.py sets itself on the serving command.
LAUNCH_OWN_FLAGS = ("--model", "--served-model-name", "--host", "--port")
REPO_ROOT = Path(__file__).resolve().parents[3]


def _positive_int(where, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{where}: must be an integer >= 1 (got {value!r})")


THINKING_FLAGS = ("--thinking", "--enable-thinking")


# Flags the suite itself produces, so a setup may not list them.
RESERVED = {"lm_eval": {"--max-length", "--repeats", "--reasoning-effort"},
            "inspect": {"--task-arg", "--epochs", "--", "--reasoning-effort"},
            "evalscope": set(), "harbor": {"--reasoning-effort"}, "tau2": {"--reasoning-effort", "--sampling-profile"},
            "bfcl": {"--reasoning-effort", "--version", "--sampling-profile"}}


def _profile(where, name, profiles):
    if name not in profiles:
        raise ValueError(f"{where}: unknown sampling profile {name!r} ({', '.join(profiles)})")


def _launch(where, launch):
    if not isinstance(launch, dict) or not set(LAUNCH_KEYS) <= set(launch) <= {*LAUNCH_KEYS, *LAUNCH_OPTIONAL}:
        raise ValueError(f"{where}: launch must be an object with {', '.join(LAUNCH_KEYS)} and optionally {', '.join(LAUNCH_OPTIONAL)}")
    env = launch.get("env", {})
    if not isinstance(env, dict) or not all(
        re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(k)) and isinstance(v, str) for k, v in env.items()
    ):
        raise ValueError(f"{where}: launch.env must map variable names to string values")
    for key in ("model_path", "environment"):
        if not isinstance(launch[key], str) or not launch[key]:
            raise ValueError(f"{where}: launch.{key} must be a non-empty string")
    install = launch.get("install", [])
    if not isinstance(install, list) or not all(isinstance(f, str) for f in install):
        raise ValueError(f"{where}: launch.install must be a list of requirements files")
    for name in install:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path.parent != Path("requirements") or not (REPO_ROOT / path).is_file():
            raise ValueError(f"{where}: launch.install {name!r} must be a file directly in requirements/")
    _positive_int(f"{where} launch.nodes", launch["nodes"])
    if not isinstance(launch["time"], str) or not re.fullmatch(r"\d+:[0-5]\d:[0-5]\d", launch["time"]):
        raise ValueError(f"{where}: launch.time must be HH:MM:SS (got {launch['time']!r})")
    args = launch["framework_args"]
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise ValueError(f"{where}: launch.framework_args must be a list of strings")
    for flag in LAUNCH_OWN_FLAGS:
        if flag in args:
            raise ValueError(f"{where}: launch.framework_args must not hold {flag}: the launcher sets it")


def _tokenizer(where, spec):
    if not isinstance(spec, dict) or set(spec) != {"repo", "revision", "path"}:
        raise ValueError(f"{where}: tokenizer must be an object with repo, revision and path")
    if not isinstance(spec["repo"], str) or not spec["repo"]:
        raise ValueError(f"{where}: tokenizer.repo must be a git URL")
    if not isinstance(spec["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", spec["revision"]):
        raise ValueError(f"{where}: tokenizer.revision must be a full commit sha")
    path = Path(spec["path"]) if isinstance(spec["path"], str) else None
    if path is None or not spec["path"] or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{where}: tokenizer.path must be a relative directory inside the repository")


def _no_reserved(where, framework, setup, items):
    reserved = (RESERVED[framework] | ({"--tokenizer"} if framework == "lm_eval" and "tokenizer" in setup else set())) & set(items)
    if reserved:
        raise ValueError(f"{where} may not contain {sorted(reserved)[0]}: it comes from max_length, the benchmark's repeats or task_args")


def validate(suite):
    """Raise ValueError naming the broken field; return the suite otherwise."""
    profiles = sampling.read_config()["profiles"]
    for key in ("name", "title", "description", "benchmarks", "setups"):
        if key not in suite:
            raise ValueError(f"suite is missing {key!r}")
    if not isinstance(suite["benchmarks"], list) or not suite["benchmarks"]:
        raise ValueError("benchmarks: must be a non-empty list")
    groups = suite.get("groups", [])
    if not isinstance(groups, list) or not all(isinstance(g, str) for g in groups) or len(set(groups)) != len(groups):
        raise ValueError("groups: must be a list of unique strings")
    tasks, labels = set(), set()
    for i, bench in enumerate(suite["benchmarks"]):
        where = f"benchmarks[{i}]"
        for key in ("label", "task", "framework", "repeats", "metric"):
            if key not in bench:
                raise ValueError(f"{where}: missing {key!r}")
        for key in ("repeats", "metric"):
            if not isinstance(bench[key], dict):
                raise ValueError(f"{where}: {key} must be an object (got {bench[key]!r})")
        where = f"{where} ({bench['task']})"
        if bench["task"] in tasks or bench["label"] in labels:
            raise ValueError(f"{where}: task and label must be unique within the suite")
        tasks.add(bench["task"])
        labels.add(bench["label"])
        if bench["framework"] not in FRAMEWORKS:
            raise ValueError(f"{where}: framework must be one of {FRAMEWORKS} (got {bench['framework']!r})")
        _positive_int(f"{where} repeats.n", bench["repeats"].get("n"))
        if bench["repeats"].get("via") not in VIA:
            raise ValueError(f"{where}: repeats.via must be one of {VIA} (got {bench['repeats'].get('via')!r})")
        label = bench["metric"].get("label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"{where}: metric.label is required")
        if label.startswith("mean@") and label != f"mean@{bench['repeats']['n']}":
            raise ValueError(f"{where}: metric.label {label!r} does not match repeats.n {bench['repeats']['n']}")
        if "group" in bench and bench["group"] not in groups:
            raise ValueError(f"{where}: group {bench['group']!r} is not in the suite's groups {groups}")
        if "sampling" in bench:
            _profile(f"{where} sampling", bench["sampling"], profiles)
        if "task_args" in bench:
            args = bench["task_args"]
            if bench["framework"] != "inspect":
                raise ValueError(f"{where}: task_args is for the inspect framework only")
            if not isinstance(args, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in args.items()):
                raise ValueError(f"{where}: task_args must be an object of strings")
        if "flags" in bench:
            if not isinstance(bench["flags"], list) or not all(isinstance(f, str) for f in bench["flags"]):
                raise ValueError(f"{where}: flags must be a list of strings")
            _no_reserved(f"{where}: flags", bench["framework"], {}, bench["flags"])
        if "key" not in bench["metric"]:
            raise ValueError(f"{where}: metric.key is required (null if not checked yet)")
    if not suite["setups"]:
        raise ValueError("setups: must not be empty")
    for name, setup in suite["setups"].items():
        _positive_int(f"setup {name} max_length", setup.get("max_length"))
        _profile(f"setup {name} sampling", setup.get("sampling"), profiles)
        if "reasoning_effort" in setup and setup["reasoning_effort"] not in thinking.EFFORTS:
            raise ValueError(f"setup {name}: reasoning_effort must be one of {thinking.EFFORTS} (got {setup['reasoning_effort']!r})")
        if "tokenizer" in setup:
            _tokenizer(f"setup {name}", setup["tokenizer"])
        if "launch" in setup:
            _launch(f"setup {name}", setup["launch"])
        serving = setup.get("serving", [])
        if not isinstance(serving, list) or not all(isinstance(a, str) for a in serving):
            raise ValueError(f"setup {name}: serving must be a list of strings")
        flags = setup.get("flags")
        if not isinstance(flags, dict) or set(flags) - set(FRAMEWORKS):
            raise ValueError(f"setup {name}: flags must be an object keyed by framework ({', '.join(FRAMEWORKS)})")
        for framework in {b["framework"] for b in suite["benchmarks"]} - set(flags):
            raise ValueError(f"setup {name}: flags has no entry for {framework!r} (use [] for none)")
        for framework, items in flags.items():
            if not isinstance(items, list) or not all(isinstance(f, str) for f in items):
                raise ValueError(f"setup {name}: flags.{framework} must be a list of strings")
            _no_reserved(f"setup {name}: flags.{framework}", framework, setup, items)
        for bench in suite["benchmarks"]:
            profile = bench.get("sampling", setup["sampling"])
            row_flags = flags[bench["framework"]]
            if any(f in THINKING_FLAGS for f in row_flags) and profiles[profile].get("temperature") == 0:
                raise ValueError(f"setup {name}: thinking runs are not deterministic, but {bench['task']} samples with {profile!r}")
    return suite


def load(name_or_path):
    path = Path(name_or_path)
    if not path.suffix:
        path = SUITES_DIR / f"{name_or_path}.json"
    with open(path) as f:
        return validate(json.load(f))


def benchmark(suite, task):
    for bench in suite["benchmarks"]:
        if bench["task"] == task:
            return bench
    raise ValueError(f"{task!r} is not in suite {suite['name']!r} ({', '.join(b['task'] for b in suite['benchmarks'])})")


def setup(suite, name):
    if name not in suite["setups"]:
        raise ValueError(f"unknown setup {name!r} ({', '.join(suite['setups'])})")
    return suite["setups"][name]


def install_command(files):
    """The shell command that installs what `launch.install` names on a serving node: the
    requirement lines of those files (comments and blanks left out), without their dependencies
    (the serving image has them) and over the image's copy. Empty for no files."""
    specs = [
        line.strip()
        for name in files
        for line in (REPO_ROOT / name).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not specs:
        return ""
    return "pip install -q --no-deps --force-reinstall --ignore-requires-python " + shlex.join(specs)


def launch_for(suite, setup_name):
    """How a setup's model is served, for scripts/direct_serving/job.py: its launch block, the
    EDF as an absolute path and the setup's `serving` arguments appended to the framework
    arguments (a string), and `pre_launch_cmds`: what its `install` files need on every serving node
    (empty without any). ValueError when the setup has no launch block."""
    chosen = setup(suite, setup_name)
    if "launch" not in chosen:
        raise ValueError(f"setup {setup_name!r} has no launch block")
    launch = chosen["launch"]
    return {
        **launch,
        "pre_launch_cmds": install_command(launch.get("install", [])),
        "environment": str(REPO_ROOT / launch["environment"]),
        "framework_args": shlex.join([*launch["framework_args"], *chosen.get("serving", [])]),
    }
