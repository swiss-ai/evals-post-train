"""What a setup means for one benchmark: its launcher flags, sampling profile and pinned tokenizer."""

import os
import subprocess
import tempfile
from pathlib import Path

from scripts.common.suites.schema import benchmark, setup


def _checkout_dir(spec):
    root = os.environ.get("EVAL_ENV_CACHE_ROOT") or f"{os.environ.get('SCRATCH') or '/tmp'}/eval-envs"
    return Path(root) / "tokenizers" / spec["revision"]


def tokenizer_dir(spec):
    """Where the pinned tokenizer lives once fetched (under the cache the environments use)."""
    return _checkout_dir(spec) / spec["path"]


def fetch_tokenizer(spec):
    """Fetch the pinned tokenizer directory into the cache (once) and return its path."""
    final = _checkout_dir(spec)
    if (final / ".complete").exists():
        return tokenizer_dir(spec)
    final.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    with tempfile.TemporaryDirectory(dir=final.parent, prefix=".fetch-") as work:
        src = Path(work) / "src"
        for step in (["init", "-q", str(src)],
                     ["-C", str(src), "fetch", "-q", "--depth", "1", spec["repo"], spec["revision"]],
                     ["-C", str(src), "sparse-checkout", "set", spec["path"]],
                     ["-C", str(src), "checkout", "-q", "FETCH_HEAD"]):
            done = subprocess.run(["git", *step], env=env, capture_output=True, text=True)
            if done.returncode:
                raise ValueError(f"could not fetch the tokenizer {spec['repo']}@{spec['revision'][:12]} "
                                 f"(needs read access): {done.stderr.strip()}")
        head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head != spec["revision"] or not (src / spec["path"]).is_dir():
            raise ValueError(f"the fetched tokenizer is not {spec['revision']}:{spec['path']}")
        (src / ".complete").touch()
        # Another launch may have fetched it meanwhile: keep that one.
        if not final.exists():
            src.rename(final)
    return tokenizer_dir(spec)


def flags_for(suite, setup_name, task, fetch=False):
    """Launcher flags for one benchmark of the suite under one setup, for the benchmark's framework.

    A setup's pinned tokenizer becomes lm_eval's --tokenizer <local copy>; fetch=True downloads the
    copy first (once), otherwise the flag names where it will be."""
    bench, chosen = benchmark(suite, task), setup(suite, setup_name)
    flags = list(chosen["flags"][bench["framework"]])
    if "reasoning_effort" in chosen and bench["framework"] in ("lm_eval", "inspect", "evalscope", "harbor", "bfcl", "tau2"):
        flags += ["--reasoning-effort", chosen["reasoning_effort"]]
    flags += bench.get("flags", [])
    repeat = bench["repeats"]["via"] == "flag"
    if bench["framework"] == "lm_eval":
        if "tokenizer" in chosen:
            flags += ["--tokenizer", str(fetch_tokenizer(chosen["tokenizer"]) if fetch else tokenizer_dir(chosen["tokenizer"]))]
        flags += ["--max-length", str(chosen["max_length"])]
        if repeat:
            flags += ["--repeats", str(bench["repeats"]["n"])]
    elif bench["framework"] == "inspect":
        for key, value in bench.get("task_args", {}).items():
            flags += ["--task-arg", f"{key}={value}"]
        if repeat:
            flags += ["--", "--epochs", str(bench["repeats"]["n"])]
    elif repeat:
        raise ValueError(f"{bench['task']}: repeats.via 'flag' is not defined for {bench['framework']}")
    return flags


def sampling_for(suite, setup_name, task):
    """The sampling profile name of one benchmark under one setup."""
    return benchmark(suite, task).get("sampling", setup(suite, setup_name)["sampling"])
