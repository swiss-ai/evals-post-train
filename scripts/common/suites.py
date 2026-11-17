"""Suites: the benchmarks of a comparison and the setups (window + launcher flags) a model can run under.

configs/suites/<name>.json holds two things:

  benchmarks  rows in table order: label, task, framework, repeats {n, via}, metric {label, key},
              optionally sampling (a profile name that overrides the setup's) and task_args
              (inspect only: -T arguments of the task, e.g. {"cot": "true"}) and flags (launcher
              flags of this row under every setup, e.g. a sandbox or judge roles)
  setups      named kinds of model setup: max_length, sampling (a profile name), flags (the
              launcher flags of each framework the suite uses: thinking, chat format, ...) and
              optionally serving (the arguments the server itself needs, e.g. a reasoning parser;
              nothing here runs them, whoever launches the model passes them to `vllm serve`) and
              tokenizer {repo, revision, path} (a tokenizer pinned in a git repository; lm_eval
              gets --tokenizer <local copy>, which `flags` fetches once into the cache) and
              task_flags {task: [flags]} (replaces the setup's flags of that one task, e.g. the
              chat-completions endpoint that native function calling needs)

`run` starts every (model, benchmark) pair as its own launch, so each one is its own Slurm job (lm_eval: the
launcher's `single` mode; inspect: `sbatch run_inspect_eval.sbatch`). --model SETUP=SERVED_NAME names a
setup and the name its model is served under, repeatable. Without --submit it only prints the commands.

A setup's reasoning_effort (optional) becomes --reasoning-effort <level> on every lm_eval and inspect row
run under it.

Profiles are the ones of configs/sampling_defaults.json; a suite names one and holds no temperature.

`repeats.via` says who repeats: "flag" (lm_eval gets --repeats n, inspect gets -- --epochs n) or
"task" (the task already repeats n times, nothing is passed). max_length is the lm_eval --max-length;
inspect has no such flag, the server's --max-model-len is the window. The code only reads the file and knows no model.

    python -m scripts.common.suites list            # names, e.g. apertus2/chonk_sft
    python -m scripts.common.suites show apertus2/chonk_sft    # title and setups
    python -m scripts.common.suites plan apertus2/chonk_sft --setup apertus-1.5-70b-thinking
    python -m scripts.common.suites flags apertus2/chonk_sft --setup apertus-1.5-70b-thinking --task ifbench
    python -m scripts.common.suites run apertus2/chonk_sft --api-base-url <url> \\
        --model chonk-lc-16k-v0.2-sft=<served name> --model qwen3-32b-thinking=<served name> [--submit]
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts.common import sampling, thinking

SUITES_DIR = Path(__file__).resolve().parents[2] / "configs" / "suites"
FRAMEWORKS = ("lm_eval", "inspect", "evalscope", "harbor", "tau2")
VIA = ("flag", "task")


def _positive_int(where, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{where}: must be an integer >= 1 (got {value!r})")


THINKING_FLAGS = ("--thinking", "--enable-thinking")
# Flags the suite itself produces, so a setup may not list them.
RESERVED = {"lm_eval": {"--max-length", "--repeats", "--reasoning-effort"},
            "inspect": {"--task-arg", "--epochs", "--", "--reasoning-effort"},
            "evalscope": set(), "harbor": {"--reasoning-effort"}, "tau2": set()}


def _profile(where, name, profiles):
    if name not in profiles:
        raise ValueError(f"{where}: unknown sampling profile {name!r} ({', '.join(profiles)})")


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
        task_flags = setup.get("task_flags", {})
        if not isinstance(task_flags, dict) or set(task_flags) - tasks:
            raise ValueError(f"setup {name}: task_flags must be an object keyed by a task of the suite")
        for task, items in task_flags.items():
            if not isinstance(items, list) or not all(isinstance(f, str) for f in items):
                raise ValueError(f"setup {name}: task_flags.{task} must be a list of strings")
            framework = next(b["framework"] for b in suite["benchmarks"] if b["task"] == task)
            _no_reserved(f"setup {name}: task_flags.{task}", framework, setup, items)
        for bench in suite["benchmarks"]:
            profile = bench.get("sampling", setup["sampling"])
            row_flags = task_flags.get(bench["task"], flags[bench["framework"]])
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


def flags_for(suite, setup_name, task, fetch=False):
    """Launcher flags for one benchmark of the suite under one setup, for the benchmark's framework.

    A setup's pinned tokenizer becomes lm_eval's --tokenizer <local copy>; fetch=True downloads the
    copy first (once), otherwise the flag names where it will be."""
    bench, chosen = benchmark(suite, task), setup(suite, setup_name)
    flags = list(chosen.get("task_flags", {}).get(task, chosen["flags"][bench["framework"]]))
    if "reasoning_effort" in chosen and bench["framework"] in ("lm_eval", "inspect", "harbor"):
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


REPO_ROOT = Path(__file__).resolve().parents[2]
LM_EVAL = ["bash", "scripts/lm_eval_harness/launch_evaluations.sh", "single"]
INSPECT = "scripts/inspect_ai/run_inspect_eval.sbatch"
HARBOR = "scripts/harbor/run_harbor_eval.sbatch"


def commands(suite, models, api_base_url=None, api_model_name=None, limit=None, no_wandb=False,
             qos=None, partition=None, fetch=False):
    """One launch per (model, benchmark), models in the order given, benchmarks in table order.

    `models` maps a setup name to the name the model is served under. Every launch is its own Slurm job
    (lm_eval: one `single` launch; inspect and harbor: one sbatch). api_model_name only applies to lm_eval, which
    takes the served name separately from the model; qos and partition go to the lm_eval launcher or to
    sbatch. Returns [{"setup", "label", "framework", "argv"}]."""
    rows = []
    for setup_name, served in models.items():
        for bench in suite["benchmarks"]:
            framework = bench["framework"]
            flags = flags_for(suite, setup_name, bench["task"], fetch=fetch)
            common = ["--task", bench["task"], "--model", served]
            if api_base_url:
                common += ["--api-base-url", api_base_url]
            if limit:
                common += ["--limit", str(limit)]
            if no_wandb:
                common += ["--no-wandb"]
            if framework == "lm_eval":
                if api_base_url:
                    common += ["--api-model-name", api_model_name or served]
                    if "--backend" not in flags:
                        common += ["--backend", "openai"]
                scheduling = (["--qos", qos] if qos else []) + (["--partition", partition] if partition else [])
                argv = [*LM_EVAL[:3], *common[:2], *common[2:], *scheduling, *flags]
            elif framework == "inspect":
                scheduling = ([f"--qos={qos}"] if qos else []) + ([f"--partition={partition}"] if partition else [])
                argv = ["sbatch", *scheduling, INSPECT, *common, *flags]
            elif framework == "harbor":
                scheduling = ([f"--qos={qos}"] if qos else []) + ([f"--partition={partition}"] if partition else [])
                argv = ["sbatch", *scheduling, HARBOR, "--benchmark", bench["task"], *common[2:], *flags]
            else:
                raise ValueError(f"{bench['task']}: no launcher for the {framework} framework yet")
            rows.append({"setup": setup_name, "label": bench["label"], "framework": framework, "argv": argv})
    return rows


def sampling_for(suite, setup_name, task):
    """The sampling profile name of one benchmark under one setup."""
    return benchmark(suite, task).get("sampling", setup(suite, setup_name)["sampling"])


def plan(suite, setup_name):
    """One row per benchmark: what a run under this setup is and what it reports."""
    return [{"label": b["label"], "task": b["task"], "framework": b["framework"],
             "flags": flags_for(suite, setup_name, b["task"]),
             "sampling": sampling_for(suite, setup_name, b["task"]), "metric": b["metric"]["label"],
             "key": b["metric"]["key"]} for b in suite["benchmarks"]]


def _table(rows, headers):
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    return "\n".join("  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip() for r in [headers] + rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    show = sub.add_parser("show")
    show.add_argument("suite")
    plan_p = sub.add_parser("plan")
    plan_p.add_argument("suite")
    plan_p.add_argument("--setup", required=True)
    flags_p = sub.add_parser("flags")
    flags_p.add_argument("suite")
    flags_p.add_argument("--setup", required=True)
    flags_p.add_argument("--task", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("suite")
    run_p.add_argument("--model", action="append", required=True, metavar="SETUP=SERVED_NAME")
    run_p.add_argument("--api-base-url")
    run_p.add_argument("--api-model-name")
    run_p.add_argument("--limit", type=int)
    run_p.add_argument("--no-wandb", action="store_true")
    run_p.add_argument("--qos")
    run_p.add_argument("--partition")
    run_p.add_argument("--submit", action="store_true", help="run the commands (default: print them)")
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            for path in sorted(SUITES_DIR.rglob("*.json")):
                print(path.relative_to(SUITES_DIR).with_suffix(""))
        elif args.command == "show":
            suite = load(args.suite)
            print(f"{suite['name']}: {suite['title']}\n{suite['description']}\n")
            for name, s in suite["setups"].items():
                effort = f", reasoning effort {s['reasoning_effort']}" if "reasoning_effort" in s else ""
                print(f"{name}: window {s['max_length']}, sampling {s['sampling']}{effort}\n  {s.get('description', '')}")
                for framework, items in s["flags"].items():
                    print(f"  {framework}: {shlex.join(items) or '(none)'}")
                if s.get("tokenizer"):
                    t = s["tokenizer"]
                    print(f"  tokenizer: {t['repo']}@{t['revision'][:12]} {t['path']}")
                if s.get("serving"):
                    print(f"  server: {shlex.join(s['serving'])}")
        elif args.command == "plan":
            suite = load(args.suite)
            print(_table([[r["label"], r["task"], shlex.join(r["flags"]), r["sampling"], r["metric"],
                           r["key"] or "(unchecked)"] for r in plan(suite, args.setup)],
                         ["label", "task", "flags", "sampling", "reports", "results key"]))
        elif args.command == "run":
            suite = load(args.suite)
            models = {}
            for item in args.model:
                setup_name, _, served = item.partition("=")
                if not served or setup_name in models:
                    raise ValueError(f"--model must be SETUP=SERVED_NAME, once per setup (got {item!r})")
                setup(suite, setup_name)
                models[setup_name] = served
            rows = commands(suite, models, args.api_base_url, args.api_model_name, args.limit, args.no_wandb,
                            args.qos, args.partition, fetch=args.submit)
            print(f"{len(rows)} launches: {len(models)} model{'s' * (len(models) != 1)} x "
                  f"{len(suite['benchmarks'])} benchmarks", file=sys.stderr)
            for row in rows:
                print(shlex.join(row["argv"]))
                if args.submit:
                    done = subprocess.run(row["argv"], cwd=REPO_ROOT)
                    if done.returncode:
                        raise ValueError(f"{row['setup']} / {row['label']} failed ({done.returncode}); the later launches were not started")
        else:
            print(shlex.join(flags_for(load(args.suite), args.setup, args.task, fetch=True)))
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
