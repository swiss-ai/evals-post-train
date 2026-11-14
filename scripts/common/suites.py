"""Suites: the benchmarks of a comparison and the setups (window + launcher flags) a model can run under.

configs/suites/<name>.json holds two things:

  benchmarks  rows in table order: label, task, framework, repeats {n, via}, metric {label, key},
              optionally sampling (a profile name that overrides the setup's) and task_args
              (inspect only: -T arguments of the task, e.g. {"cot": "true"})
  setups      named kinds of model setup: max_length, sampling (a profile name) and flags, the
              launcher flags of each framework the suite uses (thinking, chat format, ...)

Profiles are the ones of configs/sampling_defaults.json; a suite names one and holds no temperature.

`repeats.via` says who repeats: "flag" (lm_eval gets --repeats n, inspect gets -- --epochs n) or
"task" (the task already repeats n times, nothing is passed). max_length is the lm_eval --max-length;
inspect has no such flag, the server's --max-model-len is the window. The code only reads the file and knows no model.

    python -m scripts.common.suites list            # names, e.g. table25 or apertus2/chonk_sft
    python -m scripts.common.suites show table25    # title and setups
    python -m scripts.common.suites plan table25 --setup apertus-1.5-70b-thinking
    python -m scripts.common.suites flags table25 --setup apertus-1.5-70b-thinking --task ifbench
"""

import argparse
import json
import shlex
import sys
from pathlib import Path

from scripts.common import sampling

SUITES_DIR = Path(__file__).resolve().parents[2] / "configs" / "suites"
FRAMEWORKS = ("lm_eval", "inspect", "evalscope", "harbor", "tau2")
VIA = ("flag", "task")


def _positive_int(where, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{where}: must be an integer >= 1 (got {value!r})")


THINKING_FLAGS = ("--thinking", "--enable-thinking")
# Flags the suite itself produces, so a setup may not list them.
RESERVED = {"lm_eval": {"--max-length", "--repeats"}, "inspect": {"--task-arg", "--epochs", "--"},
            "evalscope": set(), "harbor": set(), "tau2": set()}


def _profile(where, name, profiles):
    if name not in profiles:
        raise ValueError(f"{where}: unknown sampling profile {name!r} ({', '.join(profiles)})")


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
        if "key" not in bench["metric"]:
            raise ValueError(f"{where}: metric.key is required (null if not checked yet)")
    if not suite["setups"]:
        raise ValueError("setups: must not be empty")
    for name, setup in suite["setups"].items():
        _positive_int(f"setup {name} max_length", setup.get("max_length"))
        _profile(f"setup {name} sampling", setup.get("sampling"), profiles)
        flags = setup.get("flags")
        if not isinstance(flags, dict) or set(flags) - set(FRAMEWORKS):
            raise ValueError(f"setup {name}: flags must be an object keyed by framework ({', '.join(FRAMEWORKS)})")
        for framework in {b["framework"] for b in suite["benchmarks"]} - set(flags):
            raise ValueError(f"setup {name}: flags has no entry for {framework!r} (use [] for none)")
        for framework, items in flags.items():
            if not isinstance(items, list) or not all(isinstance(f, str) for f in items):
                raise ValueError(f"setup {name}: flags.{framework} must be a list of strings")
            reserved = RESERVED[framework] & set(items)
            if reserved:
                raise ValueError(f"setup {name}: flags.{framework} may not contain {sorted(reserved)[0]}: it comes from max_length, the benchmark's repeats or task_args")
        for bench in suite["benchmarks"]:
            profile = bench.get("sampling", setup["sampling"])
            if any(f in THINKING_FLAGS for f in flags[bench["framework"]]) and profiles[profile].get("temperature") == 0:
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


def flags_for(suite, setup_name, task):
    """Launcher flags for one benchmark of the suite under one setup, for the benchmark's framework."""
    bench, chosen = benchmark(suite, task), setup(suite, setup_name)
    flags = list(chosen["flags"][bench["framework"]])
    repeat = bench["repeats"]["via"] == "flag"
    if bench["framework"] == "lm_eval":
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
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            for path in sorted(SUITES_DIR.rglob("*.json")):
                print(path.relative_to(SUITES_DIR).with_suffix(""))
        elif args.command == "show":
            suite = load(args.suite)
            print(f"{suite['name']}: {suite['title']}\n{suite['description']}\n")
            for name, s in suite["setups"].items():
                print(f"{name}: window {s['max_length']}, sampling {s['sampling']}\n  {s.get('description', '')}")
                for framework, items in s["flags"].items():
                    print(f"  {framework}: {shlex.join(items) or '(none)'}")
        elif args.command == "plan":
            suite = load(args.suite)
            print(_table([[r["label"], r["task"], shlex.join(r["flags"]), r["sampling"], r["metric"],
                           r["key"] or "(unchecked)"] for r in plan(suite, args.setup)],
                         ["label", "task", "flags", "sampling", "reports", "results key"]))
        else:
            print(shlex.join(flags_for(load(args.suite), args.setup, args.task)))
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
