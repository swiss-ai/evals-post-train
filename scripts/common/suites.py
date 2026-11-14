"""Suites: the benchmarks of a comparison and the setups (window + launcher flags) a model can run under.

configs/suites/<name>.json holds two things:

  benchmarks  rows in table order: label, task, framework, repeats {n, via}, metric {label, key}
  setups      named kinds of model setup: max_length and the launcher flags (thinking, chat format, ...)

`repeats.via` says who repeats: "flag" (the launcher is given --repeats n) or "task" (the task's own
yaml already repeats n times, nothing is passed). The code only reads the file and knows no model.

    python -m scripts.common.suites list
    python -m scripts.common.suites show table25
    python -m scripts.common.suites plan table25 --setup apertus-1.5-70b-thinking
    python -m scripts.common.suites flags table25 --setup apertus-1.5-70b-thinking --task ifbench
"""

import argparse
import json
import shlex
import sys
from pathlib import Path

SUITES_DIR = Path(__file__).resolve().parents[2] / "configs" / "suites"
FRAMEWORKS = ("lm_eval", "inspect", "evalscope", "harbor", "tau2")
VIA = ("flag", "task")


def _positive_int(where, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{where}: must be an integer >= 1 (got {value!r})")


def validate(suite):
    """Raise ValueError naming the broken field; return the suite otherwise."""
    for key in ("name", "title", "benchmarks", "setups"):
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
        where = f"{where} ({bench['task']})"
        if bench["task"] in tasks or bench["label"] in labels:
            raise ValueError(f"{where}: task and label must be unique within the suite")
        tasks.add(bench["task"]), labels.add(bench["label"])
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
        if "key" not in bench["metric"]:
            raise ValueError(f"{where}: metric.key is required (null if not checked yet)")
    if not suite["setups"]:
        raise ValueError("setups: must not be empty")
    for name, setup in suite["setups"].items():
        _positive_int(f"setup {name} max_length", setup.get("max_length"))
        flags = setup.get("flags", [])
        if not isinstance(flags, list) or not all(isinstance(f, str) for f in flags):
            raise ValueError(f"setup {name}: flags must be a list of strings")
        if "--max-length" in flags or "--repeats" in flags:
            raise ValueError(f"setup {name}: --max-length and --repeats come from max_length and the benchmark, not flags")
    return suite


def load(name_or_path):
    path = Path(name_or_path)
    if not path.suffix:
        path = SUITES_DIR / f"{name_or_path}.json"
    with open(path) as f:
        return validate(json.load(f))


def benchmark(suite, task):
    for bench in suite["benchmarks"]:
        if task in (bench["task"], bench["label"]):
            return bench
    raise ValueError(f"{task!r} is not in suite {suite['name']!r} ({', '.join(b['task'] for b in suite['benchmarks'])})")


def setup(suite, name):
    if name not in suite["setups"]:
        raise ValueError(f"unknown setup {name!r} ({', '.join(suite['setups'])})")
    return suite["setups"][name]


def flags_for(suite, setup_name, task):
    """Launcher flags for one benchmark of the suite under one setup."""
    bench, chosen = benchmark(suite, task), setup(suite, setup_name)
    flags = list(chosen.get("flags", [])) + ["--max-length", str(chosen["max_length"])]
    if bench["repeats"]["via"] == "flag":
        flags += ["--repeats", str(bench["repeats"]["n"])]
    return flags


def plan(suite, setup_name):
    """One row per benchmark: what a run under this setup is and what it reports."""
    return [{"label": b["label"], "task": b["task"], "framework": b["framework"],
             "flags": flags_for(suite, setup_name, b["task"]), "metric": b["metric"]["label"],
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
            for path in sorted(SUITES_DIR.glob("*.json")):
                print(path.stem)
        elif args.command == "show":
            suite = load(args.suite)
            print(f"{suite['name']}: {suite['title']}\n{suite['description']}\n")
            print(_table([[b["label"], b["task"], b["framework"], f"{b['repeats']['n']} via {b['repeats']['via']}",
                           b["metric"]["label"]] for b in suite["benchmarks"]],
                         ["label", "task", "framework", "repeats", "reports"]))
            print("\nsetups:")
            for name, s in suite["setups"].items():
                print(f"  {name}: window {s['max_length']}, {shlex.join(s.get('flags', []))}\n    {s.get('description', '')}")
        elif args.command == "plan":
            suite = load(args.suite)
            print(_table([[r["label"], r["task"], shlex.join(r["flags"]), r["metric"], r["key"] or "(unchecked)"]
                          for r in plan(suite, args.setup)], ["label", "task", "flags", "reports", "results key"]))
        else:
            print(shlex.join(flags_for(load(args.suite), args.setup, args.task)))
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
