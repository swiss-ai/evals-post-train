"""`python -m scripts.common.suites`: list, show, plan, flags and run."""

import argparse
import os
import shlex
import subprocess
import sys

from scripts.common.suites.flags import flags_for, sampling_for
from scripts.common.suites.runners import commands
from scripts.common.suites.schema import REPO_ROOT, SUITES_DIR, load, setup


def plan(suite, setup_name):
    """One row per benchmark: what a run under this setup is and what it reports."""
    return [{"label": b["label"], "group": b.get("group", ""), "task": b["task"], "framework": b["framework"],
             "flags": flags_for(suite, setup_name, b["task"]),
             "sampling": sampling_for(suite, setup_name, b["task"]), "metric": b["metric"]["label"],
             "key": b["metric"]["key"]} for b in suite["benchmarks"]]


def _table(rows, headers):
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    return "\n".join("  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip() for r in [headers] + rows)


def main(argv=None):
    from scripts.common import suites  # the package docstring is the CLI help

    parser = argparse.ArgumentParser(prog="suites.py", description=suites.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    run_p.add_argument("--helper-url", help="endpoint of the helper model (tau2's simulated user and judge), served by a job of its own")
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
                if s.get("launch"):
                    launch = s["launch"]
                    print(f"  launch: {launch['nodes']} node(s), {launch['time']}, {launch['model_path']}")
        elif args.command == "plan":
            suite = load(args.suite)
            print(_table([[r["label"], r["group"], r["task"], shlex.join(r["flags"]), r["sampling"], r["metric"],
                           r["key"] or "(unchecked)"] for r in plan(suite, args.setup)],
                         ["label", "group", "task", "flags", "sampling", "reports", "results key"]))
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
                            args.qos, args.partition, fetch=args.submit, helper_url=args.helper_url)
            print(f"{len(rows)} launches: {len(models)} model{'s' * (len(models) != 1)} x "
                  f"{len(suite['benchmarks'])} benchmarks", file=sys.stderr)
            if args.submit and suite.get("results_dir"):
                os.makedirs(os.path.join(suite["results_dir"], "slurm"), exist_ok=True)
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
