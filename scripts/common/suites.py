"""Suite definitions: which benchmarks, with which repeats, and a recipe of flags per kind of model.

A suite lives in configs/suites/<name>.json. It lists benchmarks (framework, repeats, whether
--repeats must be passed, the results key that holds the table value) and recipes (the max window
and launcher flags a kind of model needs). The code only reads the file; nothing here knows a
model. A recipe names a kind of setup, the caller picks one.

    python -m scripts.common.suites list
    python -m scripts.common.suites show table25
    python -m scripts.common.suites flags table25 --recipe apertus-1.5-70b-thinking --task ifbench
"""

import argparse
import json
import shlex
import sys
from pathlib import Path

SUITES_DIR = Path(__file__).resolve().parents[2] / "configs" / "suites"
FRAMEWORKS = ("lm_eval", "inspect", "evalscope", "harbor", "tau2")


def validate(suite):
    """Raise ValueError for a malformed suite; return it otherwise."""
    for key in ("name", "benchmarks", "recipes"):
        if key not in suite:
            raise ValueError(f"suite is missing {key!r}")
    if not suite["benchmarks"]:
        raise ValueError("suite has no benchmarks")
    for task, spec in suite["benchmarks"].items():
        if spec.get("framework") not in FRAMEWORKS:
            raise ValueError(f"{task}: framework must be one of {FRAMEWORKS} (got {spec.get('framework')!r})")
        repeats = spec.get("repeats")
        if repeats is not None and (isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1):
            raise ValueError(f"{task}: repeats must be an integer >= 1 or null (got {repeats!r})")
        if spec.get("repeats_flag") and not repeats:
            raise ValueError(f"{task}: repeats_flag needs repeats")
    for name, recipe in suite["recipes"].items():
        length = recipe.get("max_length")
        if isinstance(length, bool) or not isinstance(length, int) or length < 1:
            raise ValueError(f"recipe {name}: max_length must be a positive integer (got {length!r})")
        flags = recipe.get("flags", [])
        if not isinstance(flags, list) or not all(isinstance(f, str) for f in flags):
            raise ValueError(f"recipe {name}: flags must be a list of strings")
    return suite


def load(name_or_path):
    path = Path(name_or_path)
    if not path.suffix:
        path = SUITES_DIR / f"{name_or_path}.json"
    with open(path) as f:
        return validate(json.load(f))


def flags_for(suite, recipe, task):
    """Launcher flags for one benchmark of the suite under one recipe."""
    if task not in suite["benchmarks"]:
        raise ValueError(f"{task!r} is not in suite {suite['name']!r} ({', '.join(suite['benchmarks'])})")
    if recipe not in suite["recipes"]:
        raise ValueError(f"unknown recipe {recipe!r} ({', '.join(suite['recipes'])})")
    spec, rec = suite["benchmarks"][task], suite["recipes"][recipe]
    flags = list(rec.get("flags", [])) + ["--max-length", str(rec["max_length"])]
    if spec.get("repeats_flag"):
        flags += ["--repeats", str(spec["repeats"])]
    return flags


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    show = sub.add_parser("show")
    show.add_argument("suite")
    flags = sub.add_parser("flags")
    flags.add_argument("suite")
    flags.add_argument("--recipe", required=True)
    flags.add_argument("--task", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            for path in sorted(SUITES_DIR.glob("*.json")):
                print(path.stem)
        elif args.command == "show":
            print(json.dumps(load(args.suite), indent=2))
        else:
            print(shlex.join(flags_for(load(args.suite), args.recipe, args.task)))
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
