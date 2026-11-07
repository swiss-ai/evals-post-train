"""Snapshot a repeat policy and select its leaf-task overrides for each job."""

import argparse
import json
from pathlib import Path


def read_policy(path):
    policy = json.loads(Path(path).read_text())
    if not isinstance(policy, dict) or set(policy) - {"repeats", "repeat_scoring", "task_overrides"}:
        raise ValueError("Repeat config accepts repeats, repeat_scoring and task_overrides only")
    if not isinstance(policy.get("task_overrides", {}), dict):
        raise ValueError("task_overrides must be an object")
    return policy


def select_policy(policy, suite, chunk, classifier):
    selected = set().union(*(classifier.leaves(name) for name in suite))
    overrides = policy.get("task_overrides", {})
    unknown = overrides.keys() - selected
    if unknown:
        raise ValueError(f"Repeat overrides do not name selected leaf tasks: {sorted(unknown)}")
    leaves = set().union(*(classifier.leaves(name) for name in chunk))
    if not leaves <= selected:
        raise ValueError("Chunk contains tasks outside the evaluation suite")
    return policy | {"task_overrides": {k: v for k, v in overrides.items() if k in leaves}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--suite", help="suite task list (default: --tasks)")
    parser.add_argument("--tasks")
    args = parser.parse_args()
    policy = read_policy(args.policy)
    if args.tasks:
        try:
            import lm_eval.repeats  # noqa: F401
        except ImportError as exc:
            raise SystemExit("Repeat scoring requires the feature harness revision; the installed harness does not support it") from exc
        from lm_eval.tasks import TaskManager
        from lm_eval.tasks._yaml_loader import load_yaml
        from scripts.lm_eval_harness.task_output_types import Classifier

        chunk = args.tasks.split(",")
        suite = Path(args.suite).read_text().splitlines() if args.suite else chunk
        classifier = Classifier(TaskManager(), load_yaml)
        policy = select_policy(policy, suite, chunk, classifier)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(policy, sort_keys=True)
    args.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
