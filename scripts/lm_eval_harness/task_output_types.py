"""Sort lm-eval tasks into generative and log-likelihood ones.

Log-likelihood tasks are the ones whose output_type is multiple_choice,
loglikelihood or loglikelihood_rolling.

Reads each task's yaml (following ``include:``) through the harness's own task
index, so it must run where lm_eval is importable (the evaluation container).
A group or tag has the kinds of its members; a name the index doesn't know
(a path, a Python-only task) has none.

    python -m scripts.lm_eval_harness.task_output_types --tasks a,b,c --require-generative

(CHAT_FORMAT=apertus2, whose model only generates) fails, naming every task,
group or tag with a log-likelihood member. Only yaml is read: a Python task class can still send
log-likelihood requests its yaml doesn't declare (squadv2, scrolls).
"""

from __future__ import annotations

import argparse
import fnmatch

LOGLIKELIHOOD = frozenset({"multiple_choice", "loglikelihood", "loglikelihood_rolling"})


def _members(value, out: list) -> None:
    """Task names a group's ``task:`` list mentions, however nested."""
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, list):
        for item in value:
            _members(item, out)
    elif isinstance(value, dict):
        _members(value.get("task"), out)


class Classifier:
    def __init__(self, task_manager, load_yaml) -> None:
        self._index = task_manager.task_index
        self._load_yaml = load_yaml
        self._cache: dict[str, set[bool]] = {}
        self._tag_members: dict[str, list[str]] = {}
        for name, entry in self._index.items():
            for tag in getattr(entry, "tags", None) or ():
                self._tag_members.setdefault(tag, []).append(name)

    def _config(self, entry) -> dict:
        if getattr(entry, "yaml_path", None):
            return self._load_yaml(entry.yaml_path, resolve_func=False)
        return dict(getattr(entry, "cfg", None) or {})

    def kinds(self, name: str, seen: frozenset[str] = frozenset()) -> set[bool]:
        """{True} log-likelihood, {False} generative, both for a mix; empty
        when unknown."""
        if name in self._cache:
            return self._cache[name]
        entry = self._index.get(name)
        if entry is None or name in seen:
            return set()
        members = self._children(name, entry)
        found: set[bool] = set()
        if members is not None:
            for member in members:
                found |= self.kinds(member, seen | {name})
        elif cfg := self._config(entry):
            found = {cfg.get("output_type", "generate_until") in LOGLIKELIHOOD}
        self._cache[name] = found
        return found

    def leaves(self, name: str, seen: frozenset[str] = frozenset()) -> set[str]:
        if name in seen:
            raise ValueError(f"Cyclic task specification: {name}")
        entry = self._index.get(name)
        if entry is None and (matches := fnmatch.filter(self._index, name)):
            return set().union(*(self.leaves(match, seen | {name}) for match in matches))
        # A path or an unindexed task stays as named: lm_eval resolves it.
        members = None if entry is None else self._children(name, entry)
        if members is None:
            return {name}
        # lm_eval names a group's inline tasks group::task.
        members = [m if m in self._index else f"{name}::{m}" for m in members]
        return set().union(*(self.leaves(member, seen | {name}) for member in members))

    def _children(self, name: str, entry) -> list[str] | None:
        """Members of a group or tag; None for a task."""
        kind = getattr(entry.kind, "name", str(entry.kind)).upper()
        if kind == "TAG":
            return self._tag_members.get(name, [])
        if kind != "GROUP":
            return None
        members: list = []
        _members(self._config(entry).get("task"), members)
        return members

    def with_loglikelihood(self, tasks: list[str]) -> list[str]:
        """Tasks, groups and tags with at least one log-likelihood member."""
        return [task for task in tasks if True in self.kinds(task)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tasks", required=True, help="comma-separated lm-eval task names")
    parser.add_argument(
        "--require-generative",
        action="store_true",
        required=True,
        help="fail if any task is scored by log-likelihood",
    )
    args = parser.parse_args()

    from lm_eval.tasks import TaskManager
    from lm_eval.tasks._yaml_loader import load_yaml

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    classifier = Classifier(TaskManager(), load_yaml)
    refused = classifier.with_loglikelihood(tasks)
    if refused:
        raise SystemExit(f"Scored by log-likelihood, which this model cannot do: {','.join(refused)}")
    print(f"All {len(tasks)} tasks are generative.")


if __name__ == "__main__":
    main()
