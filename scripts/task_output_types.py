"""Sort lm-eval tasks into generative and log-likelihood ones.

For APPLY_CHAT_TEMPLATE=by-type (launch_evaluations.sh --chat-template-by-type):
evaluate.sbatch runs the generative tasks with the chat template and the
log-likelihood ones (multiple_choice, loglikelihood, loglikelihood_rolling)
without it, since those score each choice as a raw continuation of the prompt.

Reads each task's yaml (following ``include:``) through the harness's own task
index, so it must run where lm_eval is importable (the evaluation container).
A group or tag counts as log-likelihood when all its members are; one mixing
both kinds, and any name the index doesn't know (a path, a Python-only task),
goes with the generative tasks: the chat template, the launcher's default.

    python -m scripts.task_output_types --tasks a,b,c --output kinds.sh

writes CHAT_TASKS=... and RAW_TASKS=... (comma-separated, input order) for
the caller to source.
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

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
        kind = getattr(entry.kind, "name", str(entry.kind)).upper()
        found: set[bool] = set()
        if kind == "TAG":
            for member in self._tag_members.get(name, []):
                found |= self.kinds(member, seen | {name})
        else:
            cfg = self._config(entry)
            if kind == "GROUP":
                members: list = []
                _members(cfg.get("task"), members)
                for member in members:
                    found |= self.kinds(member, seen | {name})
            elif cfg:
                found = {cfg.get("output_type", "generate_until") in LOGLIKELIHOOD}
        self._cache[name] = found
        return found

    def split(self, tasks: list[str]) -> tuple[list[str], list[str]]:
        chat: list[str] = []
        raw: list[str] = []
        for task in tasks:
            (raw if self.kinds(task) == {True} else chat).append(task)
        return chat, raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tasks", required=True, help="comma-separated lm-eval task names")
    parser.add_argument("--output", type=Path, required=True, help="shell file to write")
    args = parser.parse_args()

    from lm_eval.tasks import TaskManager
    from lm_eval.tasks._yaml_loader import load_yaml

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    chat, raw = Classifier(TaskManager(), load_yaml).split(tasks)
    args.output.write_text(
        f"CHAT_TASKS={shlex.quote(','.join(chat))}\nRAW_TASKS={shlex.quote(','.join(raw))}\n"
    )
    print(f"With the chat template: {','.join(chat) or '(none)'}")
    print(f"Without (log-likelihood): {','.join(raw) or '(none)'}")


if __name__ == "__main__":
    main()
