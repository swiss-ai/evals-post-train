#!/usr/bin/env python3
"""Resolve the effective ``max_gen_toks`` for one or more lm-eval tasks.

Used by ``evaluate.sbatch`` to apply a *floor* (never a clobber) to the
generation budget: the launcher passes ``--floor <MAX_NEW_TOKENS>`` and this
script returns ``max(floor, <largest max_gen_toks declared by the task YAMLs>)``.
That way tasks that deliberately ask for more (e.g. AIME's 32768) keep their
value, while tasks with no/smaller setting still get the sane floor.

Fail-safe by design: any error or undetected value falls back to the floor, so
the worst case is identical to passing ``max_gen_toks=<floor>`` unconditionally.
Prints a single integer to stdout.
"""

import argparse
import logging

# lm-eval logs to stderr, but silence everything so stdout stays a clean int.
logging.disable(logging.CRITICAL)


def _mgt_from_config(config):
    """Return int max_gen_toks declared in a (sub)config, or None."""
    if not isinstance(config, dict):
        return None
    gen = config.get("generation_kwargs")
    if isinstance(gen, dict) and gen.get("max_gen_toks") is not None:
        try:
            return int(gen["max_gen_toks"])
        except (TypeError, ValueError):
            return None
    return None


def _collect(tm, name, seen):
    """Recursively collect declared max_gen_toks values for a task/group/tag."""
    values = []
    if name in seen or name not in tm.task_index:
        return values
    seen.add(name)

    try:
        config = tm._get_config(name)
    except Exception:
        config = {}

    v = _mgt_from_config(config)
    if v is not None:
        values.append(v)

    # Expand groups/tags one level (subtasks may be names or inline dicts).
    try:
        is_task = tm._name_is_task(name)
    except Exception:
        is_task = True
    if not is_task:
        try:
            subtasks = tm._get_tasklist(name)
        except Exception:
            subtasks = None
        if isinstance(subtasks, list):
            for sub in subtasks:
                if isinstance(sub, str):
                    values.extend(_collect(tm, sub, seen))
                elif isinstance(sub, dict):
                    v = _mgt_from_config(sub)
                    if v is not None:
                        values.append(v)
                    ref = sub.get("task")
                    if isinstance(ref, str):
                        values.extend(_collect(tm, ref, seen))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, help="Comma-separated task names")
    parser.add_argument("--floor", type=int, default=0, help="Minimum max_gen_toks")
    args = parser.parse_args()

    result = args.floor
    try:
        from lm_eval.tasks import TaskManager

        tm = TaskManager()
        seen = set()
        best = 0
        for name in args.tasks.split(","):
            name = name.strip()
            if not name:
                continue
            for v in _collect(tm, name, seen):
                best = max(best, v)
        result = max(args.floor, best)
    except Exception:
        result = args.floor

    print(int(result))


if __name__ == "__main__":
    main()
