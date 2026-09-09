"""Utilities for resumable, chunked evaluation launches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RUN_CONFIG_FILENAME = ".eval-run-config.json"
RUN_CONFIG_SCHEMA_VERSION = 1


def read_tasks(value: str) -> list[str]:
    """Read a task file or comma-separated task expression in stable order."""
    path = Path(value)
    try:
        is_file = path.is_file()
    except OSError:
        is_file = False
    if is_file:
        lines = path.read_text(encoding="utf-8").splitlines()
    else:
        lines = value.split(",")

    tasks: list[str] = []
    for line in lines:
        task = line.split("#", 1)[0].strip()
        if task:
            tasks.append(task)
    return tasks


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(
            _contains_key(child, key) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(child, key) for child in value)
    return False


def make_run_config(fields: list[str]) -> dict[str, object]:
    """Create a canonical, signed description of result-affecting settings."""
    configuration: dict[str, str] = {}
    for field in fields:
        key, separator, value = field.partition("=")
        if not separator or not key:
            raise ValueError(f"invalid run-config field {field!r}; expected key=value")
        if key in configuration:
            raise ValueError(f"duplicate run-config field: {key}")
        configuration[key] = value

    unsigned: dict[str, object] = {
        "schema_version": RUN_CONFIG_SCHEMA_VERSION,
        "configuration": dict(sorted(configuration.items())),
    }
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"))
    return unsigned | {"signature": hashlib.sha256(canonical.encode()).hexdigest()}


def run_config_matches(eval_dir: Path, expected: dict[str, object]) -> bool:
    """Return whether an evaluation directory belongs to the requested run config."""
    try:
        actual = json.loads(
            (eval_dir / RUN_CONFIG_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    return actual == expected


def scan_results(
    tasks: list[str],
    harness_dirs: Path | list[Path],
    force_patterns: list[str],
    force_after: str | None = None,
    run_config: dict[str, object] | None = None,
) -> tuple[dict[str, Path], list[str], list[Path]]:
    """Find completed tasks, missing tasks, and unique result directories."""
    completed: dict[str, Path] = {}
    if isinstance(harness_dirs, Path):
        harness_dirs = [harness_dirs]
    for harness_dir in harness_dirs:
        if not harness_dir.is_dir():
            continue
        eval_dirs = sorted(
            path
            for path in harness_dir.glob("eval_*")
            if path.is_dir() and not path.name.startswith("eval_merged")
        )
        for eval_dir in eval_dirs:
            # Legacy and differently configured outputs are intentionally ignored
            # when the orchestrator supplies a config. This prevents a limited
            # smoke run, another checkpoint, or another backend from satisfying a
            # full evaluation merely because the task names match.
            if run_config is not None and not run_config_matches(eval_dir, run_config):
                continue
            for result_file in sorted(eval_dir.rglob("results_*.json")):
                try:
                    payload = json.loads(result_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                for task in tasks:
                    forced = any(pattern in task for pattern in force_patterns)
                    if forced and force_after and eval_dir.name < force_after:
                        continue
                    if _contains_key(payload, task):
                        completed[task] = eval_dir

    if force_patterns and not force_after:
        for task in list(completed):
            if any(pattern in task for pattern in force_patterns):
                completed.pop(task)

    missing = [task for task in tasks if task not in completed]
    # Merge oldest-to-newest so a forced rerun or later recovery overrides an
    # earlier result for the same task.
    result_dirs = sorted(
        set(completed.values()), key=lambda path: (path.name, str(path))
    )
    return completed, missing, result_dirs


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{value}\n" for value in values)
    path.write_text(text, encoding="utf-8")


def command_normalize(args: argparse.Namespace) -> None:
    write_lines(args.output, read_tasks(args.tasks))


def command_chunk(args: argparse.Namespace) -> None:
    tasks = read_tasks(str(args.tasks_file))
    chunks = [
        ",".join(tasks[index : index + args.chunk_size])
        for index in range(0, len(tasks), args.chunk_size)
    ]
    write_lines(args.output, chunks)


def command_config(args: argparse.Namespace) -> None:
    config = make_run_config(args.field)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(config["signature"])


def command_scan(args: argparse.Namespace) -> None:
    tasks = read_tasks(str(args.tasks_file))
    run_config = None
    if args.run_config:
        try:
            run_config = json.loads(args.run_config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(
                f"invalid run config {args.run_config}: {error}"
            ) from error
    completed, missing, result_dirs = scan_results(
        tasks=tasks,
        harness_dirs=args.harness_dir,
        force_patterns=args.force_pattern,
        force_after=args.force_after,
        run_config=run_config,
    )
    write_lines(args.missing_output, missing)
    write_lines(args.eval_dirs_output, [str(path) for path in result_dirs])
    if args.completed_output:
        write_lines(
            args.completed_output,
            [f"{task}\t{completed[task]}" for task in tasks if task in completed],
        )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize = subparsers.add_parser("normalize")
    normalize.add_argument("--tasks", required=True)
    normalize.add_argument("--output", type=Path, required=True)
    normalize.set_defaults(func=command_normalize)

    chunk = subparsers.add_parser("chunk")
    chunk.add_argument("--tasks-file", type=Path, required=True)
    chunk.add_argument("--chunk-size", type=int, required=True)
    chunk.add_argument("--output", type=Path, required=True)
    chunk.set_defaults(func=command_chunk)

    config = subparsers.add_parser("config")
    config.add_argument("--field", action="append", default=[])
    config.add_argument("--output", type=Path, required=True)
    config.set_defaults(func=command_config)

    scan = subparsers.add_parser("scan")
    scan.add_argument("--tasks-file", type=Path, required=True)
    scan.add_argument("--harness-dir", type=Path, action="append", required=True)
    scan.add_argument("--missing-output", type=Path, required=True)
    scan.add_argument("--eval-dirs-output", type=Path, required=True)
    scan.add_argument("--completed-output", type=Path)
    scan.add_argument("--force-pattern", action="append", default=[])
    scan.add_argument("--force-after")
    scan.add_argument("--run-config", type=Path)
    scan.set_defaults(func=command_scan)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if getattr(args, "chunk_size", 1) < 1:
        raise SystemExit("--chunk-size must be >= 1")
    args.func(args)


if __name__ == "__main__":
    main()
