"""Merge results from multiple evaluation chunks into one result file."""

import json
import shutil
from pathlib import Path
from argparse import ArgumentParser


def _matches_task(candidate: str, task: str) -> bool:
    return candidate == task or candidate.startswith((f"{task}_", f"{task}/"))


def merge_split_results(
    split_dirs: list[Path], output_dir: Path, excluded_tasks: set[str] | None = None
):
    """Merge results and samples from multiple chunk directories."""
    excluded_tasks = excluded_tasks or set()
    merged_results = None

    for split_dir in split_dirs:
        result_files = list(split_dir.glob("**/results_*.json"))
        if not result_files:
            print(f"WARNING: No results file found in {split_dir}")
            continue

        result_file = result_files[0]
        with open(result_file) as f:
            split_results = json.load(f)

        if merged_results is None:
            # Use the first split as the base
            merged_results = split_results
        else:
            # Merge results from this split into the base
            if "results" in split_results:
                merged_results["results"].update(split_results["results"])
            if "configs" in split_results:
                merged_results["configs"].update(split_results["configs"])
            if "n-shot" in split_results:
                merged_results["n-shot"].update(split_results["n-shot"])
            if "versions" in split_results:
                merged_results["versions"].update(split_results["versions"])
            if "higher_is_better" in split_results:
                merged_results["higher_is_better"].update(
                    split_results["higher_is_better"]
                )
            if "n-samples" in split_results:
                merged_results["n-samples"].update(split_results["n-samples"])

        # Copy sample files
        for sample_file in split_dir.glob("**/samples_*.jsonl"):
            if any(
                sample_file.name.startswith(f"samples_{task}_")
                for task in excluded_tasks
            ):
                continue
            dest = output_dir / sample_file.name
            # Directories are passed oldest-to-newest; a retry or forced rerun
            # must replace the older samples together with its metrics.
            shutil.copy2(sample_file, dest)

    if merged_results is None:
        raise RuntimeError("No results files found in any split directory")

    for section in (
        "results",
        "configs",
        "n-shot",
        "versions",
        "higher_is_better",
        "n-samples",
    ):
        values = merged_results.get(section)
        if not isinstance(values, dict):
            continue
        for key in list(values):
            if any(_matches_task(key, task) for task in excluded_tasks):
                values.pop(key)

    # Write merged results with a consistent timestamp
    # Use the timestamp from the base results file name
    base_result_files = list(split_dirs[0].glob("**/results_*.json"))
    timestamp = (
        base_result_files[0].stem.replace("results_", "")
        if base_result_files
        else "merged"
    )

    output_file = output_dir / f"results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(merged_results, f, indent=2)

    task_count = len(merged_results.get("results", {}))
    print(f"Merged {len(split_dirs)} chunks -> {task_count} tasks in {output_file}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Merge chunked evaluation results")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--split_dirs",
        nargs="+",
        type=Path,
        help="Directories containing split results",
    )
    source.add_argument(
        "--split_dirs_file",
        type=Path,
        help="Text file containing one result directory per line",
    )
    parser.add_argument(
        "--exclude_tasks_file",
        type=Path,
        help="Optional text file of failed tasks to remove from partial aggregates",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for merged results",
    )
    args = parser.parse_args()

    split_dirs = args.split_dirs
    if args.split_dirs_file:
        split_dirs = [
            Path(line.strip())
            for line in args.split_dirs_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    excluded_tasks = set()
    if args.exclude_tasks_file:
        excluded_tasks = {
            line.strip()
            for line in args.exclude_tasks_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    merge_split_results(split_dirs, args.output_dir, excluded_tasks)
