"""Entry point for standalone benchmark execution."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
from typing import Sequence

from .registry import resolve_benchmarks
from .schemas import BenchmarkResult, RunContext, write_standalone_artifacts


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run standalone benchmarks and emit normalized artifacts.")
    parser.add_argument("--tasks", required=True, help="Comma-separated standalone task names.")
    parser.add_argument("--model", required=True, help="Model path or model identifier.")
    parser.add_argument("--name", required=True, help="Stable run/model display name.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for normalized artifacts.")
    parser.add_argument("--limit", type=int, help="Optional per-benchmark instance limit.")
    parser.add_argument("--sandbox-backend", default=os.environ.get("SANDBOX_BACKEND", "none"))
    parser.add_argument("--model-api-base", default=os.environ.get("MODEL_API_BASE"))
    parser.add_argument("--model-api-key", default=os.environ.get("MODEL_API_KEY"))
    parser.add_argument("--model-api-model", default=os.environ.get("MODEL_API_MODEL"))
    args = parser.parse_args(argv)

    specs = resolve_benchmarks(args.tasks)
    context = RunContext(
        model=args.model,
        name=args.name,
        output_dir=args.output_dir,
        benchmarks=specs,
        limit=args.limit,
        model_api_base=args.model_api_base,
        model_api_key=args.model_api_key,
        model_api_model=args.model_api_model,
        sandbox_backend=args.sandbox_backend,
        metadata={"pid": os.getpid()},
    )

    results: list[BenchmarkResult] = []
    for spec in specs:
        module = importlib.import_module(f"scripts.standalone.benchmarks.{spec.runner}")
        results.append(module.run(spec, context))

    result_path = write_standalone_artifacts(context, results)
    print(f"Wrote standalone results: {result_path}")


if __name__ == "__main__":
    main()
