"""Benchmark registry loader for standalone evaluations."""

from __future__ import annotations

import tomllib
from pathlib import Path

from .schemas import BenchmarkSpec, MetricSpec


DEFAULT_REGISTRY_DIR = Path("configs/standalone/benchmarks")


def load_benchmark_specs(registry_dir: Path = DEFAULT_REGISTRY_DIR) -> dict[str, BenchmarkSpec]:
    specs: dict[str, BenchmarkSpec] = {}
    for path in sorted(registry_dir.glob("*.toml")):
        with path.open("rb") as handle:
            raw = tomllib.load(handle)

        metrics = tuple(
            MetricSpec(
                name=item["name"],
                higher_is_better=bool(item.get("higher_is_better", True)),
            )
            for item in raw.get("metrics", [])
        )
        name = raw["name"]
        specs[name] = BenchmarkSpec(
            name=name,
            runner=raw["runner"],
            backend=raw.get("backend", "standalone"),
            description=raw.get("description", ""),
            version=raw.get("version", "unknown"),
            main_metric=raw.get("main_metric", metrics[0].name if metrics else ""),
            metrics=metrics,
            requires=raw.get("requires", {}),
        )
    return specs


def resolve_benchmarks(
    task_names: str,
    registry_dir: Path = DEFAULT_REGISTRY_DIR,
) -> tuple[BenchmarkSpec, ...]:
    specs = load_benchmark_specs(registry_dir)
    names = _split_names(task_names)

    if not names:
        raise ValueError("No standalone tasks requested.")

    missing = [name for name in names if name not in specs]
    if missing:
        available = ", ".join(sorted(specs))
        raise ValueError(f"Unknown standalone benchmark(s): {', '.join(missing)}. Available: {available}")

    return tuple(specs[name] for name in names)


def _split_names(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]
