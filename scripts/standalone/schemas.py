"""Shared schemas and artifact writers for standalone evaluations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "standalone-eval-v1"


def utc_timestamp() -> str:
    """Return an lm-eval-compatible timestamp for artifact filenames."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z").replace(":", "-")


@dataclass(frozen=True)
class MetricSpec:
    name: str
    higher_is_better: bool = True


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    runner: str
    backend: str = "standalone"
    description: str = ""
    version: str = "unknown"
    main_metric: str = ""
    metrics: tuple[MetricSpec, ...] = ()
    requires: dict[str, Any] = field(default_factory=dict)

    @property
    def metric_names(self) -> list[str]:
        return [metric.name for metric in self.metrics]


@dataclass(frozen=True)
class RunContext:
    model: str
    name: str
    output_dir: Path
    benchmarks: tuple[BenchmarkSpec, ...]
    limit: int | None = None
    model_api_base: str | None = None
    model_api_key: str | None = None
    model_api_model: str | None = None
    sandbox_backend: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StandaloneSample:
    benchmark: str
    instance_id: str
    status: str
    metrics: dict[str, float]
    input: dict[str, Any] = field(default_factory=dict)
    prediction: dict[str, Any] = field(default_factory=dict)
    trajectory_summary: dict[str, Any] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    timing: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "benchmark": self.benchmark,
            "instance_id": self.instance_id,
            "status": self.status,
            "metrics": list(self.metrics.keys()),
            "input": self.input,
            "prediction": self.prediction,
            "trajectory_summary": self.trajectory_summary,
            "artifact_paths": self.artifact_paths,
            "error": self.error,
            "timing": self.timing,
        }
        payload.update(self.metrics)
        payload.update(self.extra)
        if "is_correct" not in payload:
            payload["is_correct"] = _infer_is_correct(self.metrics)
        return payload


@dataclass(frozen=True)
class BenchmarkResult:
    benchmark: str
    metrics: dict[str, float]
    samples: tuple[StandaloneSample, ...]
    version: str = "unknown"
    config: dict[str, Any] = field(default_factory=dict)
    higher_is_better: dict[str, bool] = field(default_factory=dict)
    original_samples: int | None = None


def write_standalone_artifacts(
    context: RunContext,
    results: list[BenchmarkResult],
) -> Path:
    """Write manifest, results JSON, and per-benchmark sample JSONL files."""
    context.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = context.output_dir / "artifacts"
    artifact_dir.mkdir(exist_ok=True)

    timestamp = utc_timestamp()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": context.output_dir.name,
        "model_name": context.name,
        "model_path": context.model,
        "backend": "standalone",
        "benchmarks": [spec.name for spec in context.benchmarks],
        "limit": context.limit,
        "model_api_base": context.model_api_base,
        "model_api_model": context.model_api_model,
        "sandbox_backend": context.sandbox_backend,
        "metadata": context.metadata,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(context.output_dir / "run_manifest.json", manifest)

    payload: dict[str, Any] = {
        "results": {},
        "configs": {},
        "versions": {},
        "higher_is_better": {},
        "n-samples": {},
        "standalone_schema_version": SCHEMA_VERSION,
    }

    for result in results:
        payload["results"][result.benchmark] = result.metrics
        payload["configs"][result.benchmark] = result.config
        payload["versions"][result.benchmark] = result.version
        payload["higher_is_better"][result.benchmark] = result.higher_is_better
        payload["n-samples"][result.benchmark] = {
            "effective": len(result.samples),
            "original": result.original_samples or len(result.samples),
        }

        sample_path = context.output_dir / f"samples_{result.benchmark}_{timestamp}.jsonl"
        with sample_path.open("w") as handle:
            for sample in result.samples:
                handle.write(json.dumps(sample.to_json(), sort_keys=True) + "\n")

    result_path = context.output_dir / f"results_{timestamp}.json"
    _write_json(result_path, payload)
    return result_path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _infer_is_correct(metrics: dict[str, float]) -> bool | None:
    for name in ("resolved", "success", "pass@1", "pass_rate", "accuracy", "exact_match"):
        if name in metrics:
            return metrics[name] == 1.0
    return None

