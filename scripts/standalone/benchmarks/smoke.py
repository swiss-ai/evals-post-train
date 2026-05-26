"""Smoke benchmark for validating standalone logging plumbing."""

from __future__ import annotations

from ..schemas import BenchmarkResult, BenchmarkSpec, RunContext, StandaloneSample


def run(spec: BenchmarkSpec, context: RunContext) -> BenchmarkResult:
    sample_count = context.limit or 2
    samples = []

    for index in range(sample_count):
        success = 1.0 if index % 2 == 0 else 0.0
        samples.append(
            StandaloneSample(
                benchmark=spec.name,
                instance_id=f"smoke-{index}",
                status="scored",
                metrics={"success": success},
                input={"prompt": "Return OK."},
                prediction={"model": context.model, "text": "OK" if success else "NOT_OK"},
                trajectory_summary={"steps": 1, "sandbox_backend": context.sandbox_backend},
            )
        )

    mean_success = sum(sample.metrics["success"] for sample in samples) / len(samples)
    return BenchmarkResult(
        benchmark=spec.name,
        metrics={
            "success": mean_success,
            "num_instances": float(len(samples)),
        },
        samples=tuple(samples),
        version=spec.version,
        config={
            "runner": spec.runner,
            "backend": spec.backend,
            "requires": spec.requires,
        },
        higher_is_better={
            "success": True,
            "num_instances": True,
        },
        original_samples=len(samples),
    )

