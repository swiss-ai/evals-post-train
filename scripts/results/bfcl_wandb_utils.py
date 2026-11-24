"""Build a ModelEvaluation from a BFCL run (scripts/custom/run_bfcl.sh) for W&B upload.

A run's work dir holds summary.json (scripts/custom/bfcl.py: the accuracy of every category, the official groups
and, for v3, the overall). Like evalscope_wandb_utils, this only adapts it into the ModelEvaluation/Task/Metric
structures; the upload itself is wandb_alignment_utils.upload_multi_model_results. No samples are uploaded: the
model's answers and bfcl-eval's per-category score files stay in the work dir.
"""

import json
from pathlib import Path
from typing import Tuple

from .data_structures import Metric, ModelEvaluation, Task


def task_name(summary: dict) -> str:
    return f"bfcl_{summary['version']}"


def create_model_evaluation_from_bfcl_run(model_name: str, work_dir: Path) -> Tuple[ModelEvaluation, int]:
    """(ModelEvaluation, duration in seconds) of a BFCL run's work dir.

    One task, bfcl_v3 or bfcl_v4, with: the official groups (non_live, live, multi_turn, agentic) and the
    overall where bfcl-eval has them, every category's accuracy as `<category>` and the number of entries
    (n_samples). Accuracies are fractions (0..1)."""
    with open(work_dir / "summary.json") as f:
        summary = json.load(f)
    metrics = []
    if summary.get("overall") is not None:
        metrics.append(Metric(name="overall", score=summary["overall"]))
    metrics += [Metric(name=group, score=score) for group, score in summary["groups"].items() if score is not None]
    # v4 without the web search categories: overall_no_web_search, irrelevance and memory. Its other scores
    # (non-live, live, multi-turn) are bfcl-eval's groups above, which the package rounds to two decimals.
    present = {m.name for m in metrics}
    metrics += [Metric(name=name, score=score) for name, score in (summary.get("v4_without_web_search") or {}).items()
                if name not in present]
    for name, category in summary["categories"].items():
        metrics.append(Metric(name=name, score=category["accuracy"]))
    metrics.append(Metric(name="n_samples", score=sum(c["total"] for c in summary["categories"].values())))
    if errors := summary.get("inference_errors"):
        # Entries bfcl-eval scored wrong because their generation raised (counted in the accuracies above).
        metrics.append(Metric(name="n_inference_errors", score=errors["n_errors"]))
        metrics.append(Metric(name="n_context_overflows", score=errors["by_cause"]["context_overflow"]))
    task = Task(task_name=task_name(summary), metrics=metrics, samples=[])
    return ModelEvaluation(model_name=model_name, tasks=[task]), int(summary.get("duration_s") or 0)
