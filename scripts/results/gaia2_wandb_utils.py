"""Build a ModelEvaluation from an Omnilingual GAIA2 run (scripts/gaia2/run_gaia2_eval.sh) for W&B upload.

A run's work dir holds summary.json (scripts.gaia2.run: per language/subset cell, language,
subset and overall) and gaia2-runner's runs/<language>/<subset>/results/.../<scenario>/result.json
(one per scenario and attempt). Tasks, named omnilingual_gaia2[_<language>][_<subset>]:

- per cell, language and subset, and overall: success_rate (passed / attempts, gaia2-runner's
  own pass rate), pass@k with several attempts, n_scenarios, n_attempts and n_errors;
- samples, for the cells: stratified passed/failed scenarios plus up to n_errors unjudged ones.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .data_structures import Metric, ModelEvaluation, Sample, Task
from .wandb_alignment_utils import _select_stratified_samples

TASK_PREFIX = "omnilingual_gaia2"


def task_name(*parts: str) -> str:
    return "_".join([TASK_PREFIX, *parts])


def _metrics(entry: Dict[str, Any]) -> List[Metric]:
    metrics = []
    if entry.get("success_rate") is not None:
        metrics.append(Metric(name="success_rate", score=entry["success_rate"]))
    metrics += [Metric(name=f"pass@{k}", score=v) for k, v in (entry.get("pass_at_k") or {}).items()]
    metrics += [Metric(name=key, score=entry.get(key, 0)) for key in ("n_scenarios", "n_attempts", "n_errors")]
    return metrics


def _sample(result: Dict[str, Any], attempt: int) -> Dict[str, Any]:
    """A bounded per-scenario summary for the W&B samples table (no traces)."""
    success = result.get("success")
    sample = {
        "scenario": result.get("scenario_id"),
        "attempt": attempt,
        "success": success,
        "failure_reasons": [str(r)[:500] for r in (result.get("failure_reasons") or [])][:5],
        "error": str(result["error"])[:500] if result.get("error") else None,
        "agent_response": (result.get("agent_response") or "")[:2000],
        "num_agent_events": result.get("num_agent_events"),
        "timed_out": bool(result.get("timed_out")),
        "metrics": [],
    }
    if isinstance(success, bool):
        sample["pass@1"] = float(success)
        sample["metrics"].append("pass@1")
    return sample


def _cell_samples(results: Path, n_positive: int, n_negative: int, n_errors: int) -> List[Sample]:
    samples = []
    for path in sorted(results.glob("**/result.json")):
        try:
            result = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        run = next((p for p in path.relative_to(results).parts if p.startswith("run_") and p[4:].isdigit()), "run_1")
        samples.append(_sample(result, int(run[4:])))
    judged = [s for s in samples if "pass@1" in s]
    selected = _select_stratified_samples(judged, n_positive, n_negative)
    selected += [dict(s, is_correct=None) for s in samples if "pass@1" not in s][:n_errors]
    return [Sample(sample_data=s) for s in selected]


def _duration(summary: Dict[str, Any]) -> int:
    try:
        start = datetime.fromisoformat(summary["started_at"])
        end = datetime.fromisoformat(summary["finished_at"])
    except (KeyError, TypeError, ValueError):
        return 0
    return max(0, int((end - start).total_seconds()))


def create_model_evaluation_from_gaia2_run(
    model_name: str,
    work_dir: Path,
    n_positive: int = 2,
    n_negative: int = 3,
    n_errors: int = 3,
) -> Tuple[ModelEvaluation, int]:
    """The ModelEvaluation (and the run's duration in seconds) of a GAIA2 run's work dir."""
    summary = json.loads((work_dir / "summary.json").read_text())
    tasks = [Task(task_name=TASK_PREFIX, metrics=_metrics(summary.get("overall") or {}), samples=[])]
    for language, entry in (summary.get("languages") or {}).items():
        tasks.append(Task(task_name=task_name(language), metrics=_metrics(entry), samples=[]))
    for subset, entry in (summary.get("subsets") or {}).items():
        tasks.append(Task(task_name=task_name(subset), metrics=_metrics(entry), samples=[]))
    for cell, entry in (summary.get("cells") or {}).items():
        language, subset = cell.split("/", 1)
        results = work_dir / "runs" / language / subset / "results"
        samples = _cell_samples(results, n_positive, n_negative, n_errors) if results.is_dir() else []
        tasks.append(Task(task_name=task_name(language, subset), metrics=_metrics(entry), samples=samples))
    return ModelEvaluation(model_name=model_name, tasks=tasks), _duration(summary)
