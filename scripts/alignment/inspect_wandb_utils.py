"""Build ModelEvaluation objects from Inspect AI (.eval) log files for W&B upload.

Inspect's log schema is unrelated to lm-eval-harness's results_*.json/samples_*.jsonl --
this is a separate adapter into the same ModelEvaluation/Task/Metric/Sample structures used by
wandb_alignment_utils.upload_multi_model_results, so the actual W&B upload logic (run naming,
main table, sample tables) is shared rather than reimplemented.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from inspect_ai.log import EvalLog, EvalSample, read_eval_log

from .data_structures import Metric, ModelEvaluation, Sample, Task
from .wandb_alignment_utils import _parse_metric_score, _select_stratified_samples

# Inspect scorers commonly report a categorical Score.value instead of a number (see
# inspect_ai.scorer.{CORRECT,INCORRECT,PARTIAL,NOANSWER}).
_CATEGORICAL_SCORE_VALUES = {"C": 1.0, "I": 0.0, "P": 0.5, "N": 0.0}


def _score_value_to_float(value) -> float | None:
    """Convert an Inspect Score.value to a float, or None if it isn't classifiable."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if value.upper() in _CATEGORICAL_SCORE_VALUES:
            return _CATEGORICAL_SCORE_VALUES[value.upper()]
        try:
            return float(value)
        except ValueError:
            return None
    # Score.value can also be a list/dict for multi-part scorers -- not classifiable as a
    # single correct/incorrect signal, so leave it out of stratified sampling.
    return None


def _summarize_inspect_sample(sample: EvalSample) -> dict:
    """A bounded per-sample summary for the W&B samples table.

    Deliberately excludes the full message/event transcript (sample.messages, .events,
    .events_data, .store) -- those can run into the MBs for agentic tasks like tau2-bench and
    aren't useful flattened into a table cell.
    """
    summary = {
        "id": sample.id,
        "epoch": sample.epoch,
        "target": sample.target,
        "output": sample.output.completion if sample.output else None,
        "total_time": sample.total_time,
        "turn_count": sample.turn_count,
        "error": str(sample.error) if sample.error else None,
        "metrics": [],
    }
    for scorer_name, score in (sample.scores or {}).items():
        summary[scorer_name] = score.value
        if score.explanation:
            summary[f"{scorer_name}_explanation"] = score.explanation
        if _score_value_to_float(score.value) is not None:
            summary["metrics"].append(scorer_name)
    return summary


def _model_evaluation_from_eval_logs(
    model_name: str,
    eval_logs: List[EvalLog],
    n_positive: int = 2,
    n_negative: int = 3,
) -> Tuple[ModelEvaluation, int]:
    """Build a ModelEvaluation (and overall duration in seconds) from parsed EvalLogs."""
    tasks = []
    started_at, completed_at = None, None

    for log in eval_logs:
        task_name = log.eval.task
        if task_name.startswith("inspect_evals/"):
            task_name = task_name[len("inspect_evals/") :]

        scores = log.results.scores if log.results else []
        multi_scorer = len(scores) > 1
        metrics = []
        for score in scores:
            for metric_name, metric in score.metrics.items():
                key = f"{score.name}/{metric_name}" if multi_scorer else metric_name
                value = _parse_metric_score(task_name, key, metric.value)
                if value is not None:
                    metrics.append(Metric(name=key, score=value))

        sample_dicts = [_summarize_inspect_sample(s) for s in (log.samples or [])]
        selected = _select_stratified_samples(sample_dicts, n_positive, n_negative)
        task_samples = [Sample(sample_data=s) for s in selected]

        tasks.append(Task(task_name=task_name, metrics=metrics, samples=task_samples))

        if log.stats:
            log_start = datetime.fromisoformat(log.stats.started_at)
            log_end = datetime.fromisoformat(log.stats.completed_at)
            started_at = log_start if started_at is None else min(started_at, log_start)
            completed_at = log_end if completed_at is None else max(completed_at, log_end)

    duration = int((completed_at - started_at).total_seconds()) if started_at and completed_at else 0
    return ModelEvaluation(model_name=model_name, tasks=tasks), duration


def create_model_evaluation_from_inspect_logs(
    model_name: str,
    eval_log_paths: List[Path],
    n_positive: int = 2,
    n_negative: int = 3,
) -> Tuple[ModelEvaluation, int]:
    """Build a ModelEvaluation (and overall duration in seconds) from .eval log files."""
    eval_logs = [read_eval_log(str(path)) for path in eval_log_paths]
    return _model_evaluation_from_eval_logs(model_name, eval_logs, n_positive, n_negative)
