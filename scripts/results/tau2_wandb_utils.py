"""Build a ModelEvaluation from a tau2 run (scripts/custom/run_tau2_bench.sh) for W&B upload.

A run's results.json holds every simulation of one domain (the conversation, the reward, why it ended). The
scores are those of scripts/custom/tau2_summary.py, the figures the launcher prints: pass^k and avg_reward over all
tasks, the model's own failed tasks counted as failed (empty replies, context overflows), tasks that failed for
another model's or the infrastructure's reason left out and counted in n_unscored. Like bfcl_wandb_utils this only
adapts the summary into the ModelEvaluation/Task/Metric structures; the upload itself is
wandb_alignment_utils.upload_multi_model_results. No samples go in the table: the full file is attached as a trace
artifact (update_wandb_tau2.py).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

from scripts.custom.tau2_summary import summarize_results

from .data_structures import Metric, ModelEvaluation, Task


def task_name(data: dict) -> str:
    return f"tau2_{data['info']['environment_info']['domain_name']}"


def duration_seconds(data: dict) -> int:
    """Wall clock of the run: from the first simulation's start to the last one's end (0 when unknown)."""
    starts, ends = [], []
    for sim in data.get("simulations") or []:
        try:
            starts.append(datetime.fromisoformat(sim["start_time"]))
            ends.append(datetime.fromisoformat(sim["end_time"]))
        except (KeyError, TypeError, ValueError):
            continue
    return int((max(ends) - min(starts)).total_seconds()) if starts else 0


def create_model_evaluation_from_tau2_run(model_name: str, results_file: Path, num_trials: int = 1) -> Tuple[ModelEvaluation, int, list]:
    """(ModelEvaluation, duration in seconds, problems) of a tau2 results.json.

    One task, tau2_<domain>, with pass^k (fractions 0..1), avg_reward, the numbers of tasks and simulations, the model's
    failed tasks (n_agent_failures, split into n_empty_replies and n_context_overflows), n_unscored (tasks that failed
    for another reason: not counted, to be rerun) and, when some tasks were counted as the model's failure,
    pass^k_excluding_agent_failures, the figure tau2 itself would print. `problems` is why the run is not complete."""
    with open(results_file) as f:
        data = json.load(f)
    summary, problems = summarize_results(data, num_trials)
    if summary is None:
        raise ValueError(f"{results_file}: {problems[0]}")
    metrics = [Metric(name=name, score=score) for name, score in summary.items() if name == "avg_reward" or name.startswith("pass^")]
    by_cause = summary["agent_failures_by_cause"]
    metrics += [
        Metric(name="n_tasks", score=summary["n_tasks"]),
        Metric(name="n_simulations", score=summary["n_simulations"]),
        Metric(name="n_agent_failures", score=summary["n_agent_failures"]),
        Metric(name="n_empty_replies", score=by_cause["empty_reply"]),
        Metric(name="n_context_overflows", score=by_cause["context_overflow"]),
        Metric(name="n_unscored", score=summary["n_unscored"]),
    ]
    for name, score in (summary.get("excluding_agent_failures") or {}).items():
        if name == "avg_reward" or name.startswith("pass^"):
            metrics.append(Metric(name=f"{name}_excluding_agent_failures", score=score))
    task = Task(task_name=task_name(data), metrics=metrics, samples=[])
    return ModelEvaluation(model_name=model_name, tasks=[task]), duration_seconds(data), problems
