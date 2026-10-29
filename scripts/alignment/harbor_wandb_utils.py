"""Build ModelEvaluation objects from a Harbor run (scripts/run_harbor_eval.sh) for W&B upload.

A run's work dir holds summary.json (one entry per benchmark, written by
scripts.harbor_runner.run) and jobs/<benchmark>/<trial>/result.json (Harbor's own trial
records). Like inspect_wandb_utils, this only adapts them into the ModelEvaluation/Task/
Metric/Sample structures; the upload itself is wandb_alignment_utils.upload_multi_model_results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from .data_structures import Metric, ModelEvaluation, Sample, Task
from .wandb_alignment_utils import _select_stratified_samples


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def _summarize_trial(trial: dict) -> dict:
    """A bounded per-trial summary for the W&B samples table (no trajectories)."""
    rewards = (trial.get("verifier_result") or {}).get("rewards") or {}
    agent = trial.get("agent_result") or {}
    exc = trial.get("exception_info") or {}
    started, finished = _parse_time(trial.get("started_at")), _parse_time(trial.get("finished_at"))
    summary = {
        "task": trial.get("task_name"),
        "trial": trial.get("trial_name"),
        "rewards": rewards,
        "error": f"{exc.get('exception_type')}: {(exc.get('exception_message') or '')[:500]}" if exc else None,
        "n_input_tokens": agent.get("n_input_tokens"),
        "n_output_tokens": agent.get("n_output_tokens"),
        "duration_sec": int((finished - started).total_seconds()) if started and finished else None,
        "metrics": [],
    }
    # A binary reward is a single attempt's pass/fail: classify it for stratified sampling.
    if rewards.get("reward") in (0, 1):
        summary["pass@1"] = float(rewards["reward"])
        summary["metrics"].append("pass@1")
    return summary


def create_model_evaluation_from_harbor_run(
    model_name: str,
    work_dir: Path,
    n_positive: int = 2,
    n_negative: int = 3,
    n_errors: int = 3,
) -> Tuple[ModelEvaluation, int]:
    """Build a ModelEvaluation (and overall duration in seconds) from a Harbor run's work dir.

    Per benchmark: reward (mean over the scored trials) and every other verifier metric's mean,
    plus n_trials, n_scored and n_errors -- n_errors > 0 means the score is incomplete. Samples:
    stratified passed/failed trials plus up to n_errors unscored ones."""
    with open(work_dir / "summary.json") as f:
        summary = json.load(f)
    tasks = []
    started_at, finished_at = None, None
    for name, result in summary["benchmarks"].items():
        values = result.get("metrics")
        if values is None:  # summary.json from before per-metric means: mean_reward only
            values = {} if result.get("mean_reward") is None else {"reward": result["mean_reward"]}
        metrics = [Metric(name=key, score=value) for key, value in values.items()]
        metrics += [Metric(name=f"pass@{k}", score=v) for k, v in result.get("pass_at_k", {}).items()]
        metrics += [
            Metric(name="n_trials", score=result.get("n_trials", 0)),
            Metric(name="n_scored", score=result.get("n_scored", 0)),
            Metric(name="n_errors", score=len(result.get("evaluation_errors", []))),
        ]
        trials = []
        for path in sorted((work_dir / "jobs" / name).glob("*/result.json")):
            with open(path) as f:
                trial = json.load(f)
            trials.append(_summarize_trial(trial))
            start, end = _parse_time(trial.get("started_at")), _parse_time(trial.get("finished_at"))
            if start:
                started_at = start if started_at is None else min(started_at, start)
            if end:
                finished_at = end if finished_at is None else max(finished_at, end)
        # Unscored trials (no verifier reward) are what to look at first; the stratified
        # selection drops them whenever some trials were scored.
        scored = [t for t in trials if t["rewards"]]
        selected = _select_stratified_samples(scored, n_positive, n_negative)
        unscored = [dict(t, is_correct=None) for t in trials if not t["rewards"]]
        selected += unscored[:n_errors]
        tasks.append(Task(task_name=name, metrics=metrics, samples=[Sample(sample_data=s) for s in selected]))
    duration = int((finished_at - started_at).total_seconds()) if started_at and finished_at else 0
    return ModelEvaluation(model_name=model_name, tasks=tasks), duration
