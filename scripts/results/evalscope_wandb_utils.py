"""Build ModelEvaluation objects from an EvalScope run (scripts/evalscope/run_evalscope_eval.sh)
for W&B upload.

A run's work dir holds summary.json (written by scripts.evalscope.run: one EvalScope report per
benchmark, and the samples whose evaluation failed) and reviews/<model>/<dataset>_<subset>.jsonl
(EvalScope's per-sample records). Like harbor_wandb_utils, this only adapts them into the
ModelEvaluation/Task/Metric/Sample structures; the upload itself is
wandb_alignment_utils.upload_multi_model_results.
"""

import json
from pathlib import Path
from typing import List, Optional, Set, Tuple

from .data_structures import Metric, ModelEvaluation, Sample, Task
from .wandb_alignment_utils import _select_stratified_samples


def _required_tests(report: dict) -> Tuple[Optional[int], Optional[int], List[str]]:
    """(tests passed, tests required, the required ones that did not pass) from a sample's
    scoring report, which SWE-bench Pro and SWE-bench Verified shape differently."""
    if "f2p" in report or "p2p" in report:  # SWE-bench Pro: tests lists only the tests that ran
        required = list(report.get("f2p") or []) + list(report.get("p2p") or [])
        passed = {t.get("name") for t in report.get("tests") or [] if t.get("status") == "PASSED"}
        failed = [t for t in required if t not in passed]
        return len(required) - len(failed), len(required), failed
    # SWE-bench Verified: {instance_id: {"tests_status": {"FAIL_TO_PASS": {"success", "failure"}, ...}}}
    for result in report.values():
        if isinstance(result, dict) and "tests_status" in result:
            groups = (result["tests_status"] or {}).values()
            passed = sum(len(g.get("success") or []) for g in groups)
            failed = [t for g in groups for t in g.get("failure") or []]
            return passed, passed + len(failed), failed
    return None, None, []


def _summarize_sample(record: dict, errors: Set[str]) -> dict:
    """A bounded per-sample summary for the W&B samples table (no agent transcript)."""
    sample_score = record.get("sample_score") or {}
    score = sample_score.get("score") or {}
    result = score.get("metadata") or {}
    report = result.get("report") or {}
    meta = sample_score.get("sample_metadata") or {}
    usage = (record.get("agent_trace") or {}).get("total_usage") or {}
    instance = meta.get("instance_id") or sample_score.get("sample_id")
    tests_passed, tests_required, failed = _required_tests(report)
    summary = {
        "instance_id": instance,
        "repo": meta.get("repo"),
        "resolved": result.get("resolved"),
        "tests_passed": tests_passed,
        "tests_required": tests_required,
        "failed_tests": failed[:20],
        "error": str(report.get("error"))[:500] if report.get("error") else None,
        "n_input_tokens": usage.get("input_tokens"),
        "n_output_tokens": usage.get("output_tokens"),
        "patch": (score.get("extracted_prediction") or "")[:5000],
        "metrics": [],
    }
    # An evaluation error is scored unresolved by EvalScope, but it isn't the model's failure:
    # keep it out of the passed/failed samples.
    if instance not in errors and (score.get("value") or {}).get("acc") in (0, 1):
        summary["acc"] = float(score["value"]["acc"])
        summary["metrics"].append("acc")
    return summary


def _sample_times(record: dict) -> List[float]:
    events = (record.get("agent_trace") or {}).get("events") or []
    return [e["timestamp"] for e in events if isinstance(e.get("timestamp"), (int, float))]


def create_model_evaluation_from_evalscope_run(
    model_name: str,
    work_dir: Path,
    n_positive: int = 2,
    n_negative: int = 3,
    n_errors: int = 3,
) -> Tuple[ModelEvaluation, int]:
    """Build a ModelEvaluation (and overall duration in seconds) from an EvalScope run's work dir.

    Per benchmark: EvalScope's metrics (accuracy = share of instances resolved), n_samples and
    n_errors -- the samples whose evaluation failed, which accuracy counts as unresolved, so
    n_errors > 0 means the score is too low. Samples: stratified resolved/unresolved instances
    plus up to n_errors failed evaluations."""
    with open(work_dir / "summary.json") as f:
        summary = json.load(f)
    # EvalScope's dataset name -> ours (swe_bench_verified_agentic -> swe_bench).
    our_name = {evalscope: ours for ours, evalscope in summary.get("benchmarks", {}).items()}
    errors_by_file = summary.get("evaluation_errors") or {}
    tasks = []
    times: List[float] = []
    for report in summary["reports"]:
        dataset, model_dir = report["dataset_name"], report.get("model_name")
        metrics = [Metric(name=m["identity"]["name"], score=m["score"]) for m in report.get("metrics") or []]
        num = max((m.get("num") or 0 for m in report.get("metrics") or []), default=0)
        review_files = sorted((work_dir / "reviews" / str(model_dir)).glob(f"{dataset}_*.jsonl"))
        error_keys = [f"{model_dir}/{p.stem}" for p in review_files]
        errors = {i for key in error_keys for i in (errors_by_file.get(key) or {}).get("instances", [])}
        metrics += [Metric(name="n_samples", score=num), Metric(name="n_errors", score=len(errors))]
        samples = []
        for path in review_files:
            with open(path) as f:
                for line in f:
                    record = json.loads(line)
                    samples.append(_summarize_sample(record, errors))
                    times += _sample_times(record)
        selected = _select_stratified_samples([s for s in samples if s["metrics"]], n_positive, n_negative)
        selected += [dict(s, is_correct=None) for s in samples if not s["metrics"]][:n_errors]
        tasks.append(Task(task_name=our_name.get(dataset, dataset), metrics=metrics,
                          samples=[Sample(sample_data=s) for s in selected]))
    duration = int(max(times) - min(times)) if times else 0
    return ModelEvaluation(model_name=model_name, tasks=tasks), duration

