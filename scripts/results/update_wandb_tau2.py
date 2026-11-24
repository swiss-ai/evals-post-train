from argparse import ArgumentParser
from pathlib import Path

from .tau2_wandb_utils import create_model_evaluation_from_tau2_run
from .trace_upload import DEFAULT_MAX_MB, upload_trace
from .wandb_alignment_utils import upload_multi_model_results


def main(entity: str, project: str, name: str, results_file: Path, num_trials: int = 1, traces: bool = True,
         max_trace_mb: int = DEFAULT_MAX_MB):
    print(f"Uploading {name} from tau2 run {results_file}")

    model_eval, duration, problems = create_model_evaluation_from_tau2_run(name, results_file, num_trials)
    if problems:
        print("NOT A COMPLETE RESULT (uploaded as it is; n_unscored says how many tasks have no result): " + "; ".join(problems))
    print(f"Created evaluation with {model_eval.total_metrics_count} metrics")
    task = model_eval.tasks[0]
    print(",".join(f"{task.task_name}/{metric.name}" for metric in task.metrics))

    # The main table: pass^1, the score of one trial per task.
    upload_multi_model_results(entity, project, [model_eval], [f"{task.task_name}/pass^1"], duration)
    if traces:
        upload_trace(entity, project, name, task.task_name, results_file, max_trace_mb)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--entity", type=str, required=True, help="WandB entity name")
    parser.add_argument("--project", type=str, required=True, help="WandB project name")
    parser.add_argument("--name", type=str, required=True, help="Name of the model")
    parser.add_argument("--results-file", type=Path, required=True, help="The run's results.json (tau2-data/simulations/<save-to>/).")
    parser.add_argument("--num-trials", type=int, default=1, help="Trials per task of the run (default 1).")
    parser.add_argument("--no-traces", action="store_true", help="Do not attach the full results.json as a compressed artifact.")
    parser.add_argument("--max-trace-mb", type=int, default=DEFAULT_MAX_MB, help=f"Largest compressed trace to upload (default {DEFAULT_MAX_MB}).")
    args = parser.parse_args()

    main(entity=args.entity, project=args.project, name=args.name, results_file=args.results_file,
         num_trials=args.num_trials, traces=not args.no_traces, max_trace_mb=args.max_trace_mb)
