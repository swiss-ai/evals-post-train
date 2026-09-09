from pathlib import Path
from argparse import ArgumentParser

from .inspect_wandb_utils import create_model_evaluation_from_inspect_logs
from .wandb_alignment_utils import upload_multi_model_results


def main(entity: str, project: str, name: str, eval_logs: list):
    print(f"Uploading {name} from {len(eval_logs)} Inspect .eval log(s)")

    model_eval, duration = create_model_evaluation_from_inspect_logs(name, eval_logs)
    print(f"Created evaluation with {model_eval.total_metrics_count} metrics and {model_eval.total_samples_count} samples")
    for task in model_eval.tasks:
        print(",".join([f"{task.task_name}/{metric.name}" for metric in task.metrics]))

    main_metrics = sorted(model_eval.get_flattened_metrics().keys())
    upload_multi_model_results(entity, project, [model_eval], main_metrics, duration)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--entity", type=str, required=True, help="WandB entity name")
    parser.add_argument("--project", type=str, required=True, help="WandB project name")
    parser.add_argument("--name", type=str, required=True, help="Name of the model")
    parser.add_argument("--eval-log", dest="eval_logs", type=Path, action="append", required=True,
                         help="Path to an Inspect .eval log file. Repeatable.")
    args = parser.parse_args()

    main(entity=args.entity, project=args.project, name=args.name, eval_logs=args.eval_logs)
