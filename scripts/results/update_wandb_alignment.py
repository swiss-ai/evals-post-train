from pathlib import Path
from argparse import ArgumentParser

from .wandb_alignment_utils import upload_multi_model_results, create_model_evaluation_from_results


def main(entity: str, project: str, name: str, main_metrics: list, logs_root: Path, eval_duration: int):
    print(f"Uploading {name}, iteration: {logs_root.name}")
    
    # Create ModelEvaluation directly from results and samples
    model_eval = create_model_evaluation_from_results(name, logs_root)
    print(f"Created evaluation with {model_eval.total_metrics_count} metrics and {model_eval.total_samples_count} samples")
    for task in model_eval.tasks:
        print(",".join([f"{task.task_name}/{metric.name}" for metric in task.metrics]))
    
    # Upload using the new structured approach
    upload_multi_model_results(entity, project, [model_eval], main_metrics, eval_duration)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--entity", type=str, required=True, help="WandB entity name")
    parser.add_argument("--project", type=str, required=True, help="WandB project name")
    parser.add_argument("--name", type=str, required=True, help="Name of the model")
    parser.add_argument("--main_metrics", nargs='+', type=str, required=True, help="List of metrics for main table")
    parser.add_argument("--logs_root", type=Path, required=True, help="Root directory containing evaluation logs")
    parser.add_argument("--eval_duration", type=int, required=True, help="Evaluation duration in seconds")
    args = parser.parse_args()

    main(entity=args.entity, project=args.project, name=args.name, main_metrics=args.main_metrics, logs_root=args.logs_root, eval_duration=args.eval_duration)
