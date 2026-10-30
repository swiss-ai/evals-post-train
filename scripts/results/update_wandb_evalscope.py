from pathlib import Path
from argparse import ArgumentParser

from .evalscope_wandb_utils import create_model_evaluation_from_evalscope_run
from .wandb_alignment_utils import upload_multi_model_results


def main(entity: str, project: str, name: str, work_dir: Path):
    print(f"Uploading {name} from EvalScope run {work_dir}")

    model_eval, duration = create_model_evaluation_from_evalscope_run(name, work_dir)
    print(f"Created evaluation with {model_eval.total_metrics_count} metrics and {model_eval.total_samples_count} samples")
    for task in model_eval.tasks:
        print(",".join([f"{task.task_name}/{metric.name}" for metric in task.metrics]))

    # The main table: each benchmark's accuracy (the share of instances resolved).
    main_metrics = [f"{t.task_name}/{m.name}" for t in model_eval.tasks for m in t.metrics if m.name == "accuracy"]
    upload_multi_model_results(entity, project, [model_eval], main_metrics, duration)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--entity", type=str, required=True, help="WandB entity name")
    parser.add_argument("--project", type=str, required=True, help="WandB project name")
    parser.add_argument("--name", type=str, required=True, help="Name of the model")
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="The run's --work-dir (holds summary.json and reviews/).")
    args = parser.parse_args()

    main(entity=args.entity, project=args.project, name=args.name, work_dir=args.work_dir)
