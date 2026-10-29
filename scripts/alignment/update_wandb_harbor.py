from pathlib import Path
from argparse import ArgumentParser

from .harbor_wandb_utils import create_model_evaluation_from_harbor_run
from .wandb_alignment_utils import upload_multi_model_results


def main(entity: str, project: str, name: str, work_dir: Path):
    print(f"Uploading {name} from Harbor run {work_dir}")

    model_eval, duration = create_model_evaluation_from_harbor_run(name, work_dir)
    print(f"Created evaluation with {model_eval.total_metrics_count} metrics and {model_eval.total_samples_count} samples")
    for task in model_eval.tasks:
        print(",".join([f"{task.task_name}/{metric.name}" for metric in task.metrics]))

    # The main table: each benchmark's reward, or its verifier metrics where it has no single
    # reward (e.g. SlopCodeBench).
    main_metrics = []
    for task in model_eval.tasks:
        names = [m.name for m in task.metrics if not m.name.startswith("n_")]
        main_metrics += [f"{task.task_name}/{n}" for n in (["reward"] if "reward" in names else names)]
    upload_multi_model_results(entity, project, [model_eval], main_metrics, duration)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--entity", type=str, required=True, help="WandB entity name")
    parser.add_argument("--project", type=str, required=True, help="WandB project name")
    parser.add_argument("--name", type=str, required=True, help="Name of the model")
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="The run's --work-dir (holds summary.json and jobs/).")
    args = parser.parse_args()

    main(entity=args.entity, project=args.project, name=args.name, work_dir=args.work_dir)
