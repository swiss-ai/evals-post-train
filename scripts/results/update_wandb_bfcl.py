from argparse import ArgumentParser
from pathlib import Path

from .bfcl_wandb_utils import create_model_evaluation_from_bfcl_run
from .wandb_alignment_utils import upload_multi_model_results


def main(entity: str, project: str, name: str, work_dir: Path):
    print(f"Uploading {name} from BFCL run {work_dir}")

    model_eval, duration = create_model_evaluation_from_bfcl_run(name, work_dir)
    print(f"Created evaluation with {model_eval.total_metrics_count} metrics")
    for task in model_eval.tasks:
        print(",".join([f"{task.task_name}/{metric.name}" for metric in task.metrics]))

    # The main table: bfcl-eval's overall where there is one (v3), else v4's without web search.
    names = [m.name for t in model_eval.tasks for m in t.metrics]
    shown = [n for n in ("overall", "overall_no_web_search") if n in names]
    main_metrics = [f"{t.task_name}/{m.name}" for t in model_eval.tasks for m in t.metrics if m.name in shown]
    upload_multi_model_results(entity, project, [model_eval], main_metrics, duration)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--entity", type=str, required=True, help="WandB entity name")
    parser.add_argument("--project", type=str, required=True, help="WandB project name")
    parser.add_argument("--name", type=str, required=True, help="Name of the model")
    parser.add_argument("--work-dir", type=Path, required=True, help="The run's --workdir (holds summary.json).")
    args = parser.parse_args()

    main(entity=args.entity, project=args.project, name=args.name, work_dir=args.work_dir)
