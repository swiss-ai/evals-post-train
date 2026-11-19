import json
from argparse import ArgumentParser
from pathlib import Path

from .gaia2_wandb_utils import TASK_PREFIX, create_model_evaluation_from_gaia2_run, task_name
from .wandb_alignment_utils import upload_multi_model_results


def main(entity: str, project: str, name: str, work_dir: Path):
    print(f"Uploading {name} from GAIA2 run {work_dir}")

    model_eval, duration = create_model_evaluation_from_gaia2_run(name, work_dir)
    print(f"Created evaluation with {model_eval.total_metrics_count} metrics and {model_eval.total_samples_count} samples")

    # The main table: the overall success rate and each language's (pass@k too with several
    # attempts); per subset and per cell are in the run's metrics.
    languages = json.loads((work_dir / "summary.json").read_text()).get("languages") or {}
    main_tasks = {TASK_PREFIX} | {task_name(language) for language in languages}
    main_metrics = [
        f"{task.task_name}/{m.name}"
        for task in model_eval.tasks
        if task.task_name in main_tasks
        for m in task.metrics
        if not m.name.startswith("n_")
    ]
    upload_multi_model_results(entity, project, [model_eval], main_metrics, duration)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--entity", type=str, required=True, help="WandB entity name")
    parser.add_argument("--project", type=str, required=True, help="WandB project name")
    parser.add_argument("--name", type=str, required=True, help="Name of the model")
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="The run's --work-dir (holds summary.json and runs/).")
    args = parser.parse_args()

    main(entity=args.entity, project=args.project, name=args.name, work_dir=args.work_dir)
