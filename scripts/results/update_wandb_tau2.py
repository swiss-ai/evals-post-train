"""Upload a tau2 run's scores to W&B, and its full results.json as a compressed trace artifact.

The sample tables of other frameworks hold a handful of samples to browse; for tau2 the complete results.json (every
simulation's conversation, tool calls and reward) is attached to the model's W&B run instead: one artifact per model and
task (traces-<model>-<task>), type "traces". The file is text and compresses about 6 times (a 65 MB results.json to
10 MB), and an artifact is only downloaded on demand, unlike a table that every viewer loads. A file whose compressed size
exceeds --max-trace-mb is not uploaded, with a warning: raise the limit on purpose.
"""

import gzip
import shutil
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

from .tau2_wandb_utils import create_model_evaluation_from_tau2_run
from .wandb_names import WANDB_ARTIFACT_NAME_LIMIT, WANDB_ARTIFACT_SUFFIX_RESERVE, _shorten_with_hash, make_wandb_run_id

DEFAULT_MAX_MB = 500


def trace_artifact_name(model_name: str, task_name: str) -> str:
    """The artifact of a model's task: artifacts are versioned by name inside the project, so the model is part of it
    (one collection per model and task, one version per upload)."""
    return _shorten_with_hash(f"traces-{model_name}-{task_name}", WANDB_ARTIFACT_NAME_LIMIT - WANDB_ARTIFACT_SUFFIX_RESERVE)


def compress_trace(path: Path, directory: Path) -> Path:
    """`path` gzipped into `directory` (as <name>.gz)."""
    target = directory / (path.name + ".gz")
    with open(path, "rb") as source, gzip.open(target, "wb", compresslevel=6) as out:
        shutil.copyfileobj(source, out)
    return target


def upload_trace(entity: str, project: str, model_name: str, task_name: str, trace_path: Path,
                 max_mb: int = DEFAULT_MAX_MB) -> Optional[str]:
    """Upload `trace_path` compressed, as the artifact traces-<model>-<task_name> of the model's run; its name, or None when
    the compressed file is over `max_mb`."""
    with tempfile.TemporaryDirectory() as tmp:
        compressed = compress_trace(Path(trace_path), Path(tmp))
        size_mb = compressed.stat().st_size / 1024 / 1024
        if size_mb > max_mb:
            print(f"Not uploading the trace of {task_name}: {size_mb:.0f} MB compressed is over the limit of {max_mb} MB.")
            return None
        import wandb  # only where the upload happens: the eval image has it, the unit tests do not

        name = trace_artifact_name(model_name, task_name)
        with wandb.init(id=make_wandb_run_id(model_name, "-001"), resume="allow", entity=entity, project=project,
                        name=model_name) as run:
            artifact = wandb.Artifact(name, type="traces", metadata={"source": Path(trace_path).name})
            artifact.add_file(str(compressed))
            run.log_artifact(artifact)
            artifact.wait()
        print(f"Uploaded the trace of {task_name}: {size_mb:.1f} MB compressed, artifact {name}")
        return name


def main(entity: str, project: str, name: str, results_file: Path, num_trials: int = 1, traces: bool = True,
         max_trace_mb: int = DEFAULT_MAX_MB):
    print(f"Uploading {name} from tau2 run {results_file}")

    model_eval, duration, problems = create_model_evaluation_from_tau2_run(name, results_file, num_trials)
    if problems:
        print("NOT A COMPLETE RESULT (uploaded as it is; n_unscored says how many tasks have no result): " + "; ".join(problems))
    print(f"Created evaluation with {model_eval.total_metrics_count} metrics")
    task = model_eval.tasks[0]
    print(",".join(f"{task.task_name}/{metric.name}" for metric in task.metrics))

    from .wandb_alignment_utils import upload_multi_model_results  # imports wandb, like upload_trace: not at module level

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
