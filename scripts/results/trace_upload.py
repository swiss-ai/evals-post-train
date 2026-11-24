"""Attach a run's full trace file (every sample's conversation or generation) to its W&B run as a compressed artifact.

The sample tables of an upload hold a handful of samples to browse; this is the complete file, kept next to the scores
of the same run (one artifact per task, type "traces", in the model's W&B run). The files are text and compress about
6 to 25 times (a 65 MB tau2 results.json to 10 MB, a 245 MB lm-eval samples file to 10 MB), and an artifact is only
downloaded on demand, unlike a table that every viewer loads. A file whose compressed size exceeds `max_mb` is not
uploaded, with a warning: raise the limit on purpose.
"""

import gzip
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from .wandb_names import make_wandb_run_id

DEFAULT_MAX_MB = 500


def compress_trace(path: Path, directory: Path) -> Path:
    """`path` gzipped into `directory` (as <name>.gz)."""
    target = directory / (path.name + ".gz")
    with open(path, "rb") as source, gzip.open(target, "wb", compresslevel=6) as out:
        shutil.copyfileobj(source, out)
    return target


def upload_trace(
    entity: str,
    project: str,
    model_name: str,
    task_name: str,
    trace_path: Path,
    max_mb: int = DEFAULT_MAX_MB,
    metadata: Optional[dict] = None,
) -> Optional[str]:
    """Upload `trace_path` compressed, as the artifact traces-<task_name> of the model's run; its name, or None when
    the compressed file is over `max_mb`."""
    with tempfile.TemporaryDirectory() as tmp:
        compressed = compress_trace(Path(trace_path), Path(tmp))
        size_mb = compressed.stat().st_size / 1024 / 1024
        if size_mb > max_mb:
            print(f"Not uploading the trace of {task_name}: {size_mb:.0f} MB compressed is over the limit of {max_mb} MB.")
            return None
        import wandb  # only where the upload happens: the eval image has it, the unit tests do not

        name = f"traces-{task_name}"
        with wandb.init(id=make_wandb_run_id(model_name, "-001"), resume="allow", entity=entity, project=project,
                        name=model_name) as run:
            artifact = wandb.Artifact(name, type="traces", metadata={"source": trace_path.name, **(metadata or {})})
            artifact.add_file(str(compressed))
            run.log_artifact(artifact)
            artifact.wait()
        print(f"Uploaded the trace of {task_name}: {size_mb:.1f} MB compressed, artifact {name}")
        return name
