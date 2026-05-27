"""SWE-bench Verified runner backed by the official SWE-bench harness."""

from __future__ import annotations

import json
import os
import shutil
import sys
import gc
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ..schemas import BenchmarkResult, BenchmarkSpec, RunContext, StandaloneSample


DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_SPLIT = "test"


def run(spec: BenchmarkSpec, context: RunContext) -> BenchmarkResult | None:
    settings = _load_settings(context)
    reference_dir = settings["reference_dir"]
    _prepare_swebench_import(reference_dir)

    work_dir = context.output_dir / "artifacts" / "swebench"
    work_dir.mkdir(parents=True, exist_ok=True)

    phase = str(context.metadata.get("phase") or os.environ.get("STANDALONE_PHASE") or "full")
    predictions_path = settings["predictions_path"]
    if phase == "evaluate" and not predictions_path:
        predictions_path = str(work_dir / "predictions.jsonl")

    if predictions_path:
        normalized_predictions_path = _prepare_predictions(predictions_path, work_dir)
    else:
        normalized_predictions_path = _generate_predictions(spec, context, settings, work_dir)

    if phase == "generate":
        _write_generation_manifest(work_dir, normalized_predictions_path, settings)
        return None

    run_id = settings["run_id"] or context.output_dir.name

    with _working_directory(work_dir):
        report_path = _run_official_harness(
            dataset_name=settings["dataset_name"],
            split=settings["split"],
            predictions_path=normalized_predictions_path,
            run_id=run_id,
            max_workers=settings["max_workers"],
            timeout=settings["timeout"],
            cache_level=settings["cache_level"],
            clean=settings["clean"],
            namespace=settings["namespace"],
            instance_ids=settings["instance_ids"],
            force_rebuild=settings["force_rebuild"],
            rewrite_reports=settings["rewrite_reports"],
            open_file_limit=settings["open_file_limit"],
            instance_image_tag=settings["instance_image_tag"],
            env_image_tag=settings["env_image_tag"],
            modal=settings["modal"],
        )

    predictions = _load_predictions(normalized_predictions_path)
    report = _load_report(report_path, work_dir, predictions, run_id)
    samples = _build_samples(
        spec.name,
        work_dir,
        report,
        predictions,
        run_id,
        settings["dataset_name"],
    )
    metrics = _build_metrics(report)

    return BenchmarkResult(
        benchmark=spec.name,
        metrics=metrics,
        samples=tuple(samples),
        version=spec.version,
        config={
            "dataset_name": settings["dataset_name"],
            "split": settings["split"],
            "run_id": run_id,
            "max_workers": settings["max_workers"],
            "timeout": settings["timeout"],
            "cache_level": settings["cache_level"],
            "clean": settings["clean"],
            "namespace": settings["namespace"],
            "sandbox_backend": context.sandbox_backend,
            "use_podman_cached": settings["use_podman_cached"],
            "local_registry": os.environ.get("LOCAL_REGISTRY"),
        },
        higher_is_better={
            "resolved": True,
            "completed": True,
            "error_rate": False,
            "empty_patch_rate": False,
            "num_instances": True,
        },
        original_samples=int(report.get("submitted_instances") or len(predictions)),
    )


def _load_settings(context: RunContext) -> dict[str, Any]:
    predictions_path = os.environ.get("SWE_PREDICTIONS_PATH")

    instance_ids = _split_env_list(os.environ.get("SWE_INSTANCE_IDS"))
    if context.limit is not None and not instance_ids:
        raise ValueError(
            "SWE-bench --limit is intentionally not applied implicitly. "
            "Set SWE_INSTANCE_IDS for a small run so the selected instances are explicit."
        )

    return {
        "predictions_path": predictions_path,
        "reference_dir": Path(os.environ.get("SWE_BENCH_REFERENCE_DIR", "swe-bench-reference")).resolve(),
        "dataset_name": os.environ.get("SWE_DATASET_NAME", DEFAULT_DATASET),
        "split": os.environ.get("SWE_SPLIT", DEFAULT_SPLIT),
        "run_id": os.environ.get("SWE_RUN_ID"),
        "max_workers": int(os.environ.get("SWE_MAX_WORKERS", "4")),
        "timeout": int(os.environ.get("SWE_TIMEOUT", "1800")),
        "cache_level": os.environ.get("SWE_CACHE_LEVEL", "env"),
        "clean": _env_bool("SWE_CLEAN", False),
        "namespace": _optional_str(os.environ.get("SWE_NAMESPACE", "swebench")),
        "instance_ids": instance_ids,
        "force_rebuild": _env_bool("SWE_FORCE_REBUILD", False),
        "rewrite_reports": _env_bool("SWE_REWRITE_REPORTS", False),
        "open_file_limit": int(os.environ.get("SWE_OPEN_FILE_LIMIT", "4096")),
        "instance_image_tag": os.environ.get("SWE_INSTANCE_IMAGE_TAG", "latest"),
        "env_image_tag": os.environ.get("SWE_ENV_IMAGE_TAG", "latest"),
        "modal": _env_bool("SWE_MODAL", False),
        "use_podman_cached": _env_bool("SWE_USE_PODMAN_CACHED", False),
        "lm_model": os.environ.get("LM_EVAL_BACKEND", "vllm"),
        "lm_model_args": os.environ.get("LM_EVAL_MODEL_ARGS", ""),
        "lm_batch_size": os.environ.get("BS", "auto:20"),
        "lm_max_batch_size": int(os.environ.get("MAX_BATCH_SIZE", "32")),
        "lm_device": os.environ.get("DEVICE"),
        "apply_chat_template": _env_bool("APPLY_CHAT_TEMPLATE", False),
        "generation_max_gen_toks": int(os.environ.get("MAX_NEW_TOKENS", "2048")),
    }


def _prepare_swebench_import(reference_dir: Path) -> None:
    if not (reference_dir / "swebench" / "harness" / "run_evaluation.py").exists():
        raise ValueError(f"SWE_BENCH_REFERENCE_DIR does not look like SWE-bench: {reference_dir}")
    ref = str(reference_dir)
    if ref not in sys.path:
        sys.path.insert(0, ref)


def _prepare_predictions(predictions_path: str, work_dir: Path) -> str:
    if predictions_path == "gold":
        return "gold"

    source = Path(predictions_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"SWE_PREDICTIONS_PATH does not exist: {source}")
    destination = work_dir / source.name
    if source != destination:
        shutil.copy2(source, destination)
    return str(destination)


def _generate_predictions(
    spec: BenchmarkSpec,
    context: RunContext,
    settings: dict[str, Any],
    work_dir: Path,
) -> str:
    from lm_eval.api.instance import Instance
    from lm_eval.api.registry import get_model
    from swebench.harness.utils import load_swebench_dataset

    dataset = load_swebench_dataset(
        settings["dataset_name"],
        settings["split"],
        settings["instance_ids"],
    )
    if not dataset:
        raise ValueError("No SWE-bench instances selected for prediction generation.")

    lm = get_model(settings["lm_model"]).create_from_arg_string(
        settings["lm_model_args"],
        {
            "batch_size": settings["lm_batch_size"],
            "max_batch_size": settings["lm_max_batch_size"],
            "device": settings["lm_device"],
        },
    )

    prompts = [_build_prompt(instance) for instance in dataset]
    if settings["apply_chat_template"]:
        prompts = [_apply_chat_template(lm, prompt) for prompt in prompts]

    gen_kwargs = {
        "max_gen_toks": settings["generation_max_gen_toks"],
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
    }
    requests = [
        Instance(
            request_type="generate_until",
            doc=instance,
            arguments=(prompt, gen_kwargs),
            idx=index,
            metadata=(spec.name, index, 1),
        )
        for index, (instance, prompt) in enumerate(zip(dataset, prompts, strict=True))
    ]
    generations = lm.generate_until(requests)

    predictions_path = work_dir / "predictions.jsonl"
    with predictions_path.open("w") as handle:
        for instance, generation in zip(dataset, generations, strict=True):
            patch = _extract_patch(generation)
            handle.write(
                json.dumps(
                    {
                        "instance_id": instance["instance_id"],
                        "model_name_or_path": context.name,
                        "model_patch": patch,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    del lm
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return str(predictions_path)


def _write_generation_manifest(
    work_dir: Path,
    predictions_path: str,
    settings: dict[str, Any],
) -> None:
    manifest = {
        "predictions_path": predictions_path,
        "dataset_name": settings["dataset_name"],
        "split": settings["split"],
        "instance_ids": settings["instance_ids"],
    }
    with (work_dir / "generation_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _build_prompt(instance: dict[str, Any]) -> str:
    hints = instance.get("hints_text") or ""
    hints_block = f"\nHints:\n{hints}\n" if hints else ""
    return (
        "You are fixing a real GitHub issue. Produce a minimal unified diff patch.\n"
        "Return only the patch. Do not include prose, markdown fences, or explanations.\n\n"
        f"Repository: {instance.get('repo')}\n"
        f"Base commit: {instance.get('base_commit')}\n\n"
        f"Issue:\n{instance.get('problem_statement', '')}\n"
        f"{hints_block}\n"
        "Patch:\n"
    )


def _apply_chat_template(lm: Any, prompt: str) -> str:
    return lm.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
    )


def _extract_patch(generation: str) -> str:
    text = generation.strip()
    if "```" in text:
        parts = text.split("```")
        code_blocks = [part for index, part in enumerate(parts) if index % 2 == 1]
        if code_blocks:
            text = code_blocks[0]
            lines = text.splitlines()
            if lines and lines[0].strip().lower() in {"diff", "patch"}:
                text = "\n".join(lines[1:])
            text = text.strip()

    diff_start_markers = ["diff --git ", "--- "]
    starts = [text.find(marker) for marker in diff_start_markers if text.find(marker) >= 0]
    if starts:
        text = text[min(starts) :]
    return text


def _run_official_harness(
    *,
    dataset_name: str,
    split: str,
    predictions_path: str,
    run_id: str,
    max_workers: int,
    timeout: int,
    cache_level: str,
    clean: bool,
    namespace: str | None,
    instance_ids: list[str] | None,
    force_rebuild: bool,
    rewrite_reports: bool,
    open_file_limit: int,
    instance_image_tag: str,
    env_image_tag: str,
    modal: bool,
) -> Path | None:
    from swebench.harness.run_evaluation import main as run_evaluation_main

    return run_evaluation_main(
        dataset_name=dataset_name,
        split=split,
        instance_ids=instance_ids,
        predictions_path=predictions_path,
        max_workers=max_workers,
        force_rebuild=force_rebuild,
        cache_level=cache_level,
        clean=clean,
        open_file_limit=open_file_limit,
        run_id=run_id,
        timeout=timeout,
        namespace=namespace,
        rewrite_reports=rewrite_reports,
        modal=modal,
        instance_image_tag=instance_image_tag,
        env_image_tag=env_image_tag,
        report_dir=".",
    )


def _load_predictions(predictions_path: str) -> dict[str, dict[str, Any]]:
    if predictions_path == "gold":
        return {}

    path = Path(predictions_path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif path.suffix == ".json":
        payload = json.loads(path.read_text())
        rows = list(payload.values()) if isinstance(payload, dict) else payload
    else:
        rows = []
    return {row["instance_id"]: row for row in rows}


def _load_report(
    report_path: Path | None,
    work_dir: Path,
    predictions: dict[str, dict[str, Any]],
    run_id: str,
) -> dict[str, Any]:
    if report_path and report_path.exists():
        return json.loads(report_path.read_text())

    model_names = {row.get("model_name_or_path", "None").replace("/", "__") for row in predictions.values()}
    for model_name in sorted(model_names):
        candidate = work_dir / f"{model_name}.{run_id}.json"
        if candidate.exists():
            return json.loads(candidate.read_text())

    candidates = sorted(work_dir.glob(f"*.{run_id}.json"))
    if candidates:
        return json.loads(candidates[0].read_text())

    raise FileNotFoundError(f"Could not locate SWE-bench report for run_id={run_id} in {work_dir}")


def _build_metrics(report: dict[str, Any]) -> dict[str, float]:
    submitted = int(report.get("submitted_instances", 0))
    completed = int(report.get("completed_instances", 0))
    resolved = int(report.get("resolved_instances", 0))
    errors = int(report.get("error_instances", 0))
    empty = int(report.get("empty_patch_instances", 0))
    denominator = submitted or 1
    return {
        "resolved": resolved / denominator,
        "completed": completed / denominator,
        "error_rate": errors / denominator,
        "empty_patch_rate": empty / denominator,
        "num_instances": float(submitted),
    }


def _build_samples(
    benchmark_name: str,
    work_dir: Path,
    report: dict[str, Any],
    predictions: dict[str, dict[str, Any]],
    run_id: str,
    dataset_name: str,
) -> list[StandaloneSample]:
    resolved_ids = set(report.get("resolved_ids", []))
    unresolved_ids = set(report.get("unresolved_ids", []))
    error_ids = set(report.get("error_ids", []))
    empty_patch_ids = set(report.get("empty_patch_ids", []))
    completed_ids = set(report.get("completed_ids", []))
    submitted_ids = list(report.get("submitted_ids", sorted(predictions)))

    samples = []
    for instance_id in submitted_ids:
        prediction = predictions.get(instance_id, {})
        model_name = prediction.get("model_name_or_path", "None").replace("/", "__")
        log_dir = Path("artifacts") / "swebench" / "logs" / "run_evaluation" / run_id / model_name / instance_id
        local_log_dir = work_dir / "logs" / "run_evaluation" / run_id / model_name / instance_id
        report_file = local_log_dir / "report.json"
        patch_file = local_log_dir / "patch.diff"
        test_output_file = local_log_dir / "test_output.txt"
        run_log_file = local_log_dir / "run_instance.log"

        status = "resolved" if instance_id in resolved_ids else "unresolved"
        if instance_id in error_ids:
            status = "error"
        if instance_id in empty_patch_ids:
            status = "empty_patch"

        artifact_paths = {}
        for key, relative, absolute in (
            ("report", log_dir / "report.json", report_file),
            ("patch", log_dir / "patch.diff", patch_file),
            ("test_output", log_dir / "test_output.txt", test_output_file),
            ("run_log", log_dir / "run_instance.log", run_log_file),
        ):
            if absolute.exists():
                artifact_paths[key] = str(relative)

        samples.append(
            StandaloneSample(
                benchmark=benchmark_name,
                instance_id=instance_id,
                status=status,
                metrics={
                    "resolved": 1.0 if instance_id in resolved_ids else 0.0,
                    "completed": 1.0 if instance_id in completed_ids else 0.0,
                },
                input={
                    "dataset": dataset_name,
                    "instance_id": instance_id,
                },
                prediction={
                    "model_name_or_path": prediction.get("model_name_or_path"),
                    "patch_chars": len(prediction.get("model_patch") or ""),
                },
                trajectory_summary={
                    "run_id": run_id,
                    "status": status,
                },
                artifact_paths=artifact_paths,
                error=status if status == "error" else None,
            )
        )
    return samples


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _optional_str(value: str | None) -> str | None:
    if value is None or value.lower() == "none":
        return None
    return value


def _split_env_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.replace(",", " ").split() if item.strip()]
