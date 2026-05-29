"""SWE-bench Verified runner backed by the official SWE-bench harness."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
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
    if settings["evaluator"] == "official":
        _prepare_swebench_import(settings["reference_dir"])

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

    predictions = _load_predictions(normalized_predictions_path)
    _print_prediction_summary(normalized_predictions_path, predictions)

    if phase == "generate":
        _write_generation_manifest(work_dir, normalized_predictions_path, settings)
        return None

    run_id = settings["run_id"] or context.output_dir.name

    if settings["evaluator"] == "fast":
        with _working_directory(work_dir):
            report_path = _run_fast_harness(
                dataset_name=settings["dataset_name"],
                split=settings["split"],
                predictions_path=normalized_predictions_path,
                run_id=run_id,
                max_workers=settings["max_workers"],
                timeout=settings["timeout"],
                instance_ids=settings["instance_ids"],
                work_dir=work_dir,
                allow_x86_emulation=settings["allow_x86_emulation"],
            )
        report = _load_fast_report(report_path, predictions)
    elif settings["evaluator"] == "official":
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
        report = _load_report(report_path, work_dir, predictions, run_id)
    else:
        raise ValueError(f"Unsupported SWE_EVALUATOR: {settings['evaluator']}")
    _print_report_error_summary(work_dir, report, predictions, run_id)
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
            "evaluator": settings["evaluator"],
            "max_workers": settings["max_workers"],
            "timeout": settings["timeout"],
            "cache_level": settings["cache_level"],
            "clean": settings["clean"],
            "namespace": settings["namespace"],
            "arch": settings["arch"],
            "relax_conda_builds": settings["relax_conda_builds"],
            "relax_conda_package_pins": settings["relax_conda_package_pins"],
            "sandbox_backend": context.sandbox_backend,
            "swe_bench_fast_bin": settings["fast_bin"],
            "allow_x86_emulation": settings["allow_x86_emulation"],
            "use_podman_build": settings["use_podman_build"],
            "use_podman_cached": settings["use_podman_cached"],
            "podman_build_storage_opts": settings["podman_build_storage_opts"],
            "local_registry": os.environ.get("LOCAL_REGISTRY"),
        },
        higher_is_better={
            "resolved": True,
            "completed": True,
            "error_rate": False,
            "empty_patch_rate": False,
            "unsupported_x86_on_arm": False,
            "num_instances": True,
            "num_requested_instances": True,
        },
        original_samples=int(report.get("requested_instances") or len(predictions)),
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
        "evaluator": os.environ.get("SWE_EVALUATOR", "official").lower(),
        "fast_bin": os.environ.get("SWE_BENCH_FAST_BIN", "swe-bench-fast"),
        "allow_x86_emulation": _env_bool("SWE_BENCH_FAST_ALLOW_X86_EMULATION", False),
        "max_workers": _default_max_workers(),
        "timeout": int(os.environ.get("SWE_TIMEOUT", "1800")),
        "cache_level": os.environ.get("SWE_CACHE_LEVEL", "env"),
        "clean": _env_bool("SWE_CLEAN", False),
        "namespace": _optional_str(os.environ.get("SWE_NAMESPACE", "none")),
        "arch": os.environ.get("SWE_ARCH", "x86_64"),
        "relax_conda_builds": os.environ.get("SWE_RELAX_CONDA_BUILDS", "auto"),
        "relax_conda_package_pins": os.environ.get("SWE_RELAX_CONDA_PACKAGE_PINS", "setuptools pip python"),
        "instance_ids": instance_ids,
        "force_rebuild": _env_bool("SWE_FORCE_REBUILD", False),
        "rewrite_reports": _env_bool("SWE_REWRITE_REPORTS", False),
        "open_file_limit": int(os.environ.get("SWE_OPEN_FILE_LIMIT", "4096")),
        "instance_image_tag": os.environ.get("SWE_INSTANCE_IMAGE_TAG", "latest"),
        "env_image_tag": os.environ.get("SWE_ENV_IMAGE_TAG", "latest"),
        "modal": _env_bool("SWE_MODAL", False),
        "use_podman_build": _env_bool("SWE_USE_PODMAN_BUILD", False),
        "use_podman_cached": _env_bool("SWE_USE_PODMAN_CACHED", False),
        "podman_build_storage_opts": os.environ.get("SWE_PODMAN_BUILD_STORAGE_OPTS"),
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


def _default_max_workers() -> int:
    configured = os.environ.get("SWE_MAX_WORKERS")
    if configured:
        return int(configured)

    allocated_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated_cpus:
        return max(1, (int(allocated_cpus) * 3 + 3) // 4)

    return 4


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

    dataset = _load_swebench_dataset(
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


def _run_fast_harness(
    *,
    dataset_name: str,
    split: str,
    predictions_path: str,
    run_id: str,
    max_workers: int,
    timeout: int,
    instance_ids: list[str] | None,
    work_dir: Path,
    allow_x86_emulation: bool,
) -> Path:
    if predictions_path == "gold":
        raise ValueError("SWE_EVALUATOR=fast requires a predictions JSONL file; SWE_PREDICTIONS_PATH=gold is only supported by the official harness.")

    fast_bin = os.environ.get("SWE_BENCH_FAST_BIN", "swe-bench-fast")
    if shutil.which(fast_bin) is None and not Path(fast_bin).exists():
        raise FileNotFoundError(
            f"SWE_EVALUATOR=fast requires swe-bench-fast on PATH or SWE_BENCH_FAST_BIN, got: {fast_bin}"
        )

    dataset_path, predictions_path, evaluated_ids = _prepare_fast_inputs(
        dataset_name=dataset_name,
        split=split,
        instance_ids=instance_ids,
        predictions_path=predictions_path,
        work_dir=work_dir,
        allow_x86_emulation=allow_x86_emulation,
    )

    config_path = work_dir / "swe-bench-fast.toml"
    _write_fast_config(config_path, max_workers, timeout)

    report_path = work_dir / f"swe-bench-fast.{run_id}.json"
    if not evaluated_ids:
        _write_empty_fast_report(report_path)
        return report_path

    log_path = work_dir / f"swe-bench-fast.{run_id}.log"
    command = [
        fast_bin,
        "run",
        "--dataset",
        str(dataset_path),
        "--predictions",
        predictions_path,
        "--workers",
        str(max_workers),
        "--timeout",
        str(timeout),
        "--run-id",
        run_id,
        "--format",
        "json",
        "--output",
        str(report_path),
    ]
    print("Running swe-bench-fast: " + " ".join(command), flush=True)
    print(f"swe-bench-fast log: {log_path}", flush=True)
    with log_path.open("w") as log_handle:
        log_handle.write("Running swe-bench-fast: " + " ".join(command) + "\n")
        log_handle.flush()
        completed = subprocess.run(
            command,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"swe-bench-fast failed with exit code {completed.returncode}; see {log_path}")
    return report_path


def _prepare_fast_inputs(
    *,
    dataset_name: str,
    split: str,
    instance_ids: list[str] | None,
    predictions_path: str,
    work_dir: Path,
    allow_x86_emulation: bool,
) -> tuple[Path, str, set[str]]:
    configured_dataset_path = os.environ.get("SWE_BENCH_FAST_DATASET_PATH")
    source_dataset_name = str(Path(configured_dataset_path).expanduser().resolve()) if configured_dataset_path else dataset_name
    dataset = _load_swebench_dataset(source_dataset_name, split, instance_ids)
    predictions = _load_prediction_rows(predictions_path)

    unsupported_ids = _unsupported_fast_ids_on_this_host(dataset, allow_x86_emulation)
    evaluated_dataset = [instance for instance in dataset if instance["instance_id"] not in unsupported_ids]
    evaluated_ids = {instance["instance_id"] for instance in evaluated_dataset}
    evaluated_predictions = [row for row in predictions if row["instance_id"] in evaluated_ids]

    _write_unsupported_fast_artifacts(work_dir, unsupported_ids)

    dataset_path = work_dir / "dataset.fast.jsonl"
    with dataset_path.open("w") as handle:
        for instance in evaluated_dataset:
            handle.write(json.dumps(dict(instance), sort_keys=True) + "\n")

    filtered_predictions_path = work_dir / "predictions.fast.jsonl"
    with filtered_predictions_path.open("w") as handle:
        for prediction in evaluated_predictions:
            handle.write(json.dumps(prediction, sort_keys=True) + "\n")

    if unsupported_ids:
        print(
            "SWE-bench-fast ARM filter: "
            f"skipping {len(unsupported_ids)} x86-only instances; "
            f"evaluating {len(evaluated_dataset)} instances."
        )
    return dataset_path, str(filtered_predictions_path), evaluated_ids


def _load_prediction_rows(predictions_path: str) -> list[dict[str, Any]]:
    path = Path(predictions_path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        return list(payload.values()) if isinstance(payload, dict) else payload
    raise ValueError(f"Predictions path must be .json or .jsonl, got: {predictions_path}")


def _unsupported_fast_ids_on_this_host(
    dataset: list[dict[str, Any]],
    allow_x86_emulation: bool,
) -> set[str]:
    arch = os.environ.get("SWE_BENCH_FAST_ARCH") or os.environ.get("SWE_ARCH", "x86_64")
    if allow_x86_emulation or arch not in {"arm64", "aarch64"}:
        return set()

    supported_ids = _load_fast_arm64_supported_ids()
    if supported_ids is None:
        return set()

    return {
        instance["instance_id"]
        for instance in dataset
        if instance["instance_id"] not in supported_ids
    }


def _load_fast_arm64_supported_ids() -> set[str] | None:
    explicit = os.environ.get("SWE_BENCH_FAST_ARM64_DATASET")
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())

    source_dir = os.environ.get("SWE_BENCH_FAST_SOURCE_DIR")
    if source_dir:
        candidates.append(Path(source_dir).expanduser() / "swe-bench-arm64.jsonl")

    fast_bin = os.environ.get("SWE_BENCH_FAST_BIN")
    if fast_bin:
        bin_path = Path(fast_bin).expanduser()
        candidates.extend([
            bin_path.parent / "swe-bench-arm64.jsonl",
            bin_path.parent.parent / "swe-bench-arm64.jsonl",
        ])

    candidates.extend([
        Path("swe-bench-fast-main/swe-bench-arm64.jsonl"),
        Path("swe-bench-fast/swe-bench-arm64.jsonl"),
        Path("external/swe-bench-fast/swe-bench-arm64.jsonl"),
    ])

    for candidate in candidates:
        if candidate.exists():
            return {
                json.loads(line)["instance_id"]
                for line in candidate.read_text().splitlines()
                if line.strip()
            }

    print(
        "WARNING: could not find swe-bench-fast ARM64 support list; "
        "x86-only filtering is disabled."
    )
    return None


def _write_unsupported_fast_artifacts(work_dir: Path, unsupported_ids: set[str]) -> None:
    txt_path = work_dir / "x86_unsupported_ids.txt"
    json_path = work_dir / "x86_unsupported.json"
    sorted_ids = sorted(unsupported_ids)
    txt_path.write_text("\n".join(sorted_ids) + ("\n" if sorted_ids else ""))
    json_path.write_text(
        json.dumps(
            {
                "count": len(sorted_ids),
                "ids": sorted_ids,
                "reason": "x86-only SWE-bench-fast image on ARM host without x86 emulation",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _write_empty_fast_report(report_path: Path) -> None:
    report_path.write_text(
        json.dumps(
            {
                "reports": [],
                "summary": {
                    "total": 0,
                    "resolved": 0,
                    "partial": 0,
                    "unresolved": 0,
                    "errors": 0,
                    "resolved_pct": 0,
                    "total_time_ms": 0,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _load_swebench_dataset(
    name: str,
    split: str,
    instance_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if name.lower() in {"swe-bench", "swebench", "swe_bench"}:
        name = "SWE-bench/SWE-bench"
    elif name.lower() in {
        "swe-bench-lite",
        "swebench-lite",
        "swe_bench_lite",
        "swe-bench_lite",
        "lite",
    }:
        name = "SWE-bench/SWE-bench_Lite"

    if name.endswith(".json"):
        raw_dataset = json.loads(Path(name).read_text())
        if isinstance(raw_dataset, dict):
            raw_dataset = list(raw_dataset.values())
    elif name.endswith(".jsonl"):
        raw_dataset = [json.loads(line) for line in Path(name).read_text().splitlines() if line.strip()]
    elif name.endswith(".parquet"):
        from datasets import load_dataset

        raw_dataset = load_dataset("parquet", data_files=name, split="train")
    else:
        from datasets import load_dataset, load_from_disk

        parquet_path = Path(name) / f"{split}.parquet"
        disk_path = Path(name) / split
        if parquet_path.exists():
            raw_dataset = load_dataset("parquet", data_files=str(parquet_path), split="train")
        elif (disk_path / "dataset_info.json").exists():
            raw_dataset = load_from_disk(disk_path)
        else:
            raw_dataset = load_dataset(name, split=split)

    dataset = [dict(instance) for instance in raw_dataset]
    if not instance_ids:
        return dataset

    selected_ids = set(instance_ids)
    dataset_ids = {instance["instance_id"] for instance in dataset}
    missing = selected_ids - dataset_ids
    if missing:
        raise ValueError(
            "Some SWE-bench instance IDs were not found in the dataset: "
            + " ".join(sorted(missing))
        )
    return [instance for instance in dataset if instance["instance_id"] in selected_ids]


def _write_fast_config(config_path: Path, max_workers: int, timeout: int) -> None:
    arch = os.environ.get("SWE_BENCH_FAST_ARCH") or os.environ.get("SWE_ARCH", "x86_64")
    if arch == "arm64":
        arch = "aarch64"
    config = {
        "name": "swe-bench-fast",
        "workers": int(os.environ.get("SWE_BENCH_FAST_WORKERS", str(max_workers))),
        "timeout": int(os.environ.get("SWE_BENCH_FAST_TIMEOUT", str(timeout))),
        "arch": arch,
        "checkpoint_dir": os.environ.get("SWE_BENCH_FAST_CHECKPOINT_DIR", ".checkpoints"),
        "arm64_registry": os.environ.get("SWE_BENCH_FAST_ARM64_REGISTRY", "docker.io/greynewell/swe-bench-arm64"),
        "x86_registry": os.environ.get("SWE_BENCH_FAST_X86_REGISTRY", "ghcr.io/epoch-research"),
        "x86_prefix": os.environ.get("SWE_BENCH_FAST_X86_PREFIX", "swe-bench.eval"),
        "mem_limit": os.environ.get("SWE_BENCH_FAST_MEM_LIMIT", "4g"),
        "tmpfs": _env_bool("SWE_BENCH_FAST_TMPFS", False),
        "runtime": os.environ.get("SWE_BENCH_FAST_RUNTIME", ""),
        "build_workers": int(os.environ.get("SWE_BENCH_FAST_BUILD_WORKERS", "4")),
    }
    lines = []
    for key, value in config.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, int):
            rendered = str(value)
        else:
            rendered = json.dumps(value)
        lines.append(f"{key} = {rendered}")
    config_path.write_text("\n".join(lines) + "\n")


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


def _load_fast_report(
    report_path: Path,
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = json.loads(report_path.read_text())
    reports = payload.get("reports", [])
    by_id = {row.get("instance_id"): row for row in reports if row.get("instance_id")}
    unsupported_ids = _load_unsupported_ids(report_path.parent)
    requested_ids = sorted(predictions)
    submitted_ids = sorted(by_id)
    empty_patch_ids = {
        instance_id
        for instance_id, prediction in predictions.items()
        if instance_id in by_id
        if not (prediction.get("model_patch") or "").strip()
    }
    error_ids = {
        instance_id
        for instance_id, row in by_id.items()
        if row.get("error")
    }
    resolved_ids = {
        instance_id
        for instance_id, row in by_id.items()
        if row.get("resolved") == "RESOLVED_FULL" and not row.get("error")
    }
    completed_ids = {
        instance_id
        for instance_id, row in by_id.items()
        if not row.get("error") and instance_id not in empty_patch_ids
    }
    unresolved_ids = set(submitted_ids) - resolved_ids - error_ids - empty_patch_ids
    return {
        "requested_instances": len(requested_ids),
        "submitted_instances": len(submitted_ids),
        "completed_instances": len(completed_ids),
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "empty_patch_instances": len(empty_patch_ids),
        "error_instances": len(error_ids),
        "unsupported_x86_instances": len(unsupported_ids),
        "requested_ids": requested_ids,
        "submitted_ids": submitted_ids,
        "completed_ids": sorted(completed_ids),
        "resolved_ids": sorted(resolved_ids),
        "unresolved_ids": sorted(unresolved_ids),
        "empty_patch_ids": sorted(empty_patch_ids),
        "error_ids": sorted(error_ids),
        "unsupported_x86_ids": unsupported_ids,
        "fast_report_path": str(report_path),
        "fast_summary": payload.get("summary", {}),
        "fast_errors": {
            instance_id: row.get("error", "")
            for instance_id, row in by_id.items()
            if row.get("error")
        },
    }


def _load_unsupported_ids(work_dir: Path) -> list[str]:
    path = work_dir / "x86_unsupported.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return list(payload.get("ids", []))


def _print_prediction_summary(
    predictions_path: str,
    predictions: dict[str, dict[str, Any]],
) -> None:
    if predictions_path == "gold":
        print("SWE-bench prediction summary: using gold predictions.")
        return

    total = len(predictions)
    empty = 0
    diff_like = 0
    nonempty_non_diff: list[tuple[str, str]] = []
    for instance_id, prediction in predictions.items():
        patch = prediction.get("model_patch") or ""
        stripped = patch.strip()
        if not stripped:
            empty += 1
            continue
        if "diff --git " in stripped or stripped.startswith("--- "):
            diff_like += 1
        elif len(nonempty_non_diff) < 3:
            nonempty_non_diff.append((instance_id, " ".join(stripped.split())[:180]))

    print(
        "SWE-bench prediction summary: "
        f"{total} submitted, {empty} empty, {diff_like} diff-like, "
        f"{total - empty - diff_like} non-empty/non-diff."
    )
    if nonempty_non_diff:
        print("SWE-bench non-diff prediction samples:")
        for instance_id, preview in nonempty_non_diff:
            print(f"  {instance_id}: {preview}")


def _print_report_error_summary(
    work_dir: Path,
    report: dict[str, Any],
    predictions: dict[str, dict[str, Any]],
    run_id: str,
) -> None:
    submitted = int(report.get("submitted_instances", 0))
    completed = int(report.get("completed_instances", 0))
    unsupported = int(report.get("unsupported_x86_instances", 0))
    errors = list(report.get("error_ids", []))
    empty = list(report.get("empty_patch_ids", []))
    print(
        "SWE-bench report summary: "
        f"{completed}/{submitted} completed, {len(errors)} errors, "
        f"{len(empty)} empty patches, {unsupported} unsupported x86-only skipped."
    )
    if not errors:
        return

    print("SWE-bench error samples:")
    fast_errors = report.get("fast_errors", {})
    for instance_id in errors[:5]:
        if instance_id in fast_errors:
            print(f"  {instance_id}: {fast_errors[instance_id][:240]}")
            continue
        prediction = predictions.get(instance_id, {})
        model_name = prediction.get("model_name_or_path", "None").replace("/", "__")
        log_file = work_dir / "logs" / "run_evaluation" / run_id / model_name / instance_id / "run_instance.log"
        reason = _extract_error_reason(log_file)
        print(f"  {instance_id}: {reason}")


def _extract_error_reason(log_file: Path) -> str:
    if not log_file.exists():
        return f"missing run log: {log_file}"

    lines = log_file.read_text(errors="replace").splitlines()
    markers = (
        "Failed to apply patch",
        "APPLY_PATCH_FAIL",
        "EvaluationError",
        "BuildImageError",
        "Test timed out",
        "Timeout error",
        "Error in evaluating model",
        "docker.errors",
    )
    matches = [line.strip() for line in lines if any(marker in line for marker in markers)]
    if matches:
        return matches[-1][:240]
    for line in reversed(lines):
        stripped = line.strip()
        if stripped:
            return stripped[:240]
    return "run log is empty"


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
    requested = int(report.get("requested_instances", submitted))
    completed = int(report.get("completed_instances", 0))
    resolved = int(report.get("resolved_instances", 0))
    errors = int(report.get("error_instances", 0))
    empty = int(report.get("empty_patch_instances", 0))
    unsupported = int(report.get("unsupported_x86_instances", 0))
    denominator = submitted or 1
    return {
        "resolved": resolved / denominator,
        "completed": completed / denominator,
        "error_rate": errors / denominator,
        "empty_patch_rate": empty / denominator,
        "unsupported_x86_on_arm": float(unsupported),
        "num_instances": float(submitted),
        "num_requested_instances": float(requested),
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
