"""Offline export from lm-evaluation-harness results to EEE and HF previews.

This module deliberately contains no publishing functionality. The generated
files are intended for review before they are copied or submitted elsewhere.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING_FILE = REPO_ROOT / "configs/eval_export/task_mappings.json"
SELF_CONSISTENCY_SUFFIX = "_self_consistency"

SKIP_RESULT_KEYS = {"alias", "samples", "name", "sample_len", "sample_count"}
KNOWN_METRICS: dict[str, dict[str, Any]] = {
    "acc": {
        "metric_id": "accuracy",
        "metric_name": "Accuracy",
        "metric_kind": "accuracy",
        "metric_unit": "proportion",
        "min_score": 0.0,
        "max_score": 1.0,
    },
    "accuracy": {
        "metric_id": "accuracy",
        "metric_name": "Accuracy",
        "metric_kind": "accuracy",
        "metric_unit": "proportion",
        "min_score": 0.0,
        "max_score": 1.0,
    },
    "acc_norm": {
        "metric_id": "normalized_accuracy",
        "metric_name": "Normalized accuracy",
        "metric_kind": "accuracy",
        "metric_unit": "proportion",
        "min_score": 0.0,
        "max_score": 1.0,
    },
    "exact_match": {
        "metric_id": "exact_match",
        "metric_name": "Exact match",
        "metric_kind": "accuracy",
        "metric_unit": "proportion",
        "min_score": 0.0,
        "max_score": 1.0,
    },
    "f1": {
        "metric_id": "f1",
        "metric_name": "F1",
        "metric_kind": "f1",
        "metric_unit": "proportion",
        "min_score": 0.0,
        "max_score": 1.0,
    },
    "em": {
        "metric_id": "exact_match",
        "metric_name": "Exact match",
        "metric_kind": "accuracy",
        "metric_unit": "proportion",
        "min_score": 0.0,
        "max_score": 1.0,
    },
    "mcc": {
        "metric_id": "matthews_correlation",
        "metric_name": "Matthews correlation coefficient",
        "metric_kind": "correlation",
        "metric_unit": "coefficient",
        "min_score": -1.0,
        "max_score": 1.0,
    },
    "pass@1": {
        "metric_id": "pass_at_1",
        "metric_name": "pass@1",
        "metric_kind": "pass_rate",
        "metric_unit": "proportion",
        "min_score": 0.0,
        "max_score": 1.0,
        "metric_parameters": {"k": 1},
    },
    "pass_at_1": {
        "metric_id": "pass_at_1",
        "metric_name": "pass@1",
        "metric_kind": "pass_rate",
        "metric_unit": "proportion",
        "min_score": 0.0,
        "max_score": 1.0,
        "metric_parameters": {"k": 1},
    },
}


class ExportError(RuntimeError):
    """An input or export invariant was violated."""


def _format_items(items: Iterable[str], limit: int = 20) -> str:
    values = sorted(set(items))
    rendered = ", ".join(values[:limit])
    if len(values) > limit:
        rendered += f", ... ({len(values) - limit} more)"
    return rendered or "(none)"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"Could not read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExportError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=False)
        handle.write("\n")


def _find_results_file(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise ExportError(f"Results path does not exist: {path}")
    matches = sorted(path.glob("**/results_*.json"))
    if len(matches) != 1:
        raise ExportError(
            f"Expected exactly one results_*.json below {path}, found "
            f"{len(matches)}. Point the exporter at a completed single run or "
            "the merged split-results directory."
        )
    return matches[0]


def _load_mapping(path: Path) -> dict[str, Any]:
    mapping = _read_json(path)
    if mapping.get("mapping_version") != 1:
        raise ExportError(f"Unsupported mapping_version in {path}")
    tasks = mapping.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ExportError(f"Mapping file has no tasks: {path}")
    for task_name, task_mapping in tasks.items():
        eee = task_mapping.get("eee", {})
        if not eee.get("benchmark") or not eee.get("evaluation_name"):
            raise ExportError(f"Incomplete EEE mapping for {task_name}")
    return mapping


def _resolve_task_mapping(
    mapping: dict[str, Any], task_name: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve exact tasks and the repository's self-consistency task variants."""
    task_mapping = mapping["tasks"].get(task_name)
    if task_mapping is not None:
        return task_mapping, task_name
    if task_name.endswith(SELF_CONSISTENCY_SUFFIX):
        base_name = task_name.removesuffix(SELF_CONSISTENCY_SUFFIX)
        task_mapping = mapping["tasks"].get(base_name)
        if task_mapping is not None:
            return task_mapping, base_name
    return None, None


def _metric_candidates(
    raw: dict[str, Any],
    task_name: str,
    target_mapping: dict[str, Any],
) -> list[str] | None:
    """Return task-variant candidates, expanding the logged repeat count."""
    candidate_key = (
        "self_consistency_metric_candidates"
        if task_name.endswith(SELF_CONSISTENCY_SUFFIX)
        else "metric_candidates"
    )
    candidates = target_mapping.get(candidate_key)
    if candidates is None and candidate_key != "metric_candidates":
        candidates = target_mapping.get("metric_candidates")
    if not candidates:
        return None

    repeats = _task_config(raw, task_name).get("repeats")
    expanded: list[str] = []
    for candidate in candidates:
        if "{repeats}" not in candidate:
            expanded.append(candidate)
        elif isinstance(repeats, int) and repeats > 0:
            expanded.append(candidate.replace("{repeats}", str(repeats)))
        else:
            # Preserve the unresolved template so metric matching fails with
            # the task's available/configured metric diagnostic instead of
            # silently exporting every numeric metric.
            expanded.append(candidate)
    return expanded


def _parse_model_args(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(key): str(item) for key, item in value.items()}
    if not isinstance(value, str):
        return {}
    parsed: dict[str, str] = {}
    for part in value.split(","):
        if "=" in part:
            key, item = part.split("=", 1)
            parsed[key.strip()] = item.strip()
        elif parsed:
            last_key = next(reversed(parsed))
            parsed[last_key] += f",{part}"
    return parsed


def _parse_generation_kwargs(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    parsed: dict[str, Any] = {}
    for key, item in _parse_model_args(value).items():
        try:
            parsed[key] = ast.literal_eval(item)
        except (SyntaxError, ValueError):
            parsed[key] = item
    return parsed


def _looks_like_hf_id(value: str) -> bool:
    return bool(re.fullmatch(r"[^/\s]+/[^/\s]+", value)) and not value.startswith(
        (".", "/")
    )


def _model_id(raw: dict[str, Any], override: str | None) -> str:
    if override:
        if not _looks_like_hf_id(override):
            raise ExportError(f"--model-id must be owner/model, got {override!r}")
        return override
    config = raw.get("config", {})
    model_args = _parse_model_args(config.get("model_args"))
    candidates = [
        raw.get("model_name"),
        model_args.get("pretrained"),
        model_args.get("load"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and _looks_like_hf_id(candidate):
            return candidate
    raise ExportError(
        "Could not infer a Hugging Face model ID from the lm-eval log. "
        "Pass --model-id owner/model (required for local checkpoints)."
    )


def _epoch_string(value: Any, fallback: float) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = fallback
    if not math.isfinite(number):
        number = fallback
    return str(number)


def _iso_date(epoch: str) -> str:
    value = dt.datetime.fromtimestamp(float(epoch), tz=dt.timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def _stable_uuid(source_sha: str, benchmark: str, model_id: str) -> str:
    digest = hashlib.sha256(f"{source_sha}\0{benchmark}\0{model_id}".encode()).digest()[
        :16
    ]
    return str(uuid.UUID(bytes=digest, version=4))


def _metric_parts(key: str) -> tuple[str, str]:
    if "," not in key:
        return key, "none"
    metric, filter_name = key.split(",", 1)
    return metric.strip(), filter_name.strip()


def _numeric_metrics(results: dict[str, Any]) -> Iterable[tuple[str, float]]:
    for key, value in results.items():
        if key in SKIP_RESULT_KEYS or "_stderr" in key:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(float(value)):
            yield key, float(value)


def _task_config(raw: dict[str, Any], task_name: str) -> dict[str, Any]:
    value = raw.get("configs", {}).get(task_name, {})
    return value if isinstance(value, dict) else {}


def _generation_config(
    raw: dict[str, Any],
    task_name: str,
    filter_name: str,
    warnings: list[str],
) -> dict[str, Any] | None:
    task_config = _task_config(raw, task_name)
    generation_kwargs: dict[str, Any] = {}
    for candidate in (
        raw.get("config", {}).get("gen_kwargs"),
        task_config.get("generation_kwargs"),
    ):
        generation_kwargs.update(_parse_generation_kwargs(candidate))

    args: dict[str, Any] = {}
    temperature = generation_kwargs.get("temperature")
    if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
        args["temperature"] = float(temperature)
    elif generation_kwargs.get("do_sample") is False:
        args["temperature"] = 0.0

    for source_key, target_key in (("top_p", "top_p"), ("top_k", "top_k")):
        value = generation_kwargs.get(source_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            args[target_key] = value

    max_tokens = next(
        (
            generation_kwargs[key]
            for key in ("max_gen_toks", "max_new_tokens", "max_tokens")
            if key in generation_kwargs
        ),
        None,
    )
    if isinstance(max_tokens, int) and max_tokens > 0:
        args["max_tokens"] = max_tokens

    prompt_template = task_config.get("doc_to_text")
    if isinstance(prompt_template, str) and prompt_template:
        args["prompt_template"] = prompt_template

    output_type = str(task_config.get("output_type", "unknown"))
    if output_type in {"generate_until", "generate"}:
        if "temperature" not in args:
            warnings.append(f"{task_name}: generation temperature is not logged")
        if "max_tokens" not in args:
            warnings.append(f"{task_name}: maximum generation tokens are not logged")

    details: dict[str, str] = {
        "lm_eval_task": task_name,
        "metric_filter": filter_name,
        "output_type": output_type,
    }
    captured_generation_keys = {
        "temperature",
        "top_p",
        "top_k",
        "max_gen_toks",
        "max_new_tokens",
        "max_tokens",
    }
    for key, value in generation_kwargs.items():
        if key not in captured_generation_keys:
            details[f"generation_{key}"] = (
                value if isinstance(value, str) else json.dumps(value)
            )
    num_fewshot = task_config.get("num_fewshot", raw.get("n-shot", {}).get(task_name))
    if num_fewshot is not None:
        details["num_fewshot"] = str(num_fewshot)
    repeats = task_config.get("repeats")
    if isinstance(repeats, int) and repeats > 0:
        details["repeats"] = str(repeats)
    if task_name.endswith(SELF_CONSISTENCY_SUFFIX):
        details["evaluation_variant"] = "self_consistency"
    limit = raw.get("config", {}).get("limit")
    if limit is not None:
        details["limit"] = str(limit)
    for key in ("fewshot_as_multiturn", "chat_template_sha", "system_instruction_sha"):
        value = raw.get(key)
        if value is not None:
            details[key] = str(value)

    if not args and not details:
        return None
    return {"generation_args": args, "additional_details": details}


def _source_data(
    raw: dict[str, Any], task_name: str, task_mapping: dict[str, Any]
) -> dict[str, Any]:
    eee = task_mapping["eee"]
    config = _task_config(raw, task_name)
    dataset_id = eee.get("dataset_id") or config.get("dataset_path")
    dataset_name = str(eee["evaluation_name"])
    samples = raw.get("n-samples", {}).get(task_name, {})
    count = samples.get("effective") if isinstance(samples, dict) else None

    if isinstance(dataset_id, str) and _looks_like_hf_id(dataset_id):
        value: dict[str, Any] = {
            "dataset_name": dataset_name,
            "source_type": "hf_dataset",
            "hf_repo": dataset_id,
        }
        split = config.get("test_split") or config.get("validation_split")
        if split:
            value["hf_split"] = str(split)
        if isinstance(count, int):
            value["samples_number"] = count
        dataset_config = config.get("dataset_name")
        if dataset_config:
            value["additional_details"] = {"hf_dataset_config": str(dataset_config)}
        return value
    return {
        "dataset_name": dataset_name,
        "source_type": "other",
        "additional_details": {"lm_eval_dataset_path": str(dataset_id or "unknown")},
    }


def _metric_config(
    raw: dict[str, Any],
    task_name: str,
    metric: str,
    filter_name: str,
    task_mapping: dict[str, Any],
) -> dict[str, Any]:
    higher = raw.get("higher_is_better", {}).get(task_name, {}).get(metric, True)
    value: dict[str, Any] = {
        "evaluation_description": (
            metric if filter_name == "none" else f"{metric} (filter: {filter_name})"
        ),
        "lower_is_better": not bool(higher),
    }
    known = KNOWN_METRICS.get(metric)
    pass_at_match = re.fullmatch(r"pass(?:@|_at_)(\d+)", metric)
    if known is None and pass_at_match:
        k = int(pass_at_match.group(1))
        known = {
            "metric_id": f"pass_at_{k}",
            "metric_name": f"pass@{k}",
            "metric_kind": "pass_rate",
            "metric_unit": "proportion",
            "min_score": 0.0,
            "max_score": 1.0,
            "metric_parameters": {"k": k},
        }
    if known:
        value.update(known)
        value["score_type"] = "continuous"
    else:
        value["metric_id"] = metric
        value["metric_name"] = metric
    override = task_mapping["eee"].get("metric_overrides", {}).get(metric)
    if override:
        value.update(override)
        value["score_type"] = "continuous"
    return value


def _uncertainty(
    raw: dict[str, Any],
    task_name: str,
    metric: str,
    filter_name: str,
    task_results: dict[str, Any],
) -> dict[str, Any] | None:
    value: dict[str, Any] = {}
    stderr = task_results.get(f"{metric}_stderr,{filter_name}")
    if isinstance(stderr, (int, float)) and not isinstance(stderr, bool):
        value["standard_error"] = {
            "value": float(stderr),
            "method": "bootstrap",
        }
    samples = raw.get("n-samples", {}).get(task_name, {})
    count = samples.get("effective") if isinstance(samples, dict) else None
    if isinstance(count, int):
        value["num_samples"] = count
    return value or None


def _evaluation_result(
    raw: dict[str, Any],
    task_name: str,
    result_key: str,
    score: float,
    task_mapping: dict[str, Any],
    evaluation_timestamp: str,
    warnings: list[str],
) -> dict[str, Any]:
    metric, filter_name = _metric_parts(result_key)
    canonical_name = task_mapping["eee"]["evaluation_name"]
    evaluation_name = (
        canonical_name if filter_name == "none" else f"{canonical_name}/{filter_name}"
    )
    result_id = hashlib.sha256(
        f"{task_name}\0{result_key}\0{score}".encode()
    ).hexdigest()
    value: dict[str, Any] = {
        "evaluation_result_id": result_id,
        "evaluation_name": evaluation_name,
        "source_data": _source_data(raw, task_name, task_mapping),
        "evaluation_timestamp": evaluation_timestamp,
        "metric_config": _metric_config(
            raw, task_name, metric, filter_name, task_mapping
        ),
        "score_details": {"score": score},
    }
    uncertainty = _uncertainty(
        raw, task_name, metric, filter_name, raw["results"][task_name]
    )
    if uncertainty:
        value["score_details"]["uncertainty"] = uncertainty
    generation = _generation_config(raw, task_name, filter_name, warnings)
    if generation:
        value["generation_config"] = generation
    return value


def _extract_prompt(sample: dict[str, Any]) -> str:
    arguments = sample.get("arguments", {})
    if not isinstance(arguments, dict) or not arguments:
        return str(sample.get("doc", sample.get("input", "")))
    first = arguments.get("gen_args_0", {})
    return str(first.get("arg_0", "")) if isinstance(first, dict) else ""


def _extract_choices(sample: dict[str, Any]) -> list[str] | None:
    arguments = sample.get("arguments", {})
    if not isinstance(arguments, dict) or "gen_args_1" not in arguments:
        return None
    choices: list[str] = []
    index = 0
    while f"gen_args_{index}" in arguments:
        argument = arguments[f"gen_args_{index}"]
        if isinstance(argument, dict) and "arg_1" in argument:
            choices.append(str(argument["arg_1"]).strip())
        index += 1
    return choices or None


def _extract_output(sample: dict[str, Any], choices: list[str] | None) -> str:
    source = sample.get("filtered_resps") or sample.get("resps") or []
    if choices:
        try:
            scores = []
            for response in source:
                item = response[0] if isinstance(response, list) else response
                item = item[0] if isinstance(item, list) else item
                scores.append(float(item))
            return choices[scores.index(max(scores))]
        except (TypeError, ValueError, IndexError):
            return str(source)
    first = source[0] if source else ""
    if isinstance(first, list):
        first = first[0] if first else ""
    return str(first)


def _sample_score(sample: dict[str, Any]) -> tuple[float, bool]:
    for metric in sample.get("metrics", []):
        value = sample.get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            score = float(value)
            return score, score == 1.0
    return 0.0, False


def _convert_sample(
    sample: dict[str, Any],
    evaluation_id: str,
    model_id: str,
    canonical_name: str,
) -> dict[str, Any]:
    prompt = _extract_prompt(sample)
    target = str(sample.get("target", ""))
    choices = _extract_choices(sample)
    output = _extract_output(sample, choices)
    score, is_correct = _sample_score(sample)
    filter_name = str(sample.get("filter", "none"))
    evaluation_name = (
        canonical_name if filter_name == "none" else f"{canonical_name}/{filter_name}"
    )
    sample_hash = hashlib.sha256(
        json.dumps(
            {"raw": prompt, "reference": target},
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    return {
        "schema_version": "0.2.2",
        "evaluation_id": evaluation_id,
        "model_id": model_id,
        "evaluation_name": evaluation_name,
        "sample_id": str(sample.get("doc_id", 0)),
        "sample_hash": sample_hash,
        "interaction_type": "single_turn",
        "input": {
            "raw": prompt,
            "reference": [target],
            "choices": choices,
        },
        "output": {"raw": [output]},
        "answer_attribution": [
            {
                "turn_idx": 0,
                "source": "output.raw",
                "extracted_value": output,
                "extraction_method": ("none" if filter_name == "none" else filter_name),
                "is_terminal": True,
            }
        ],
        "evaluation": {"score": score, "is_correct": is_correct},
        "metadata": {
            "doc_hash": str(sample.get("doc_hash", "")),
            "prompt_hash": str(sample.get("prompt_hash", "")),
            "target_hash": str(sample.get("target_hash", "")),
            "filter": filter_name,
        },
    }


def _sample_file(results_file: Path, task_name: str) -> Path | None:
    matches = sorted(results_file.parent.glob(f"**/samples_{task_name}_*.jsonl"))
    return matches[-1] if matches else None


def _hf_metric(
    task_results: dict[str, Any], candidates: list[str]
) -> tuple[str, float] | None:
    numeric = dict(_numeric_metrics(task_results))
    for candidate in candidates:
        if candidate in numeric:
            return candidate, numeric[candidate]
    return None


def _selected_metrics(
    raw: dict[str, Any],
    task_name: str,
    task_results: dict[str, Any],
    task_mapping: dict[str, Any],
) -> list[tuple[str, float]]:
    candidates = _metric_candidates(raw, task_name, task_mapping["eee"])
    if not candidates:
        return list(_numeric_metrics(task_results))
    selected = _hf_metric(task_results, candidates)
    return [selected] if selected else []


def _yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _hf_yaml(entry: dict[str, Any]) -> str:
    lines = [
        "- dataset:",
        f"    id: {_yaml_string(entry['dataset_id'])}",
        f"    task_id: {_yaml_string(entry['task_id'])}",
        f"  value: {entry['value']!r}",
        f"  date: {_yaml_string(entry['date'])}",
        "  source:",
        f"    url: {_yaml_string(entry['source_url'])}",
        '    name: "Every Eval Ever"',
        f"  notes: {_yaml_string(entry['notes'])}",
    ]
    return "\n".join(lines) + "\n"


def _notes(raw: dict[str, Any], task_name: str, metric_key: str) -> str:
    config = _task_config(raw, task_name)
    generation = _generation_config(raw, task_name, "none", [])
    args = generation.get("generation_args", {}) if generation else {}
    parts = [
        f"lm-eval {raw.get('lm_eval_version', 'unknown')}",
        f"task={task_name}",
        f"metric={metric_key}",
    ]
    if config.get("num_fewshot") is not None:
        parts.append(f"{config['num_fewshot']}-shot")
    if "temperature" in args:
        parts.append(f"temperature={args['temperature']}")
    if "max_tokens" in args:
        parts.append(f"max_tokens={args['max_tokens']}")
    if raw.get("fewshot_as_multiturn"):
        parts.append("fewshot_as_multiturn=true")
    if raw.get("chat_template"):
        parts.append("chat_template=true")
    return "; ".join(parts)


def _installed_package_version(raw: dict[str, Any], package: str) -> str | None:
    environment = raw.get("pretty_env_info")
    if not isinstance(environment, str):
        return None
    match = re.search(
        rf"^\[(?:pip3|conda)\]\s+{re.escape(package)}(?:==|\s+)([^\s]+)",
        environment,
        re.MULTILINE | re.IGNORECASE,
    )
    return match.group(1) if match else None


def _validate_record(record: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version",
        "evaluation_id",
        "retrieved_timestamp",
        "source_metadata",
        "model_info",
        "eval_library",
        "evaluation_results",
    }
    missing = sorted(required - record.keys())
    if missing:
        errors.append(f"{path}: missing top-level fields: {', '.join(missing)}")
    if record.get("schema_version") != "0.2.2":
        errors.append(f"{path}: expected EEE schema_version 0.2.2")
    model = record.get("model_info", {})
    if not model.get("name") or not model.get("id"):
        errors.append(f"{path}: model_info requires name and id")
    results = record.get("evaluation_results")
    if not isinstance(results, list) or not results:
        errors.append(f"{path}: evaluation_results must be non-empty")
    else:
        for index, result in enumerate(results):
            label = f"{path}: evaluation_results[{index}]"
            for key in (
                "evaluation_name",
                "source_data",
                "metric_config",
                "score_details",
            ):
                if key not in result:
                    errors.append(f"{label}: missing {key}")
            score = result.get("score_details", {}).get("score")
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                errors.append(f"{label}: score must be numeric")
            if "lower_is_better" not in result.get("metric_config", {}):
                errors.append(f"{label}: metric_config.lower_is_better is required")
    return errors


def _validate_instance(record: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version",
        "evaluation_id",
        "model_id",
        "evaluation_name",
        "sample_id",
        "interaction_type",
        "input",
        "answer_attribution",
        "evaluation",
    }
    missing = sorted(required - record.keys())
    if missing:
        errors.append(f"{path}: missing instance fields: {', '.join(missing)}")
    if record.get("schema_version") != "0.2.2":
        errors.append(f"{path}: expected instance schema_version 0.2.2")
    if record.get("interaction_type") != "single_turn":
        errors.append(f"{path}: lm-eval export must be single_turn")
    input_data = record.get("input", {})
    if not isinstance(input_data.get("raw"), str) or not isinstance(
        input_data.get("reference"), list
    ):
        errors.append(f"{path}: input requires raw text and a reference list")
    evaluation = record.get("evaluation", {})
    if not isinstance(evaluation.get("score"), (int, float)) or isinstance(
        evaluation.get("score"), bool
    ):
        errors.append(f"{path}: evaluation.score must be numeric")
    if not isinstance(evaluation.get("is_correct"), bool):
        errors.append(f"{path}: evaluation.is_correct must be boolean")
    return errors


def export_results(
    results_path: Path,
    output_dir: Path,
    mapping_file: Path = DEFAULT_MAPPING_FILE,
    *,
    model_id_override: str | None = None,
    source_organization_name: str = "Swiss AI Initiative",
    source_organization_url: str | None = "https://www.swiss-ai.org/",
    evaluator_relationship: str | None = None,
    include_samples: bool = False,
    strict_mappings: bool = False,
    retrieved_timestamp: str | None = None,
) -> dict[str, Any]:
    """Export one completed lm-eval run and return its manifest."""
    results_file = _find_results_file(results_path)
    raw = _read_json(results_file)
    if not isinstance(raw.get("results"), dict):
        raise ExportError(f"{results_file} has no lm-eval results object")
    mapping = _load_mapping(mapping_file)
    model_id = _model_id(raw, model_id_override)
    developer, model_name = model_id.split("/", 1)
    relationship = evaluator_relationship or (
        "first_party" if developer.lower() == "swiss-ai" else "third_party"
    )
    source_bytes = results_file.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    evaluation_timestamp = _epoch_string(raw.get("date"), results_file.stat().st_mtime)
    retrieved = _epoch_string(retrieved_timestamp, time.time())
    warnings: list[str] = []
    skipped: list[str] = []
    numeric_tasks: dict[str, list[str]] = {}
    metric_mismatches: dict[str, tuple[list[str], list[str]]] = {}
    resolved_task_aliases: dict[str, str] = {}
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)

    for task_name, task_results in raw["results"].items():
        if not isinstance(task_results, dict):
            continue
        available_metrics = [key for key, _ in _numeric_metrics(task_results)]
        if not available_metrics:
            continue
        numeric_tasks[task_name] = available_metrics
        task_mapping, mapped_task_name = _resolve_task_mapping(mapping, task_name)
        if not task_mapping:
            skipped.append(task_name)
            continue
        if mapped_task_name != task_name:
            resolved_task_aliases[task_name] = str(mapped_task_name)
        candidates = _metric_candidates(raw, task_name, task_mapping["eee"])
        if candidates and _hf_metric(task_results, candidates) is None:
            metric_mismatches[task_name] = (
                available_metrics,
                list(candidates),
            )
            continue
        grouped[task_mapping["eee"]["benchmark"]].append((task_name, task_mapping))

    if not numeric_tasks:
        raise ExportError(
            "The lm-eval results file contains no tasks with numeric scores.\n"
            f"Results file: {results_file}\n"
            "Check that the run completed successfully and that its `results` "
            "object contains scored leaf or aggregate tasks."
        )
    if strict_mappings and (skipped or metric_mismatches):
        details = []
        if skipped:
            details.append("Unmapped lm-eval tasks: " + _format_items(skipped))
        if metric_mismatches:
            details.append(
                "Mapped tasks without a configured metric match: "
                + _format_items(metric_mismatches)
            )
        raise ExportError("\n".join(details))
    if not grouped:
        lines = [
            "No exportable benchmark results were found.",
            f"Results file: {results_file}",
            f"Task mapping: {mapping_file}",
            (
                f"Detected {len(numeric_tasks)} task(s) with numeric results: "
                f"{_format_items(numeric_tasks)}"
            ),
        ]
        if skipped:
            lines.append(
                "None of these exact task names are mapped: " + _format_items(skipped)
            )
        for task_name, (available, candidates) in sorted(metric_mismatches.items()):
            lines.append(
                f"{task_name}: mapped, but available metrics "
                f"[{_format_items(available)}] do not match configured "
                f"candidates [{_format_items(candidates)}]"
            )
        lines.append(
            "Add reviewed entries for these exact lm-eval task names to the "
            "mapping file. Do not map variants such as AIME to an unrelated "
            "EEE collection merely to make the export pass."
        )
        raise ExportError("\n".join(lines))

    for task_name, (available, candidates) in sorted(metric_mismatches.items()):
        warnings.append(
            f"{task_name}: mapped, but available metrics "
            f"[{_format_items(available)}] do not match configured candidates "
            f"[{_format_items(candidates)}]"
        )

    records_manifest: list[dict[str, Any]] = []
    for benchmark, task_entries in sorted(grouped.items()):
        record_uuid = _stable_uuid(source_sha, benchmark, model_id)
        evaluation_id = f"{benchmark}/{model_id}/{retrieved}"
        record_results: list[dict[str, Any]] = []
        instance_rows: list[dict[str, Any]] = []
        hf_entries: list[dict[str, Any]] = []

        for task_name, task_mapping in task_entries:
            task_results = raw["results"][task_name]
            selected_metrics = _selected_metrics(
                raw, task_name, task_results, task_mapping
            )
            if not selected_metrics:
                warnings.append(
                    f"{task_name}: none of the configured EEE metric "
                    "candidates were present"
                )
            for result_key, score in selected_metrics:
                record_results.append(
                    _evaluation_result(
                        raw,
                        task_name,
                        result_key,
                        score,
                        task_mapping,
                        evaluation_timestamp,
                        warnings,
                    )
                )

            if include_samples:
                sample_path = _sample_file(results_file, task_name)
                if sample_path is None:
                    warnings.append(f"{task_name}: no sample JSONL file found")
                else:
                    with sample_path.open(encoding="utf-8") as handle:
                        for line_number, line in enumerate(handle, start=1):
                            if not line.strip():
                                continue
                            try:
                                sample = json.loads(line)
                            except json.JSONDecodeError as exc:
                                raise ExportError(
                                    f"{sample_path}:{line_number}: {exc}"
                                ) from exc
                            instance_rows.append(
                                _convert_sample(
                                    sample,
                                    evaluation_id,
                                    model_id,
                                    task_mapping["eee"]["evaluation_name"],
                                )
                            )

            hf = task_mapping.get("huggingface")
            if hf:
                hf_candidates = _metric_candidates(raw, task_name, hf) or []
                selected = _hf_metric(dict(selected_metrics), hf_candidates)
                if selected is None:
                    warnings.append(
                        f"{task_name}: none of the configured Hugging Face "
                        "metric candidates were present"
                    )
                else:
                    metric_key, value = selected
                    flat_url = (
                        "https://huggingface.co/datasets/"
                        f"{mapping['datastore']['repo_id']}/blob/"
                        f"{mapping['datastore']['revision']}/flat/objects/"
                        f"{record_uuid[:2]}/{record_uuid[2:4]}/{record_uuid}.json"
                    )
                    hf_entries.append(
                        {
                            "dataset_id": hf["dataset_id"],
                            "task_id": hf["task_id"],
                            "value": value,
                            "date": _iso_date(evaluation_timestamp),
                            "source_url": flat_url,
                            "notes": _notes(raw, task_name, metric_key),
                        }
                    )

        raw_config = raw.get("config", {})
        library_details = {
            "git_hash": str(raw.get("git_hash") or "unknown"),
        }
        for key in (
            "batch_size",
            "device",
            "limit",
            "bootstrap_iters",
            "random_seed",
            "numpy_seed",
            "torch_seed",
            "fewshot_seed",
        ):
            value = raw_config.get(key)
            if value is not None:
                library_details[key] = str(value)
        task_hashes = raw.get("task_hashes")
        if isinstance(task_hashes, dict) and task_hashes:
            library_details["task_hashes"] = json.dumps(task_hashes, sort_keys=True)

        record: dict[str, Any] = {
            "schema_version": mapping["eee_schema_version"],
            "evaluation_id": evaluation_id,
            "evaluation_timestamp": evaluation_timestamp,
            "retrieved_timestamp": retrieved,
            "source_metadata": {
                "source_name": "SwissAI lm-evaluation-harness run",
                "source_type": "evaluation_run",
                "source_organization_name": source_organization_name,
                "evaluator_relationship": relationship,
            },
            "eval_library": {
                "name": "lm_eval",
                "version": str(raw.get("lm_eval_version") or "unknown"),
                "additional_details": library_details,
            },
            "model_info": {
                "name": model_id,
                "id": model_id,
                "developer": developer,
            },
            "evaluation_results": record_results,
        }
        if source_organization_url:
            record["source_metadata"]["source_organization_url"] = (
                source_organization_url
            )

        engine = str(raw_config.get("model") or "")
        engine_name = {"hf": "transformers"}.get(engine, engine)
        if engine_name:
            engine_data: dict[str, str] = {"name": engine_name}
            if engine == "hf" and raw.get("transformers_version"):
                engine_data["version"] = str(raw["transformers_version"])
            elif package_version := _installed_package_version(raw, engine_name):
                engine_data["version"] = package_version
            record["model_info"]["inference_engine"] = engine_data
        model_details = {
            key: str(value)
            for key, value in {
                "revision": raw_config.get("model_revision"),
                "sha": raw_config.get("model_sha"),
                "dtype": raw_config.get("model_dtype"),
                "num_parameters": raw_config.get("model_num_parameters"),
                "model_args": raw_config.get("model_args"),
            }.items()
            if value not in (None, "")
        }
        if model_details:
            record["model_info"]["additional_details"] = model_details

        record_dir = output_dir / "eee/data" / benchmark / developer / model_name
        record_path = record_dir / f"{record_uuid}.json"
        samples_path: Path | None = None
        if instance_rows:
            samples_path = record_dir / f"{record_uuid}_samples.jsonl"
            samples_path.parent.mkdir(parents=True, exist_ok=True)
            with samples_path.open("w", encoding="utf-8") as handle:
                for row in instance_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            record["detailed_evaluation_results"] = {
                "format": "jsonl",
                "file_path": samples_path.name,
                "hash_algorithm": "sha256",
                "checksum": hashlib.sha256(samples_path.read_bytes()).hexdigest(),
                "total_rows": len(instance_rows),
            }

        validation_errors = _validate_record(record, str(record_path))
        if validation_errors:
            raise ExportError("\n".join(validation_errors))
        _write_json(record_path, record)

        hf_path: Path | None = None
        if hf_entries:
            hf_path = (
                output_dir
                / "huggingface/.eval_results"
                / f"{re.sub(r'[^a-z0-9]+', '-', benchmark.lower()).strip('-')}.yaml"
            )
            hf_path.parent.mkdir(parents=True, exist_ok=True)
            with hf_path.open("w", encoding="utf-8") as handle:
                for entry in hf_entries:
                    handle.write(_hf_yaml(entry))

        records_manifest.append(
            {
                "benchmark": benchmark,
                "lm_eval_tasks": [name for name, _ in task_entries],
                "uuid": record_uuid,
                "eee_record": str(record_path.relative_to(output_dir)),
                "instance_results": (
                    str(samples_path.relative_to(output_dir)) if samples_path else None
                ),
                "huggingface_preview": (
                    str(hf_path.relative_to(output_dir)) if hf_path else None
                ),
            }
        )

    manifest = {
        "format_version": 1,
        "source_results": str(results_file),
        "source_sha256": source_sha,
        "model_id": model_id,
        "evaluation_timestamp": evaluation_timestamp,
        "retrieved_timestamp": retrieved,
        "mapping_file": str(mapping_file),
        "mapping_version": mapping["mapping_version"],
        "eee_schema_version": mapping["eee_schema_version"],
        "records": records_manifest,
        "skipped_unmapped_tasks": sorted(skipped),
        "resolved_task_aliases": dict(sorted(resolved_task_aliases.items())),
        "warnings": sorted(set(warnings)),
        "publishing_performed": False,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def validate_export(output_dir: Path) -> list[str]:
    """Validate the exporter manifest, aggregate records, and checksums."""
    errors: list[str] = []
    manifest_path = output_dir / "manifest.json"
    try:
        manifest = _read_json(manifest_path)
    except ExportError as exc:
        return [str(exc)]
    if manifest.get("publishing_performed") is not False:
        errors.append("manifest.json: publishing_performed must be false")
    for item in manifest.get("records", []):
        record_path = output_dir / item["eee_record"]
        try:
            record = _read_json(record_path)
        except ExportError as exc:
            errors.append(str(exc))
            continue
        errors.extend(_validate_record(record, str(record_path)))
        detailed = record.get("detailed_evaluation_results")
        if detailed:
            samples_path = record_path.parent / detailed["file_path"]
            if not samples_path.is_file():
                errors.append(f"{samples_path}: missing instance results")
            else:
                actual = hashlib.sha256(samples_path.read_bytes()).hexdigest()
                if actual != detailed.get("checksum"):
                    errors.append(f"{samples_path}: checksum mismatch")
                rows = 0
                with samples_path.open(encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        rows += 1
                        try:
                            instance = json.loads(line)
                        except json.JSONDecodeError as exc:
                            errors.append(f"{samples_path}:{line_number}: {exc}")
                            continue
                        errors.extend(
                            _validate_instance(
                                instance, f"{samples_path}:{line_number}"
                            )
                        )
                if rows != detailed.get("total_rows"):
                    errors.append(f"{samples_path}: row-count mismatch")
        hf_preview = item.get("huggingface_preview")
        if hf_preview and not (output_dir / hf_preview).is_file():
            errors.append(f"{output_dir / hf_preview}: missing HF preview")
    return errors


def check_remote_mappings(mapping_file: Path = DEFAULT_MAPPING_FILE) -> list[str]:
    """Check EEE collection names and registered HF benchmark task IDs."""
    mapping = _load_mapping(mapping_file)
    datastore = mapping["datastore"]
    path = urllib.parse.quote("data", safe="/")
    url = (
        "https://huggingface.co/api/datasets/"
        f"{datastore['repo_id']}/tree/{datastore['revision']}/{path}"
        "?recursive=false&expand=false&limit=1000"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            entries = json.load(response)
    except Exception as exc:
        return [f"Could not read EEE datastore tree: {exc}"]
    available = {
        entry["path"].removeprefix("data/")
        for entry in entries
        if entry.get("type") == "directory"
    }
    configured = {task["eee"]["benchmark"] for task in mapping["tasks"].values()}
    errors = [
        f"EEE collection is not present at data/{name}"
        for name in sorted(configured - available)
    ]
    hf_targets = {
        (task["huggingface"]["dataset_id"], task["huggingface"]["task_id"])
        for task in mapping["tasks"].values()
        if task.get("huggingface")
    }
    for dataset_id, task_id in sorted(hf_targets):
        eval_url = (
            "https://huggingface.co/datasets/"
            f"{urllib.parse.quote(dataset_id, safe='/')}/raw/main/eval.yaml"
        )
        try:
            with urllib.request.urlopen(eval_url, timeout=30) as response:
                eval_yaml = response.read().decode("utf-8")
        except Exception as exc:
            errors.append(
                f"Could not read eval.yaml for HF benchmark {dataset_id}: {exc}"
            )
            continue
        task_pattern = re.compile(
            rf"^\s*-\s+id:\s*[\"']?{re.escape(task_id)}[\"']?\s*(?:#.*)?$",
            re.MULTILINE,
        )
        if not task_pattern.search(eval_yaml):
            errors.append(
                f"HF benchmark {dataset_id} has no task id {task_id!r} in eval.yaml"
            )
    return errors


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export lm-eval results to reviewable EEE records and Hugging "
            "Face .eval_results YAML. This tool never publishes."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export", help="export a completed run")
    export.add_argument("results_path", type=Path)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--mapping-file", type=Path, default=DEFAULT_MAPPING_FILE)
    export.add_argument("--model-id")
    export.add_argument("--source-organization-name", default="Swiss AI Initiative")
    export.add_argument(
        "--source-organization-url", default="https://www.swiss-ai.org/"
    )
    export.add_argument(
        "--evaluator-relationship",
        choices=["auto", "first_party", "third_party", "collaborative", "other"],
        default="auto",
        help=(
            "Default: first_party for swiss-ai/* models and third_party "
            "for other model owners"
        ),
    )
    export.add_argument("--include-samples", action="store_true")
    export.add_argument("--strict-mappings", action="store_true")
    export.add_argument(
        "--retrieved-timestamp",
        help="Override record-creation epoch (mainly for reproducible rebuilds)",
    )

    validate = subparsers.add_parser(
        "validate", help="validate an existing export directory"
    )
    validate.add_argument("output_dir", type=Path)

    check = subparsers.add_parser(
        "check-mappings",
        help="read the current EEE datastore tree and check collection names",
    )
    check.add_argument("--mapping-file", type=Path, default=DEFAULT_MAPPING_FILE)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        if args.command == "export":
            manifest = export_results(
                args.results_path,
                args.output_dir,
                args.mapping_file,
                model_id_override=args.model_id,
                source_organization_name=args.source_organization_name,
                source_organization_url=args.source_organization_url,
                evaluator_relationship=(
                    None
                    if args.evaluator_relationship == "auto"
                    else args.evaluator_relationship
                ),
                include_samples=args.include_samples,
                strict_mappings=args.strict_mappings,
                retrieved_timestamp=args.retrieved_timestamp,
            )
            print(
                f"Exported {len(manifest['records'])} EEE record(s) to "
                f"{args.output_dir}"
            )
            print(
                f"Skipped {len(manifest['skipped_unmapped_tasks'])} "
                "unmapped task(s); "
                f"{len(manifest['warnings'])} warning(s)."
            )
            if manifest["skipped_unmapped_tasks"]:
                print(
                    "Unmapped tasks: "
                    + _format_items(manifest["skipped_unmapped_tasks"])
                )
            if manifest["resolved_task_aliases"]:
                print("Resolved self-consistency task aliases:")
                for source, target in manifest["resolved_task_aliases"].items():
                    print(f"  {source} -> {target}")
            for warning in manifest["warnings"]:
                print(f"Warning: {warning}")
            print("No files were published.")
        elif args.command == "validate":
            errors = validate_export(args.output_dir)
            if errors:
                print("\n".join(errors), file=sys.stderr)
                raise SystemExit(1)
            print(f"Export is internally valid: {args.output_dir}")
        else:
            errors = check_remote_mappings(args.mapping_file)
            if errors:
                print("\n".join(errors), file=sys.stderr)
                raise SystemExit(1)
            print(
                "All configured EEE collections and Hugging Face benchmark tasks exist."
            )
    except ExportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
