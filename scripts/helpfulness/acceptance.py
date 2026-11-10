"""Predeclared, artifact-based replay, regeneration and judge acceptance checks."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import re
import sys
import tomllib
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

from scripts.helpfulness import acceptance_scoring as scoring
from scripts.helpfulness import adapters, native
from scripts.helpfulness.core import (
    digest, endpoint_model, mean, number, read_json, read_jsonl, render_request, write_json,
)

KINDS = ("scorer_replay", "reference_regeneration", "judge_calibration")
PACKAGES = ("lm_eval", "alpaca_eval", "numpy", "scipy", "pandas", "scikit-learn", "nltk")


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha256(value) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected a lowercase SHA-256, not a placeholder or mutable revision")
    return value


def integer(value, label: str, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{label} must be an integer between {lower} and {upper}")
    return value


def nonempty(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.startswith("REPLACE_"):
        raise ValueError(f"{label} must be explicit, nonempty and not a placeholder")
    return value


def fingerprint() -> dict:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "source_sha256": digest({
            path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        }),
        "packages": versions,
        "python": ".".join(map(str, sys.version_info[:3])),
        "harness_revision": native.harness_pin(),
    }


def checked_artifact(config: dict, base: Path) -> dict:
    path = (base / nonempty(config.get("path"), "artifact path")).resolve()
    expected = sha256(config.get("sha256"))
    if file_hash(path) != expected:
        raise ValueError(f"Artifact checksum mismatch: {path}")
    return {"path": str(path), "sha256": expected}


def deployment(value: dict) -> dict:
    required = {"schema_version", "model_id", "weights", "tokenizer", "chat_template_sha256", "inference"}
    if not isinstance(value, dict) or set(value) != required or value["schema_version"] != 1:
        raise ValueError("Deployment assertion must have the exact documented schema")
    nonempty(value["model_id"], "served model ID")
    sha256(value["chat_template_sha256"])
    for kind in ("weights", "tokenizer"):
        pin = value[kind]
        if not isinstance(pin, dict) or set(pin) != {"source", "revision"}:
            raise ValueError(f"{kind} must identify its source and immutable revision")
        nonempty(pin["source"], f"{kind} source")
        if not isinstance(pin["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", pin["revision"]):
            raise ValueError(f"{kind} needs a commit/content hash, not a model name or branch")
    inference = value["inference"]
    if not isinstance(inference, dict) or set(inference) != {"backend", "revision", "dtype", "settings"}:
        raise ValueError("Inference must pin backend, revision, dtype and effective settings")
    nonempty(inference["backend"], "inference backend")
    nonempty(inference["dtype"], "inference dtype")
    if not isinstance(inference["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", inference["revision"]):
        raise ValueError("Inference needs an immutable code/image revision")
    if not isinstance(inference["settings"], dict) or not inference["settings"]:
        raise ValueError("Record effective inference settings, including relevant backend defaults")
    reject_secrets(value)
    return value


def reject_secrets(value) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {
                "api_key", "password", "secret", "token", "authorization", "credentials",
                "hf_token", "access_token", "api_token", "openai_api_key", "github_token",
                "client_secret", "aws_secret_access_key", "cscs_serving_api", "env", "environment",
            }:
                raise ValueError("Artifacts must not contain credentials")
            reject_secrets(item)
    elif isinstance(value, list):
        for item in value:
            reject_secrets(item)


def validate_rows(rows: list[dict], name: str) -> None:
    if not isinstance(rows, list) or not rows:
        raise ValueError("Acceptance requires nonempty frozen inputs")
    ids = []
    sources = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Acceptance examples must be objects")
        ids.append(nonempty(row.get("id"), "case ID"))
        nonempty(row.get("cluster_id"), "independent prompt/article cluster ID")
        nonempty(row.get("stratum"), "case stratum")
        if not isinstance(row.get("doc"), dict) or not row["doc"]:
            raise ValueError("Every example requires its exact scoring document")
        if set(row["doc"]) & {"human_scores", "expert_annotations", "turker_annotations", "expected_scores"}:
            raise ValueError("Keep held-out human/oracle labels outside model/judge evidence")
        scoring.validate_document(name, row["doc"])
        if name == "summeval":
            source_id = row["doc"]["source_id"]
            if row["cluster_id"] != source_id:
                raise ValueError("SummEval clusters must be source article IDs, not summary IDs")
            if sources.setdefault(source_id, row["doc"]["source"]) != row["doc"]["source"]:
                raise ValueError("A SummEval source ID maps to conflicting articles")
        if name == "mt_bench" and (
            row["cluster_id"] != str(row["doc"]["question_id"])
            or row["stratum"] != row["doc"]["category"]
        ):
            raise ValueError("MT-Bench clusters/strata must preserve question IDs/categories")
        if not isinstance(row.get("requests"), list) or not row["requests"] or any(
            not isinstance(request, dict) or not request for request in row["requests"]
        ):
            raise ValueError("Freeze effective request templates, not just model output")
        if name != "sycophancy":
            count = 2 if name in ("mt_bench", "ifbench_multiturn") else 1
            if not isinstance(row.get("answers"), list) or len(row["answers"]) != count or any(
                not isinstance(answer, str) or not answer.strip() for answer in row["answers"]
            ):
                raise ValueError("Answers must be nonempty and include every evaluated turn")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate acceptance case IDs")
    reject_secrets(rows)


def load_evidence(plan: dict) -> tuple[list[dict], dict]:
    for artifact in (plan["inputs"], plan["expected"]):
        if file_hash(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError("Frozen acceptance evidence changed")
    rows = read_jsonl(Path(plan["inputs"]["path"]))
    validate_rows(rows, plan["benchmark"])
    expected = read_json(Path(plan["expected"]["path"]))
    if not isinstance(expected, dict) or expected.get("schema_version") != 1 or expected.get("inputs_sha256") != plan["inputs"]["sha256"]:
        raise ValueError("The oracle must identify the exact frozen input file")
    for key in ("source", "revision", "protocol", "label_origin"):
        nonempty(expected.get(key), f"oracle {key}")
    cases = expected.get("cases")
    if not isinstance(cases, list) or len(cases) != len(rows) or any(
        not isinstance(case, dict) for case in cases
    ):
        raise ValueError("The oracle must cover every selected case")
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids) or set(ids) != {row["id"] for row in rows}:
        raise ValueError("Oracle IDs must exactly match the frozen inputs")
    by_id = {case["id"]: case for case in cases}
    for row in rows:
        case = by_id[row["id"]]
        if case.get("input_sha256") != digest(row):
            raise ValueError(f"Oracle case/input alignment failed: {row['id']}")
        scoring.validate_scores(plan["benchmark"], case["scores"], row["doc"])
    if plan["kind"] == "judge_calibration" and expected["label_origin"] not in (
        "original_judgments", "human_annotations"
    ):
        raise ValueError("Judge calibration needs independently sourced original/human labels")
    if plan["kind"] != "judge_calibration":
        if not isinstance(expected.get("aggregate"), dict):
            raise ValueError("Replay/regeneration require an independently captured aggregation oracle")
    return rows, expected


def validate_policy(policy: dict, kind: str) -> None:
    integer(policy.get("min_cases"), "min_cases", 1, 10000000)
    if kind == "scorer_replay":
        if set(policy) != {"min_cases", "float_tolerance"}:
            raise ValueError("Replay policy accepts only min_cases and float_tolerance")
        number(policy.get("float_tolerance"), "float_tolerance", 0, 1e-6)
        return
    keys = {"min_cases", "min_clusters", "bootstrap_iterations", "seed", "confidence", "strata", "metrics"}
    if kind == "reference_regeneration":
        keys.add("reference_replay_tolerance")
        number(policy.get("reference_replay_tolerance"), "reference_replay_tolerance", 0, 1e-6)
    if set(policy) != keys:
        raise ValueError("Acceptance policy contains missing or unsupported settings")
    integer(policy.get("min_clusters"), "min_clusters", 2, 10000000)
    integer(policy.get("bootstrap_iterations"), "bootstrap_iterations", 200, 100000)
    integer(policy.get("seed"), "bootstrap seed", 0, 2**32 - 1)
    number(policy.get("confidence"), "confidence", 0.5, 0.999)
    strata = policy.get("strata")
    if not isinstance(strata, list) or len(set(strata)) != len(strata) or any(
        not isinstance(item, str) or not item.strip() or item == "overall" for item in strata
    ):
        raise ValueError("Predeclare unique strata; overall is always evaluated")
    metrics = policy.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("Predeclare metric-specific acceptance thresholds")
    for criterion in metrics.values():
        if not isinstance(criterion, dict):
            raise ValueError("Every metric needs its own acceptance criteria")
        if kind == "reference_regeneration":
            if set(criterion) != {"equivalence_margin"}:
                raise ValueError("Regeneration metrics accept only an equivalence_margin")
            number(criterion.get("equivalence_margin"), "equivalence_margin", 0, 10)
        else:
            if set(criterion) - {
                "max_abs_bias", "max_mae", "min_cluster_agreement", "agreement_tolerance", "min_spearman"
            }:
                raise ValueError("Unknown judge acceptance criterion")
            number(criterion.get("max_abs_bias"), "max_abs_bias", 0, 10)
            number(criterion.get("max_mae"), "max_mae", 0, 10)
            number(criterion.get("min_cluster_agreement"), "min_cluster_agreement", 0, 1)
            number(criterion.get("agreement_tolerance"), "agreement_tolerance", 0, 10)
            if "min_spearman" in criterion:
                number(criterion["min_spearman"], "min_spearman", -1, 1)


def create_plan(config_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("A sealed plan cannot be overwritten")
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    if config.get("schema_version") != 1 or config.get("kind") not in KINDS:
        raise ValueError("Expected schema_version=1 and one of the three acceptance kinds")
    adapters.validate_benchmark(config["benchmark"])
    if config.get("protocol") != adapters.PROTOCOLS[config["benchmark"]]:
        raise ValueError("Protocol must identify the implemented scorer, including adapted judge labels")
    nonempty(config.get("scope"), "evaluation scope")
    nonempty(config.get("data_license"), "data license")
    validate_policy(config["policy"], config["kind"])
    plan = {
        "schema_version": 1, "kind": config["kind"], "benchmark": config["benchmark"],
        "protocol": config["protocol"], "scope": config["scope"],
        "data_license": config["data_license"], "policy": config["policy"],
        "inputs": checked_artifact(config["inputs"], config_path.parent),
        "expected": checked_artifact(config["expected"], config_path.parent),
        "runtime": fingerprint(), "assets": {},
        "published_protocol_equivalence": False,
    }
    for key, asset in config.get("assets", {}).items():
        if key not in ("nltk", "alpaca"):
            raise ValueError(f"Unknown scorer asset family: {key}")
        directory = (config_path.parent / asset["directory"]).resolve()
        expected_hash = sha256(asset["sha256"])
        files = native.asset_hashes(directory)
        if not files or digest(files) != expected_hash:
            raise ValueError(f"Scorer assets do not match: {key}")
        plan["assets"][key] = {"directory": str(directory), "sha256": expected_hash}
    if config["kind"] != "scorer_replay":
        artifact = checked_artifact(config["deployment"], config_path.parent)
        plan["deployment"] = deployment(read_json(Path(artifact["path"])))
        plan["deployment_assurance"] = "operator_assertion_not_verified_server_state"
        decoding = config.get("decoding")
        if not isinstance(decoding, dict) or not decoding:
            raise ValueError("Record every effective decoding setting; no implicit defaults")
        reject_secrets(decoding)
        plan["decoding"] = decoding
    if config["kind"] == "judge_calibration":
        if config["benchmark"] not in adapters.JUDGED:
            raise ValueError("This benchmark does not use a replacement judge")
        plan["rubric_sha256"] = digest(scoring.rubric(config["benchmark"], plan["deployment"]["model_id"]))
        if plan["decoding"] != scoring.judge_decoding(config["benchmark"], plan["deployment"]["model_id"]):
            raise ValueError("Judge decoding does not match the benchmark runner being calibrated")
    rows, expected = load_evidence(plan)
    if len(rows) < plan["policy"]["min_cases"]:
        raise ValueError("Frozen fixture is smaller than the predefined minimum")
    if plan["kind"] != "scorer_replay":
        metrics = set(scoring.vectors(plan["benchmark"], expected["cases"][0]["scores"]))
        if plan["kind"] == "reference_regeneration" and plan["benchmark"] == "alpaca_eval_2_lc":
            metrics.add("length_controlled_winrate")
        if set(plan["policy"]["metrics"]) != metrics:
            raise ValueError("Predeclare thresholds for every benchmark metric, not a favorable subset")
    if config["kind"] == "reference_regeneration":
        for row in rows:
            count = 2 if plan["benchmark"] in ("mt_bench", "ifbench_multiturn") else 1
            if len(row["requests"]) != count:
                raise ValueError("Regeneration requires one explicit request template per evaluated turn")
            for turn, request in enumerate(row["requests"]):
                if plan["benchmark"] == "sycophancy":
                    if set(request) != {"prompt", "choices"} or request["choices"] != row["doc"]["choices"]:
                        raise ValueError("Freeze the actual likelihood prompt and ordered continuations")
                    nonempty(request["prompt"], "likelihood prompt")
                elif set(request) == {"input_text"}:
                    nonempty(request["input_text"], "encoder-decoder input")
                else:
                    render_request(request, row["answers"][:turn])
        if expected.get("deployment") != plan["deployment"] or expected.get("decoding") != plan["decoding"]:
            raise ValueError("Reference weights/tokenizer/template/inference/decoding pins are missing or differ")
        if "judge" in config:
            judge = config["judge"]
            plan["judge"] = {
                "deployment": deployment(judge["deployment"]),
                "decoding": judge["decoding"],
                "rubric_sha256": sha256(judge["rubric_sha256"]),
            }
            if plan["judge"]["rubric_sha256"] != digest(scoring.rubric(
                plan["benchmark"], plan["judge"]["deployment"]["model_id"]
            )):
                raise ValueError("Regeneration judge rubric does not match the implemented protocol")
        if config["benchmark"] in adapters.JUDGED and "judge" not in plan:
            raise ValueError("Judge-based regeneration must also pin the scoring judge/rubric")
    plan["sha256"] = digest(plan)
    write_json(output, plan)
    return plan


def load_plan(path: Path) -> dict:
    plan = read_json(path)
    if not isinstance(plan, dict) or plan.get("schema_version") != 1 or plan.get("kind") not in KINDS:
        raise ValueError("Unsupported acceptance plan schema")
    expected = plan.pop("sha256", None)
    if expected != digest(plan):
        raise ValueError("Acceptance plan integrity failed")
    plan["sha256"] = expected
    if plan["runtime"] != fingerprint():
        raise ValueError("Scorer source/runtime changed since the acceptance plan was sealed")
    validate_policy(plan["policy"], plan["kind"])
    return plan


def differences(expected, actual, tolerance: float, path: str = "") -> list[dict]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return [{"path": path, "expected_keys": sorted(expected), "actual_keys": sorted(actual)}]
        return [
            mismatch for key in expected
            for mismatch in differences(expected[key], actual[key], tolerance, f"{path}/{key}")
        ]
    if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        return [
            mismatch for index, (left, right) in enumerate(zip(expected, actual))
            for mismatch in differences(left, right, tolerance, f"{path}/{index}")
        ]
    equal = type(expected) is type(actual) and expected == actual
    if "/denominators/" not in path and type(expected) in (int, float) and type(actual) in (int, float):
        equal = math.isfinite(expected) and math.isfinite(actual) and (
            expected == actual if type(expected) is int and type(actual) is int
            else abs(expected - actual) <= tolerance
        )
    return [] if equal else [{"path": path, "expected": expected, "actual": actual}]


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def cluster_interval(successes: int, count: int, alpha: float) -> list[float]:
    z = NormalDist().inv_cdf(1 - alpha / 2)
    rate = successes / count
    divisor = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / divisor
    radius = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count**2)) / divisor
    return [max(0, center - radius), min(1, center + radius)]


def statistics(
    plan: dict, rows: list[dict], reference: list[dict], candidate: list[dict],
    candidate_evidence: list[dict] | None = None,
) -> dict:
    policy, name = plan["policy"], plan["benchmark"]
    candidate_evidence = rows if candidate_evidence is None else candidate_evidence
    groups = {"overall": list(range(len(rows)))}
    for stratum in policy["strata"]:
        groups[stratum] = [index for index, row in enumerate(rows) if row["stratum"] == stratum]
    metric_names = set(policy["metrics"])
    alpha = (1 - policy["confidence"]) / (len(groups) * len(metric_names) * (
        3 if plan["kind"] == "judge_calibration" else 1
    ))
    rng = random.Random(policy["seed"])
    result = {}
    for group, indices in groups.items():
        clusters = {}
        for index in indices:
            clusters.setdefault(rows[index]["cluster_id"], []).append(index)
        if len(indices) < policy["min_cases"] or len(clusters) < policy["min_clusters"]:
            raise ValueError(f"{group}: insufficient cases/independent clusters for the predefined policy")
        selected = [rows[index] for index in indices]
        selected_candidate = [candidate_evidence[index] for index in indices]
        left = [reference[index] for index in indices]
        right = [candidate[index] for index in indices]
        if plan["kind"] == "reference_regeneration":
            left_aggregate = scoring.aggregate(name, selected, left)["metrics"]
            right_aggregate = scoring.aggregate(name, selected_candidate, right)["metrics"]
        else:
            left_aggregate = scoring.aggregate(name, selected, left, length_control=False)["metrics"]
            right_aggregate = scoring.aggregate(name, selected, right, length_control=False)["metrics"]
        if metric_names - left_aggregate.keys() or metric_names - right_aggregate.keys():
            raise ValueError("Acceptance policy names an unavailable metric")
        draws = {key: {"bias": [], "mae": []} for key in metric_names}
        cluster_list = list(clusters.values())
        for _ in range(policy["bootstrap_iterations"]):
            sampled = [index for _ in cluster_list for index in rng.choice(cluster_list)]
            sample_rows = [rows[index] for index in sampled]
            sample_candidate_rows = [candidate_evidence[index] for index in sampled]
            sample_left = [reference[index] for index in sampled]
            sample_right = [candidate[index] for index in sampled]
            if plan["kind"] == "reference_regeneration":
                a = scoring.aggregate(name, sample_rows, sample_left)["metrics"]
                b = scoring.aggregate(name, sample_candidate_rows, sample_right)["metrics"]
                for key in metric_names:
                    draws[key]["bias"].append(b[key] - a[key])
            else:
                for key in metric_names:
                    pairs = [
                        pair for a, b in zip(sample_left, sample_right)
                        for pair in zip(scoring.vectors(name, a)[key], scoring.vectors(name, b)[key])
                    ]
                    draws[key]["bias"].append(mean([b - a for a, b in pairs]))
                    draws[key]["mae"].append(mean([abs(b - a) for a, b in pairs]))
        metrics = {}
        for key, criterion in policy["metrics"].items():
            interval = [quantile(draws[key]["bias"], alpha / 2), quantile(draws[key]["bias"], 1 - alpha / 2)]
            entry = {
                "reference": left_aggregate[key], "candidate": right_aggregate[key],
                "bias": right_aggregate[key] - left_aggregate[key],
                "bias_interval": interval, "degenerate_bootstrap_interval": interval[0] == interval[1],
            }
            if plan["kind"] == "reference_regeneration":
                margin = criterion["equivalence_margin"]
                entry["passed"] = interval[0] >= -margin and interval[1] <= margin
            else:
                paired = [
                    pair for a, b in zip(left, right)
                    for pair in zip(scoring.vectors(name, a)[key], scoring.vectors(name, b)[key])
                ]
                tolerance = criterion["agreement_tolerance"]
                agreements = [abs(a - b) <= tolerance for a, b in paired]
                successful_clusters = sum(all(
                    abs(a - b) <= tolerance
                    for index in cluster
                    for a, b in zip(scoring.vectors(name, reference[index])[key], scoring.vectors(name, candidate[index])[key])
                ) for cluster in cluster_list)
                agreement_interval = cluster_interval(successful_clusters, len(clusters), alpha)
                mae_interval = [
                    quantile(draws[key]["mae"], alpha / 2),
                    quantile(draws[key]["mae"], 1 - alpha / 2),
                ]
                a, b = [pair[0] for pair in paired], [pair[1] for pair in paired]
                spearman = adapters.correlation(adapters.ranks(a), adapters.ranks(b))
                entry.update({
                    "agreement": mean(agreements), "agreement_tolerance": tolerance,
                    "cluster_agreement": successful_clusters / len(clusters),
                    "cluster_agreement_interval": agreement_interval,
                    "mae": mean([abs(x - y) for x, y in paired]), "mae_interval": mae_interval,
                    "pearson": adapters.correlation(a, b), "spearman": spearman,
                })
                entry["passed"] = (
                    interval[0] >= -criterion["max_abs_bias"]
                    and interval[1] <= criterion["max_abs_bias"]
                    and mae_interval[1] <= criterion["max_mae"]
                    and agreement_interval[0] >= criterion["min_cluster_agreement"]
                    and ("min_spearman" not in criterion or (
                        spearman is not None and spearman >= criterion["min_spearman"]
                    ))
                )
            metrics[key] = entry
        result[group] = {
            "cases": len(indices), "clusters": len(clusters), "metrics": metrics,
            "confusion": scoring.confusion(name, left, right),
        }
    return {
        "method": "paired_prompt_or_article_cluster_percentile_bootstrap",
        "agreement_method": "Wilson_interval_for_clusters_with_all_verdicts_within_tolerance",
        "multiple_comparisons": "Bonferroni_metrics_strata_and_gated_intervals",
        "family_confidence": policy["confidence"], "iterations": policy["bootstrap_iterations"],
        "alpaca_length_control": "GLM_refit_in_each_regeneration_bootstrap_draw" if name == "alpaca_eval_2_lc" else None,
        "groups": result,
        "passed": all(entry["passed"] for group in result.values() for entry in group["metrics"].values()),
    }


def scorer_assets(plan: dict, stack: ExitStack) -> None:
    for key, asset in plan["assets"].items():
        if digest(native.asset_hashes(Path(asset["directory"]))) != asset["sha256"]:
            raise ValueError("Scorer assets changed after plan sealing")
    if plan["benchmark"] in ("ifeval", "ifbench", "ifbench_multiturn"):
        if "nltk" not in plan["assets"]:
            raise ValueError("Instruction replay requires prepared, checksummed NLTK assets")
        stack.enter_context(native.offline_scorers(Path(plan["assets"]["nltk"]["directory"])))
    if plan["benchmark"] == "alpaca_eval_2_lc" and plan["kind"] != "judge_calibration":
        if "alpaca" not in plan["assets"]:
            raise ValueError("LC replay/regeneration requires the frozen GLM feature asset")
        stack.enter_context(native.offline_alpaca_assets(Path(plan["assets"]["alpaca"]["directory"])))
    stack.enter_context(native.environment({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}))


def candidate_rows(plan: dict, path: Path, frozen: list[dict]) -> tuple[list[dict], dict]:
    contents = path.read_bytes()
    artifact = json.loads(contents)
    if not isinstance(artifact, dict) or artifact.get("schema_version") != 1 or artifact.get("acceptance_plan_sha256") != plan["sha256"]:
        raise ValueError("Candidate artifacts must identify the predeclared acceptance plan")
    if artifact.get("inputs_sha256") != plan["inputs"]["sha256"]:
        raise ValueError("Candidate artifacts evaluated a different dataset/scope")
    if artifact.get("status", "succeeded") != "succeeded":
        raise ValueError("Candidate producer reported incomplete generation/judging")
    if artifact.get("deployment") != plan["deployment"] or artifact.get("decoding") != plan["decoding"]:
        raise ValueError("Candidate weights/tokenizer/template/inference/decoding differ from the pins")
    if plan["kind"] == "judge_calibration" and artifact.get("rubric_sha256") != plan["rubric_sha256"]:
        raise ValueError("Candidate judge did not use the predeclared rubric")
    if plan["kind"] == "reference_regeneration" and artifact.get("judge") != plan.get("judge"):
        raise ValueError("Regeneration's judge/protocol differs from the predeclared scoring judge")
    rows = artifact.get("cases")
    validate_rows(rows, plan["benchmark"])
    by_id = {row["id"]: row for row in rows}
    if set(by_id) != {row["id"] for row in frozen}:
        raise ValueError("Missing/extra candidate examples; failures must not be dropped")
    for row in frozen:
        candidate = by_id[row["id"]]
        if candidate.get("status", "succeeded") != "succeeded" or (
            plan["kind"] == "reference_regeneration" and candidate.get("generation_truncated") is True
        ):
            raise ValueError(f"Candidate generation/judging did not complete: {row['id']}")
        for key in ("doc", "requests", "cluster_id", "stratum"):
            if differences(row[key], candidate[key], 0):
                raise ValueError(f"Candidate changed frozen input {row['id']}/{key}")
        if plan["kind"] == "judge_calibration" and row["answers"] != candidate["answers"]:
            raise ValueError("Judge calibration must compare judgments of identical frozen answers")
    reject_secrets(artifact)
    return [by_id[row["id"]] for row in frozen], {
        "path": str(path.resolve()), "sha256": hashlib.sha256(contents).hexdigest(), "artifact": artifact,
    }


def evaluate(plan: dict, rows: list[dict], oracle: dict, candidate: list[dict], progress=None) -> dict:
    name = plan["benchmark"]
    by_id = {case["id"]: case["scores"] for case in oracle["cases"]}
    reference = [by_id[row["id"]] for row in rows]
    actual, cases = [], []
    for row in candidate:
        try:
            scores = scoring.replay_case(name, row)
        except (ValueError, KeyError, TypeError, ImportError, OSError, RuntimeError):
            if progress is not None:
                progress({
                    "scored_cases": len(cases), "failed_cases": 1,
                    "unprocessed_cases": len(rows) - len(cases) - 1, "cases": cases,
                    "failed_case_id": row["id"],
                })
            raise
        actual.append(scores)
        cases.append({
            "id": row["id"], "reference": by_id[row["id"]], "actual": scores,
            "differences": differences(
                by_id[row["id"]], scores,
                plan["policy"]["float_tolerance"] if plan["kind"] == "scorer_replay" else 0,
            ),
            "answers_exact_match": row.get("answers") == rows[len(cases)].get("answers"),
            "replay_stage": row.get("replay_stage", "raw_parser_or_verifier"),
        })
        if progress is not None:
            progress({
                "scored_cases": len(cases), "failed_cases": 0,
                "unprocessed_cases": len(rows) - len(cases), "cases": cases,
            })
    length_control = plan["kind"] != "judge_calibration"
    aggregate = scoring.aggregate(name, candidate, actual, length_control=length_control)
    result = {
        "selected_cases": len(rows), "scored_cases": len(cases), "failed_cases": 0,
        "unprocessed_cases": 0, "cases": cases, "aggregate": aggregate,
        "reference_protocol": oracle["protocol"], "reference_label_origin": oracle["label_origin"],
        "published_protocol_equivalence": False,
    }
    if plan["kind"] == "scorer_replay":
        result["aggregate_differences"] = differences(
            oracle["aggregate"], aggregate, plan["policy"]["float_tolerance"]
        )
        result["passed"] = not result["aggregate_differences"] and all(
            not case["differences"] for case in cases
        )
    else:
        if plan["kind"] == "reference_regeneration":
            replay_aggregate = scoring.aggregate(name, rows, reference)
            mismatches = differences(
                oracle["aggregate"], replay_aggregate, plan["policy"]["reference_replay_tolerance"]
            )
            if mismatches:
                raise ValueError("Reference aggregation is not reproducible; run scorer replay before regeneration")
        result["uncertainty"] = statistics(plan, rows, reference, actual, candidate)
        result["passed"] = result["uncertainty"]["passed"]
    return result


def run_plan(plan_path: Path, candidate_path: Path | None, output: Path) -> dict:
    if output.exists():
        raise ValueError("Acceptance output exists; do not overwrite frozen evidence")
    plan = load_plan(plan_path)
    rows, oracle = load_evidence(plan)
    report = {
        "schema_version": 1, "kind": plan["kind"], "benchmark": plan["benchmark"],
        "protocol": plan["protocol"], "scope": plan["scope"], "status": "incomplete",
        "plan": {"path": str(plan_path.resolve()), "sha256": plan["sha256"]},
        "runtime": fingerprint(), "selected_cases": len(rows), "scored_cases": 0,
        "failed_cases": 0, "unprocessed_cases": len(rows),
        "published_protocol_equivalence": False, "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "report.json", report)
    try:
        if plan["kind"] == "scorer_replay":
            if candidate_path is not None:
                raise ValueError("Scorer replay uses only frozen input artifacts")
            candidate = rows
        else:
            if candidate_path is None:
                raise ValueError("Supply actual regenerated/judged artifacts with --candidate")
            candidate, report["candidate"] = candidate_rows(plan, candidate_path, rows)
            report["deployment_assurance"] = plan["deployment_assurance"]
            write_json(output / "candidate.json", report["candidate"].pop("artifact"))
            report["candidate"] = {
                "path": str((output / "candidate.json").resolve()),
                "sha256": file_hash(output / "candidate.json"),
            }

        def progress(value):
            report.update(value)
            write_json(output / "report.json", report)

        with ExitStack() as stack:
            scorer_assets(plan, stack)
            report.update(evaluate(plan, rows, oracle, candidate, progress))
        report["status"] = "passed" if report["passed"] else "failed"
    except (ValueError, KeyError, TypeError, ImportError, OSError, RuntimeError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["passed"] = False
    report["sha256"] = digest(report)
    write_json(output / "report.json", report)
    return report


def freeze(report_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("A judge lock cannot be overwritten")
    report = read_json(report_path)
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("Unsupported calibration report schema")
    seal = report.pop("sha256", None)
    if seal != digest(report):
        raise ValueError("Calibration report integrity failed")
    if report.get("kind") != "judge_calibration" or report.get("status") != "passed":
        raise ValueError("Only a complete, passing judge calibration can be frozen")
    plan = load_plan(Path(report["plan"]["path"]))
    if plan["sha256"] != report["plan"]["sha256"]:
        raise ValueError("Calibration plan changed")
    rows, oracle = load_evidence(plan)
    path = Path(report["candidate"]["path"])
    if file_hash(path) != report["candidate"]["sha256"]:
        raise ValueError("Judgment evidence changed")
    candidate, _ = candidate_rows(plan, path, rows)
    with ExitStack() as stack:
        scorer_assets(plan, stack)
        verified = evaluate(plan, rows, oracle, candidate)
    if not verified["passed"] or any(
        report.get(key) != verified[key] for key in verified
    ):
        raise ValueError("Stored acceptance results do not reproduce; judge freeze refused")
    lock = {
        "schema_version": 1, "benchmark": plan["benchmark"], "protocol": plan["protocol"],
        "scope": plan["scope"], "deployment": plan["deployment"],
        "deployment_assurance": plan["deployment_assurance"], "decoding": plan["decoding"],
        "rubric_sha256": plan["rubric_sha256"], "runtime": plan["runtime"],
        "policy": plan["policy"], "calibration_report_sha256": seal,
        "acceptance_plan_sha256": plan["sha256"], "inputs_sha256": plan["inputs"]["sha256"],
        "expected_sha256": plan["expected"]["sha256"],
        "reference_protocol": oracle["protocol"], "reference_label_origin": oracle["label_origin"],
        "checked_strata": plan["policy"]["strata"],
        "observed_strata": sorted({row["stratum"] for row in rows}),
        "published_protocol_equivalence": False,
    }
    lock["sha256"] = digest(lock)
    write_json(output, lock)
    return lock


def validate_judge_lock(
    path: Path, name: str, judge: dict, base: Path, required_strata: list[str] | None = None,
) -> dict:
    lock = read_json(path)
    if not isinstance(lock, dict) or lock.get("schema_version") != 1:
        raise ValueError("Unsupported judge lock schema")
    seal = lock.pop("sha256", None)
    if seal != digest(lock):
        raise ValueError("Judge lock integrity failed")
    if lock["benchmark"] != name or lock["protocol"] != adapters.PROTOCOLS[name]:
        raise ValueError("Frozen judge belongs to another benchmark/protocol")
    if lock["runtime"] != fingerprint():
        raise ValueError("Frozen judge's scorer source/runtime changed; recalibrate explicitly")
    if lock["deployment"]["model_id"] != endpoint_model(judge):
        raise ValueError("Served judge model differs from the frozen calibration")
    asserted = deployment(read_json((base / judge["deployment_manifest"]).resolve()))
    if asserted != lock["deployment"]:
        raise ValueError("Judge deployment pins changed; recalibrate explicitly")
    if lock["rubric_sha256"] != digest(scoring.rubric(name, endpoint_model(judge))):
        raise ValueError("Judge rubric changed; recalibrate explicitly")
    if lock["decoding"] != scoring.judge_decoding(name, endpoint_model(judge)):
        raise ValueError("Frozen judge decoding differs from the benchmark runner")
    if required_strata and set(required_strata) - set(lock["checked_strata"]):
        raise ValueError("Judge was not accepted for every selected MT-Bench category")
    return {"sha256": seal, "calibration_report_sha256": lock["calibration_report_sha256"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--config", required=True, type=Path)
    plan_parser.add_argument("--output", required=True, type=Path)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--plan", required=True, type=Path)
    run_parser.add_argument("--candidate", type=Path)
    run_parser.add_argument("--output", required=True, type=Path)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--report", required=True, type=Path)
    freeze_parser.add_argument("--output", required=True, type=Path)
    for command in ("regenerate", "judge"):
        producer = subparsers.add_parser(command)
        producer.add_argument("--plan", required=True, type=Path)
        producer.add_argument("--endpoint", required=True, type=Path)
        producer.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            create_plan(args.config, args.output)
        elif args.command == "freeze":
            freeze(args.report, args.output)
        elif args.command in ("regenerate", "judge"):
            from scripts.helpfulness.acceptance_producers import produce

            artifact = produce(args.plan, args.endpoint, args.output, args.command)
            print(json.dumps({"status": artifact["status"], "kind": args.command}))
            if artifact["status"] != "succeeded":
                print(artifact.get("error", "Candidate generation/judging incomplete; inspect per-case errors"), file=sys.stderr)
                return 1
        else:
            report = run_plan(args.plan, args.candidate, args.output)
            print(json.dumps({"status": report["status"], "kind": report["kind"]}))
            if report.get("error"):
                print(report["error"], file=sys.stderr)
            return 0 if report["status"] == "passed" else 1
    except (ValueError, KeyError, TypeError, OSError, ImportError) as error:
        print(f"Acceptance check failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
