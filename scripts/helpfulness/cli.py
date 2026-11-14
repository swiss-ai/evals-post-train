"""python -m scripts.helpfulness.cli {prepare,run} --suite suite.toml."""

import argparse
import csv
import importlib.metadata
import json
import re
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from scripts.helpfulness import native
from scripts.helpfulness.adapters import (
    CUSTOM_TASKS, EXTERNAL_TASKS, JUDGED, NATIVE_TASKS, PROTOCOLS, generate_case, load_custom,
    score_case, summarize_custom, summarize_native, validate_benchmark,
)
from scripts.helpfulness.core import (
    ChatClient, append_jsonl, digest, endpoint_key, endpoint_model, endpoint_url,
    number, public_endpoint, read_json, read_jsonl, write_json,
)


def safe_name(name: str) -> str:
    if (
        not isinstance(name, str) or name in (".", "..")
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name)
    ):
        raise ValueError("Names must be path-safe: letters, digits, '.', '_', '-'")
    return name


def load_suite(path: Path) -> dict:
    with path.open("rb") as stream:
        suite = tomllib.load(stream)
    if suite.get("schema_version") != 1:
        raise ValueError("Expected schema_version = 1")
    for config in [suite.get("judge", {}), *[row.get("endpoint", {}) for row in suite.get("checkpoints", [])],
                   *suite.get("reference_models", {}).values()]:
        if "provenance_file" in config:
            config["provenance_file"] = str((path.parent / config["provenance_file"]).resolve())
    for key in ("behavior_spec", "style_spec", "code_spec"):
        if key in suite:
            source = (path.parent / suite[key]).resolve()
            suite[key + "_digest"] = digest(source.read_text(encoding="utf-8"))
    number(suite.get("max_tokens", 2048), "max_tokens", 1, 1000000)
    if not isinstance(suite.get("max_tokens", 2048), int):
        raise ValueError("max_tokens must be an integer")
    checkpoints = suite.get("checkpoints", [])
    benchmarks = suite.get("benchmarks", [])
    if not checkpoints or not benchmarks:
        raise ValueError("Suite requires checkpoints and benchmarks")
    names = []
    for checkpoint in checkpoints:
        names.append(safe_name(checkpoint["name"]))
        for key in ("path", "revision", "tokenizer_revision", "chat_template_revision"):
            if not isinstance(checkpoint.get(key), str) or not checkpoint[key].strip():
                raise ValueError(f"Checkpoint requires provenance field {key}")
        if "api_key" in checkpoint["endpoint"]:
            raise ValueError("Credentials must be environment variables, never suite fields")
        endpoint_url(checkpoint["endpoint"])
        number(checkpoint["endpoint"].get("requests_per_minute", 14), "requests_per_minute", 0, 100000)
    if len(names) != len(set(names)):
        raise ValueError("Checkpoint names must be unique")
    names = []
    for benchmark in benchmarks:
        name = benchmark["name"]
        validate_benchmark(name)
        from scripts.helpfulness import open_benchmarks

        if name in open_benchmarks.NAMES | EXTERNAL_TASKS:
            if suite.get("purpose") != "helpfulness_chitchat_teacher" or suite.get("model_policy") != "open_weights_only":
                raise ValueError("Teacher extensions require purpose=helpfulness_chitchat_teacher and model_policy=open_weights_only")
            for key in ("upstream_directory", "upstream_terms"):
                if not isinstance(benchmark.get(key), str) or not benchmark[key].strip():
                    raise ValueError(f"{name} requires {key}; upstream material is not redistributed")
            benchmark["upstream_directory"] = str((path.parent / benchmark["upstream_directory"]).resolve())
            if "nltk_directory" in benchmark:
                benchmark["nltk_directory"] = str((path.parent / benchmark["nltk_directory"]).resolve())
        names.append(name)
        mode = benchmark.get("mode", "generate")
        if mode not in ("generate", "validate") or (mode == "validate" and name != "summeval"):
            raise ValueError("Only SummEval supports mode=validate")
        if name in CUSTOM_TASKS:
            for key in ("source_file", "source_origin", "source_revision", "data_license"):
                if not isinstance(benchmark.get(key), str) or not benchmark[key].strip():
                    raise ValueError(f"{name} requires {key}")
            if name == "ifbench_multiturn" and "IFBench_multi-turn" not in benchmark["source_origin"]:
                raise ValueError("IFBench multiturn must identify the held-out IFBench_multi-turn release")
        if benchmark.get("categories") and name != "mt_bench":
            raise ValueError("Category filtering is supported only for MT-Bench")
    if len(names) != len(set(names)):
        raise ValueError("Benchmark names must be unique")
    judge = suite.get("judge", {})
    if any(name in JUDGED for name in names):
        if not judge:
            raise ValueError("This suite requires an independent judge")
        if "api_key" in judge:
            raise ValueError("Judge credentials must be environment variables")
        endpoint_url(judge)
        number(judge.get("requests_per_minute", 14), "requests_per_minute", 0, 100000)
        if type(judge.get("require_calibration", False)) is not bool:
            raise ValueError("judge.require_calibration must be a boolean")
        profiles = judge.get("calibration_profiles", {})
        if not isinstance(profiles, dict) or set(profiles) - JUDGED:
            raise ValueError("Calibration profiles must identify judge-based benchmarks")
        if any(not isinstance(value, str) or not value.strip() for value in profiles.values()):
            raise ValueError("Calibration profile paths must be nonempty strings")
        if profiles or judge.get("require_calibration"):
            if not isinstance(judge.get("deployment_manifest"), str) or not judge["deployment_manifest"]:
                raise ValueError("Frozen-judge enforcement requires an operator-exported deployment manifest")
            judge["deployment_manifest"] = str((path.parent / judge["deployment_manifest"]).resolve())
            judge["calibration_profiles"] = {
                key: str((path.parent / value).resolve()) for key, value in profiles.items()
            }
    return suite


def chosen(entries: list[dict], requested: list[str] | None) -> list[dict]:
    if not requested:
        return entries
    unknown = set(requested) - {entry["name"] for entry in entries}
    if unknown:
        raise ValueError(f"Unknown selection: {sorted(unknown)}")
    return [entry for entry in entries if entry["name"] in requested]


def select_documents(name: str, documents: dict, categories: list[str], limit: int | None) -> dict:
    if categories:
        available = {row["category"] for row in documents["mt_bench"]}
        if set(categories) - available:
            raise ValueError(f"Unknown MT-Bench categories: {sorted(set(categories) - available)}")
    result = {}
    for task, rows in documents.items():
        selected = [
            row for row in rows
            if not categories or row["category"] in categories
        ]
        if limit is not None:
            selected = selected[:limit]
        if not selected:
            raise ValueError(f"{name}/{task} has no selected examples")
        result[task] = selected
    return result


def prepare(suite: dict, manifest: Path, bundle: Path, benchmarks: list[dict]) -> None:
    if bundle.exists():
        raise ValueError("Bundle already exists; use a new directory to preserve provenance")
    loaded = {}
    # Check mandatory local sources before any native dataset acquisition.
    for benchmark in benchmarks:
        name = benchmark["name"]
        from scripts.helpfulness import eqbench, open_benchmarks

        if name in EXTERNAL_TASKS:
            loaded[name] = eqbench.prepare(benchmark)
            continue
        if name in CUSTOM_TASKS:
            for key in ("source_revision", "data_license"):
                if benchmark[key].startswith("REPLACE_WITH_"):
                    raise ValueError(f"{name}: verify and replace {key} before preparation")
            source = (manifest.parent / benchmark["source_file"]).resolve()
            loaded[name] = {
                "origin": benchmark["source_origin"], "revision": benchmark["source_revision"],
                "data_license": benchmark["data_license"],
            }
            if name in open_benchmarks.NAMES:
                loaded[name].update(open_benchmarks.prepare(name, source, benchmark))
            else:
                loaded[name]["cases"] = load_custom(name, source, benchmark.get("mode", "generate"))
    index = {"schema_version": 1, "suite_digest": digest(suite), "benchmarks": {}}
    if any(benchmark["name"] == "alpaca_eval_2_lc" for benchmark in benchmarks):
        native.require_harness()
        index["alpaca_assets"] = native.prepare_alpaca_assets(bundle / "scorer-assets" / "alpaca")
    scorer_directory = None
    if any(benchmark["name"] in ("ifeval", "ifbench", "ifbench_multiturn") for benchmark in benchmarks):
        native.require_harness()
        scorer_directory = bundle / "scorer-assets" / "nltk"
        index["scorer_assets"] = native.prepare_assets(scorer_directory)
    with native.offline_scorers(scorer_directory):
        for benchmark in benchmarks:
            name = benchmark["name"]
            if name in NATIVE_TASKS:
                loaded[name] = {
                    "documents": native.prepare_native(name),
                    "origin": "pinned-native-task:" + ",".join(NATIVE_TASKS[name]),
                    "revision": native.harness_pin(),
                }
            elif name == "ifbench_multiturn":
                native.validate_ifbench_cases(loaded[name]["cases"])
    for name, content in loaded.items():
        content["content_digest"] = digest(content)
        write_json(bundle / (name + ".json"), content)
        index["benchmarks"][name] = content["content_digest"]
    write_json(bundle / "index.json", index)


def load_bundle(suite: dict, bundle: Path, benchmarks: list[dict]) -> dict:
    index = read_json(bundle / "index.json")
    if index.get("schema_version") != 1 or index.get("suite_digest") != digest(suite):
        raise ValueError("Bundle belongs to a different suite configuration; prepare a new bundle")
    result = {}
    if any(benchmark["name"] == "alpaca_eval_2_lc" for benchmark in benchmarks):
        expected = index.get("alpaca_assets")
        actual = native.asset_hashes(bundle / "scorer-assets" / "alpaca")
        if not expected or not actual or digest(actual) != expected["digest"] or actual != expected["files"]:
            raise ValueError("AlpacaEval scorer-asset integrity check failed; prepare a new bundle")
        result["alpaca_assets"] = expected
    if any(benchmark["name"] in ("ifeval", "ifbench", "ifbench_multiturn") for benchmark in benchmarks):
        expected = index.get("scorer_assets")
        actual = native.asset_hashes(bundle / "scorer-assets" / "nltk")
        if not expected or not actual or digest(actual) != expected["digest"] or actual != expected["files"]:
            raise ValueError("NLTK scorer-asset integrity check failed; prepare a new bundle")
        result["scorer_assets"] = expected
    for benchmark in benchmarks:
        name = benchmark["name"]
        if name not in index["benchmarks"]:
            raise ValueError(f"Benchmark {name} was not prepared")
        content = read_json(bundle / (name + ".json"))
        stored = content.pop("content_digest")
        if digest(content) != stored or stored != index["benchmarks"][name]:
            raise ValueError(f"Dataset integrity check failed: {name}")
        result[name] = content
    return result


class RecordingJudge:
    def __init__(self, client):
        self.client = client
        self.last_output = ""

    def judge(self, instruction: str, evidence: dict) -> str:
        self.last_output = self.client.judge(instruction, evidence)
        return self.last_output


def custom_run(
    name: str, cases: list[dict], mode: str, target, judge, path: Path, max_tokens: int,
    assets: dict | None = None, options: dict | None = None, context: dict | None = None,
) -> dict:
    from scripts.helpfulness import open_benchmarks

    events = read_jsonl(path) if path.exists() else []
    by_id = {row["id"]: row for row in events}
    expected = {case["id"]: digest(case) for case in cases}
    for row in events:
        if row["id"] not in expected or row["case_digest"] != expected[row["id"]]:
            raise ValueError("Saved sample belongs to a different dataset")
        if row["status"] not in ("generated", "failed", "succeeded"):
            raise ValueError("Saved sample has an invalid status")
    for case in cases:
        previous = by_id.get(case["id"], {})
        if previous.get("status") == "succeeded":
            continue
        answers = previous.get("answers", [])
        raw_judge = RecordingJudge(judge) if judge else None
        extension_trace = []
        row = {"id": case["id"], "case": case, "case_digest": digest(case)}
        try:
            if not answers:
                answers = open_benchmarks.generate(name, case, target, max_tokens) if name in open_benchmarks.NAMES else (
                    [case["summary"]] if mode == "validate"
                    else generate_case(name, case, target, max_tokens)
                )
                append_jsonl(path, {**row, "status": "generated", "answers": answers})
            if name in open_benchmarks.NAMES:
                scores, trace = open_benchmarks.score(
                    name, case, answers[-1], judge, assets, options,
                    {**(context or {}), "answers": answers}, extension_trace,
                )
                raw = json.dumps(trace, ensure_ascii=False)
            else:
                scores, raw = score_case(name, case, answers, raw_judge)
            row.update({"status": "succeeded", "answers": answers, "scores": scores, "raw_judge": raw})
        except (ValueError, RuntimeError) as error:
            row.update({
                "status": "failed", "answers": answers, "error": str(error),
                "phase": "judging" if answers else "generation",
                "raw_judge": json.dumps(extension_trace, ensure_ascii=False) if name in open_benchmarks.NAMES
                else raw_judge.last_output if raw_judge else "",
            })
            print(f"{name}/{case['id']}: {row['phase']} failed: {error}", file=sys.stderr)
        append_jsonl(path, row)
        by_id[case["id"]] = row
    if name in open_benchmarks.NAMES:
        return open_benchmarks.summarize(name, list(by_id.values()), len(cases))
    return summarize_custom(name, list(by_id.values()), len(cases), mode)


def write_csv(path: Path, report: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["checkpoint", "benchmark", "protocol", "status", "metric", "value", "denominator"])
        for checkpoint, benchmarks in report["results"].items():
            for name, summary in benchmarks.items():
                for metric, value in summary.get("metrics", {}).items():
                    fields = value if isinstance(value, dict) else {"value": value}
                    writer.writerow([
                        checkpoint, name, summary["protocol"], summary["status"], metric,
                        fields.get("value", fields.get("rate")),
                        fields.get("denominator", summary.get("denominators", {}).get(metric, "")),
                    ])


def run(
    suite: dict, data: dict, output: Path, checkpoints: list[dict], benchmarks: list[dict],
    limit: int | None = None, resume: bool = False,
) -> dict:
    from scripts.helpfulness import eqbench, open_benchmarks, open_models

    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer")
    if any(item["name"] in open_benchmarks.NAMES | EXTERNAL_TASKS for item in benchmarks):
        if suite.get("model_policy") != "open_weights_only":
            raise ValueError("Teacher extensions cannot run without the open-weight model policy")
    model_provenance = open_models.validate_suite(suite, checkpoints, benchmarks)
    for benchmark in benchmarks:
        if benchmark["name"] == "arena_hard_v2":
            references = model_provenance.get("references", {})
            required = {case["reference_model"] for case in data["arena_hard_v2"]["cases"]}
            if required - references.keys():
                raise ValueError("Arena-Hard adapted baselines require explicit open-weight reference provenance")
            for model in required:
                record = references[model]
                cases = [case for case in data["arena_hard_v2"]["cases"] if case["reference_model"] == model]
                if record.get("kind") == "released_answers" and digest(cases) != record["cases_digest"]:
                    raise ValueError("Prepared Arena cases differ from the verified released reference source")
    resolved_targets = {
        checkpoint["name"]: public_endpoint(checkpoint["endpoint"]) for checkpoint in checkpoints
    }
    needs_judge = any(benchmark["name"] in JUDGED for benchmark in benchmarks)
    judge_config = suite.get("judge", {}) if needs_judge else {}
    resolved_judge = public_endpoint(judge_config) if needs_judge else None
    for checkpoint in checkpoints:
        for key in ("revision", "tokenizer_revision", "chat_template_revision"):
            if checkpoint[key].startswith("REPLACE_WITH_"):
                raise ValueError(f"{checkpoint['name']}: verify and replace {key} before running")
        endpoint_key(checkpoint["endpoint"])
        if needs_judge and endpoint_model(checkpoint["endpoint"]) == endpoint_model(judge_config):
            raise ValueError("The judge must be independent of every evaluated model")
        if any(benchmark["name"] == "sycophancy" for benchmark in benchmarks):
            tokenizer = checkpoint["endpoint"].get("tokenizer")
            if not tokenizer or not Path(tokenizer).exists():
                raise ValueError("Sycophancy likelihood scoring requires a runner-local tokenizer path")
    if needs_judge:
        endpoint_key(judge_config)
        if any(benchmark["name"] in ("mt_bench", "alpaca_eval_2_lc") for benchmark in benchmarks) and not endpoint_key(judge_config):
            raise ValueError("Native SDK-based judging requires an explicit judge API-key environment variable")
    calibrations = {}
    if needs_judge:
        from scripts.helpfulness.acceptance import validate_judge_lock

        for benchmark in benchmarks:
            name = benchmark["name"]
            if name not in JUDGED:
                continue
            profile = judge_config.get("calibration_profiles", {}).get(name)
            if not profile and judge_config.get("require_calibration"):
                raise ValueError(f"{name}: a passing frozen judge calibration is required")
            if profile:
                strata = None
                if name == "mt_bench":
                    selected = select_documents(
                        name, data[name]["documents"], benchmark.get("categories", []), limit
                    )
                    strata = sorted({row["category"] for row in selected["mt_bench"]})
                calibrations[name] = validate_judge_lock(
                    Path(profile), name, judge_config, Path.cwd(), strata,
                )
    native_selected = any(benchmark["name"] in NATIVE_TASKS or benchmark["name"] == "ifbench_multiturn" for benchmark in benchmarks)
    if native_selected:
        native.require_harness()
    relevant_packages = {
        "lm-eval", "datasets", "alpaca-eval", "openai", "numpy", "pandas",
        "nltk", "emoji", "syllapy", "setuptools", "langdetect", "immutabledict",
        "spacy", "torch", "transformers", "tokenizers", "huggingface-hub",
        "statsmodels", "scikit-learn", "scipy", "tiktoken",
        "glicko2", "trueskill", "joblib", "python-dotenv", "wordfreq", "requests", "tqdm",
    }
    versions = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"].lower().replace("_", "-") in relevant_packages
    }
    identity = {
        "suite": suite, "targets": resolved_targets, "judge": resolved_judge,
        "datasets": {key: digest(value) for key, value in data.items()},
        "checkpoints": [checkpoint["name"] for checkpoint in checkpoints],
        "benchmarks": [benchmark["name"] for benchmark in benchmarks],
        "limit": limit,
        "harness_pin": native.harness_pin() if native_selected else None,
        "runner_code_digest": digest({
            path.name: path.read_text(encoding="utf-8")
            for path in Path(__file__).parent.glob("*.py")
        }),
        "dependency_versions": versions,
        "judge_calibrations": calibrations,
        "open_model_provenance": model_provenance,
    }
    fingerprint = digest(identity)
    metadata_path = output / "run.json"
    if output.exists():
        if not resume or not metadata_path.exists():
            raise ValueError("Output already exists; use --resume or a new run directory")
        if read_json(metadata_path)["fingerprint"] != fingerprint:
            raise ValueError("Cannot resume: configuration, endpoint identity, dependencies, or data changed")
    else:
        write_json(metadata_path, {
            "fingerprint": fingerprint, "identity": identity,
            "started_at": datetime.now(timezone.utc).isoformat(),
        })
    report = {
        "schema_version": 1, "fingerprint": fingerprint, "status": "running",
        "results": {
            checkpoint["name"]: {
                benchmark["name"]: {
                    "status": "unprocessed", "protocol": PROTOCOLS[benchmark["name"]],
                    "mode": benchmark.get("mode", "generate"),
                }
                for benchmark in benchmarks
            }
            for checkpoint in checkpoints
        },
    }
    write_json(output / "summary.json", report)
    try:
        for checkpoint in checkpoints:
            target_config = checkpoint["endpoint"]
            target = ChatClient(target_config)
            judge = ChatClient(judge_config) if needs_judge else None
            summaries = report["results"].setdefault(checkpoint["name"], {})
            for benchmark in benchmarks:
                name = benchmark["name"]
                mode = benchmark.get("mode", "generate")
                directory = output / checkpoint["name"] / name
                summary_path = directory / "summary.json"
                if resume and summary_path.exists() and read_json(summary_path)["status"] == "succeeded":
                    summaries[name] = read_json(summary_path)
                    continue
                summary = {
                    "protocol": PROTOCOLS[name], "status": "running",
                    "mode": mode, "sample_limit": limit,
                    "categories": benchmark.get("categories", []),
                }
                if name in calibrations:
                    summary["judge_calibration"] = calibrations[name]
                summaries[name] = summary
                write_json(summary_path, summary)
                completed = False
                try:
                    if name in EXTERNAL_TASKS:
                        if limit is not None:
                            raise ValueError("Native EQ-Bench rubric runs do not support --limit; use a declared upstream subset")
                        summary.update(eqbench.run(
                            benchmark, data[name], target_config, judge_config, directory, resume,
                        ))
                    elif name in NATIVE_TASKS:
                        documents = select_documents(
                            name, data[name]["documents"], benchmark.get("categories", []), limit
                        )
                        result = native.run_native(name, documents, target_config, judge_config, directory)
                        write_json(directory / "native_results.json", result)
                        summary.update(summarize_native(
                            name, result, {key: len(rows) for key, rows in documents.items()}
                        ))
                        if name == "alpaca_eval_2_lc":
                            diagnostics = read_json(directory / "alpaca" / "validated_metrics.json")
                            summary["metrics"]["alpaca_eval/raw_win_rate"] = diagnostics["raw_win_rate"]
                            summary["denominators"]["alpaca_eval/raw_win_rate"] = diagnostics["judged_cases"]
                            summary["judged_cases"] = diagnostics["judged_cases"]
                    else:
                        cases = data[name]["cases"][:limit] if limit else data[name]["cases"]
                        summary.update(custom_run(
                            name, cases, mode, target, judge, directory / "samples.jsonl",
                            suite.get("max_tokens", 2048),
                            data[name].get("assets"), benchmark,
                            model_provenance.get("targets", {}).get(checkpoint["name"], {}),
                        ))
                    completed = True
                finally:
                    if not completed:
                        summary["status"] = "incomplete"
                        summary["error"] = "Benchmark did not complete; see runner error output"
                    write_json(summary_path, summary)
                    write_json(output / "summary.json", report)
    finally:
        expected = len(checkpoints) * len(benchmarks)
        summaries = [summary for rows in report["results"].values() for summary in rows.values()]
        report["status"] = (
            "succeeded" if len(summaries) == expected and all(
                summary["status"] == "succeeded" for summary in summaries
            ) else "incomplete"
        )
        write_json(output / "summary.json", report)
        write_csv(output / "summary.csv", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", action="append")
    parser.add_argument("--benchmark", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        suite = load_suite(args.suite)
        benchmarks = chosen(suite["benchmarks"], args.benchmark)
        if args.command == "prepare":
            prepare(suite, args.suite.resolve(), args.bundle, benchmarks)
            print(f"Prepared {len(benchmarks)} benchmark bundles in {args.bundle}")
            return 0
        if args.output is None:
            parser.error("run requires --output")
        data = load_bundle(suite, args.bundle, benchmarks)
        # Dataset acquisition is an explicit prepare step, never an evaluation side effect.
        scorer_directory = args.bundle / "scorer-assets" / "nltk" if "scorer_assets" in data else None
        alpaca_directory = args.bundle / "scorer-assets" / "alpaca" if "alpaca_assets" in data else None
        with (
            native.environment({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "WANDB_MODE": "disabled"}),
            native.offline_scorers(scorer_directory),
            native.offline_alpaca_assets(alpaca_directory),
        ):
            report = run(
                suite, data, args.output, chosen(suite["checkpoints"], args.checkpoint),
                benchmarks, args.limit, args.resume,
            )
        print(f"{report['status']}: {args.output / 'summary.json'}")
        return 0 if report["status"] == "succeeded" else 1
    except (ValueError, RuntimeError, OSError, ImportError, KeyError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
