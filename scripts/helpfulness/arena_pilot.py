"""Frozen twenty-case Arena-Hard v2 diagnostic with independent answer-order replay."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from scripts.helpfulness import open_benchmarks, teacher_smoke
from scripts.helpfulness.abstention_pilot import outcome_status
from scripts.helpfulness.benchmark_pilot import record_sdk_calls, verified_bytes
from scripts.helpfulness.core import append_jsonl, read_json, read_jsonl, write_json
from scripts.helpfulness.open_models import reject_closed


BENCHMARK = "arena_hard_v2"
PROTOCOL = "adapted-arena-hard-v2-open-reference-native-rubric-pilot-v1"
SOURCE_SHA256 = "d96d8bcb30be531aff9af9fbd4fd8822e247e02f0523e73a1b385bbddcb202cc"
SEED = "arena-hard-v2-pilot20-v1:"
COUNTS = {"hard_prompt": 13, "creative_writing": 7}
JUDGE_TOKENS = 2048
PREFERENCES = {"A>>B": -2, "A>B": -1, "A=B": 0, "B>A": 1, "B>>A": 2}


def select(rows):
    normalized = open_benchmarks.normalize(BENCHMARK, rows, {})
    if Counter(row["category"] for row in normalized) != {"hard_prompt": 500, "creative_writing": 250}:
        raise ValueError("Expected the pinned 500/250-category Arena-Hard v2 release")
    if {row["reference_model"] for row in normalized} != {"Qwen/Qwen3-32B"}:
        raise ValueError("Pilot requires the explicitly attributed open Qwen3-32B reference")
    selected = []
    for category, count in COUNTS.items():
        ranked = sorted(
            (hashlib.sha256((SEED + row["id"]).encode()).hexdigest(), row["id"], row)
            for row in normalized if row["category"] == category
        )
        selected.extend(row for _, _, row in ranked[:count])
    return selected


def load_selection(args):
    manifest = json.loads(verified_bytes(args.selection, args.selection_sha256))
    rows = read_jsonl(args.source)
    verified_bytes(args.source, SOURCE_SHA256)
    cases = select(rows)
    if (
        manifest.get("benchmark") != BENCHMARK
        or manifest.get("source_sha256") != SOURCE_SHA256
        or manifest.get("selection_frozen_before_generation") is not True
        or manifest.get("case_ids") != [case["id"] for case in cases]
        or manifest.get("categories") != COUNTS
        or manifest.get("full_benchmark") is not False
    ):
        raise ValueError("Arena pilot differs from its frozen deterministic selection")
    return cases


def load_samples(args, cases):
    verified_bytes(args.samples, args.samples_sha256)
    samples = read_jsonl(args.samples)
    if [row.get("case") for row in samples] != cases:
        raise ValueError("Saved Arena responses differ from the frozen cases")
    for row in samples:
        if row.get("status") != outcome_status(row, True):
            raise ValueError("Arena generation status does not match its native evidence")
    return samples


def judge_messages(samples, assets):
    return [
        {"case_id": row["case"]["id"], **entry}
        for row in samples if row["status"] != "model_no_final_reply"
        for entry in open_benchmarks.arena_messages(row["case"], row["answer"], assets)
    ]


def parse_preference(raw, candidate_position):
    if candidate_position not in ("A", "B") or not isinstance(raw, str):
        raise ValueError("Arena preference requires text and a declared answer position")
    matches = re.findall(r"\[\[([AB><=]+)\]\]", raw)
    terminal = re.search(r"\[\[([AB><=]+)\]\]\s*[.!]?\s*$", raw)
    if (
        terminal is None or terminal[1] not in PREFERENCES
        or not raw[:terminal.start()].strip()
        or set(matches) != {terminal[1]}
    ):
        raise ValueError("Arena needs a reason and one unambiguous terminal five-way preference")
    return PREFERENCES[terminal[1]] * (1 if candidate_position == "B" else -1)


def battle_scores(margins):
    if len(margins) != 2 or any(type(value) is not int or value not in range(-2, 3) for value in margins):
        raise ValueError("Arena requires both five-way answer-order verdicts")
    weights = [3 if abs(value) == 2 else 1 for value in margins]
    return {
        "win_credit": sum(weight * (1 if value > 0 else 0.5 if value == 0 else 0)
                          for value, weight in zip(margins, weights)),
        "battle_weight": sum(weights),
    }


def judgment(sample, responses, preference_parser=None):
    preference_parser = preference_parser or parse_preference
    if sample["status"] == "model_no_final_reply":
        if responses:
            raise ValueError("A failed checkpoint reply must not be judged")
        return {"case": sample["case"], "answer": sample["answer"], "status": "failed",
                "error": "model_no_final_reply", "scores": {}, "orders": []}
    if len(responses) != 2:
        raise ValueError("Each valid checkpoint reply requires both answer orders")
    orders = []
    for position, response in zip(("B", "A"), responses):
        item = {"candidate_position": position, "response": response}
        try:
            choices = response["choices"]
            if len(choices) != 1 or choices[0]["finish_reason"] != "stop":
                raise ValueError("Arena judge response is missing or truncated")
            item["margin"] = preference_parser(choices[0]["message"]["content"], position)
            item["status"] = "succeeded"
        except (ValueError, KeyError, TypeError) as error:
            item.update(status="failed", error=str(error))
        orders.append(item)
    valid = all(item["status"] == "succeeded" for item in orders)
    return {
        "case": sample["case"], "answer": sample["answer"],
        "status": "succeeded" if valid else "failed",
        "scores": battle_scores([item["margin"] for item in orders]) if valid else {},
        "orders": orders,
    }


def summarize(rows):
    result = open_benchmarks.summarize(BENCHMARK, rows, 20)
    valid = [row for row in rows if row["status"] == "succeeded"]
    result.update(
        protocol=PROTOCOL, full_benchmark=False, independent_judge_validation=False,
        position_exact_agreement=sum(row["orders"][0]["margin"] == row["orders"][1]["margin"] for row in valid),
        position_direction_agreement=sum(
            (row["orders"][0]["margin"] > 0) - (row["orders"][0]["margin"] < 0)
            == (row["orders"][1]["margin"] > 0) - (row["orders"][1]["margin"] < 0) for row in valid
        ),
        position_agreement_denominator=len(valid),
        source_population_categories={"hard_prompt": 500, "creative_writing": 250},
        pilot_categories=COUNTS,
        invalid_cases_retained_in_selected_denominator=True,
    )
    return result


def replay(samples, calls, assets, model):
    expected = judge_messages(samples, assets)
    if len(calls) != 2 * len(expected):
        raise ValueError("Incomplete Arena raw judge evidence")
    responses = []
    for plan, request, returned in zip(expected, calls[::2], calls[1::2]):
        payload = request.get("request", {})
        response = returned.get("response", {})
        if (
            request.get("phase") != "requested" or returned.get("phase") != "returned"
            or payload != {"model": model, "messages": plan["messages"],
                           "temperature": 0, "max_tokens": JUDGE_TOKENS, "seed": 1234}
            or response.get("model") != model
        ):
            raise ValueError("Arena raw request/model differs from the frozen protocol")
        responses.append(response)
    rows = []
    offset = 0
    for sample in samples:
        count = 0 if sample["status"] == "model_no_final_reply" else 2
        rows.append(judgment(sample, responses[offset:offset + count]))
        offset += count
    return rows, summarize(rows)


def preflight(args, cases):
    assets = open_benchmarks.staged_assets(BENCHMARK, args.upstream)
    result = {
        "benchmark": BENCHMARK, "protocol": PROTOCOL, "selected_cases": 20,
        "source_sha256": SOURCE_SHA256, "selection_sha256": args.selection_sha256,
        "upstream": {key: value for key, value in assets.items() if key != "files"},
        "full_benchmark": False, "headline_bradley_terry_score_computed": False,
    }
    if args.action in ("generate", "preflight-target"):
        result["native_preflight"] = teacher_smoke.preflight(args.checkpoint)
    else:
        samples = load_samples(args, cases)
        identity = read_json(args.identity)
        reject_closed(args.model)
        reject_closed(identity["public_model"])
        if (
            identity["served_model"] != args.model or identity["base_url"] != args.base_url
            or identity["source_samples_sha256"] != args.samples_sha256
            or identity.get("license") != "Apache-2.0"
            or not identity.get("public_repository_revision")
        ):
            raise ValueError("Arena judge requires a bound open-model revision, license and frozen samples")
        result.update(judge_identity=identity, judge_messages=judge_messages(samples, assets))
        if not result["judge_messages"]:
            raise ValueError("No valid final checkpoint replies to judge")
    return result, assets


def run(args):
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    report = {
        "benchmark": BENCHMARK, "protocol": PROTOCOL, "status": "running", "phase": "preflight",
        "model": args.model, "base_url": args.base_url, "selected_cases": 20,
        "completed_cases": 0, "recorded_cases": 0, "model_failure_cases": 0,
        "full_benchmark": False, "automatic_scoring_ready": False,
        "target_max_tokens": args.max_tokens, "selection_sha256": args.selection_sha256,
        "model_weights_downloaded": False,
    }
    write_json(args.output / "report.json", report)
    try:
        cases = load_selection(args)
        checked, assets = preflight(args, cases)
        report["preflight"] = {key: value for key, value in checked.items() if key != "judge_messages"}
        teacher_smoke.wait_for_server(args.base_url, args.model, args.ready_timeout)
        if args.action == "generate":
            report["phase"] = "generation"
            model = teacher_smoke.native_model(args)
            for index, case in enumerate(cases):
                row = teacher_smoke.generate_native(
                    model, case, [{"role": "user", "content": case["question"]}], BENCHMARK, index,
                    {"temperature": 0, "max_gen_toks": args.max_tokens, "until": [], "seed": 1234},
                )
                append_jsonl(args.output / "generation-evidence.jsonl", row)
                row["status"] = outcome_status(row, True)
                append_jsonl(args.output / "samples.jsonl", row)
                report["recorded_cases"] += 1
                report["completed_cases"] += int(row["status"] != "model_no_final_reply")
                report["model_failure_cases"] += int(row["status"] == "model_no_final_reply")
                write_json(args.output / "report.json", report)
        else:
            from openai import OpenAI

            report["phase"] = "judging"
            samples = load_samples(args, cases)
            with (
                record_sdk_calls(args.output / "judge-calls.jsonl", args.base_url, args.model,
                                 checked["judge_messages"]),
                OpenAI(base_url=args.base_url, api_key="allocation-local-no-auth",
                       max_retries=0, timeout=180) as sdk,
            ):
                for sample in samples:
                    responses = []
                    if sample["status"] != "model_no_final_reply":
                        for plan in open_benchmarks.arena_messages(sample["case"], sample["answer"], assets):
                            response = sdk.chat.completions.create(
                                model=args.model, messages=plan["messages"],
                                temperature=0, max_tokens=JUDGE_TOKENS, seed=1234,
                            )
                            responses.append(response.model_dump(mode="json"))
                    row = judgment(sample, responses)
                    append_jsonl(args.output / "judgments.jsonl", row)
                    report["recorded_cases"] += 1
                    report["completed_cases"] += int(row["status"] == "succeeded")
                    write_json(args.output / "report.json", report)
            rows, report["summary"] = replay(
                samples, read_jsonl(args.output / "judge-calls.jsonl"), assets, args.model,
            )
            if rows != read_jsonl(args.output / "judgments.jsonl"):
                raise ValueError("Stored Arena verdicts differ from raw replay")
        report.update(phase="completed", status="succeeded" if report["completed_cases"] == 20 else "incomplete")
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(args.output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight-target", "generate", "preflight-judge", "judge"))
    for name in ("source", "selection", "upstream", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-tokens", type=int, choices=(2048,), default=2048)
    parser.add_argument("--max-model-len", type=int, choices=(16384,), default=16384)
    parser.add_argument("--ready-timeout", type=int, default=1200)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--samples-sha256")
    parser.add_argument("--identity", type=Path)
    args = parser.parse_args(argv)
    if args.action in ("generate", "preflight-target"):
        if args.checkpoint is None:
            parser.error("Target actions require --checkpoint")
    elif not all((args.samples, args.samples_sha256, args.identity)):
        parser.error("Judge actions require frozen samples, SHA256 and identity")
    if args.action.startswith("preflight"):
        if args.output.exists():
            raise FileExistsError("Preflight already exists")
        value, _ = preflight(args, load_selection(args))
        write_json(args.output, value)
    else:
        run(args)


if __name__ == "__main__":
    main()
