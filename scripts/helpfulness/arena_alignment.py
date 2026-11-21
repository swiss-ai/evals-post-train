"""Both-order Arena judge diagnostics against separately frozen review intervals."""

import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.helpfulness import arena_pilot, open_benchmarks, teacher_smoke
from scripts.helpfulness.benchmark_pilot import record_sdk_calls, verified_bytes
from scripts.helpfulness.core import append_jsonl, digest, read_json, read_jsonl, write_json
from scripts.helpfulness.open_models import reject_closed


PROTOCOL = "adapted-arena-hard-v2-both-order-alignment-v1"
COMPACT_PROTOCOL = "adapted-arena-hard-v2-compact-both-order-alignment-v2"
LIMIT = 2048
COMPACT_LIMIT = 512
SPLITS = {"pilot_review": 20, "fresh_constructed_controls": 20}


def load_inputs(args):
    verified_bytes(args.cases, args.cases_sha256)
    verified_bytes(args.oracle, args.oracle_sha256)
    cases, oracle = read_jsonl(args.cases), read_json(args.oracle)
    ids = [row["case"]["id"] for row in cases]
    if (
        len(set(ids)) != 40 or Counter(row["split"] for row in cases) != SPLITS
        or [label["case_id"] for label in oracle["cases"]] != ids
        or oracle.get("frozen_before_judge_outputs") is not True
        or oracle.get("gate") != {"minimum_matches_per_split": 17, "all_orders_valid": True,
                                  "production_scoring_ready": False}
    ):
        raise ValueError("Arena alignment requires forty frozen cases, both twenty-case splits and the fixed gate")
    for row, label in zip(cases, oracle["cases"]):
        case = row["case"]
        if case.get("category") not in ("hard_prompt", "creative_writing", "coding", "math"):
            raise ValueError("Unknown Arena category")
        for value in (case.get("question"), case.get("reference_answer")):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Alignment requires complete nonempty question and both answers")
        failed_model = row.get("status") == "model_no_final_reply"
        if not isinstance(row.get("answer"), str) or (not row["answer"].strip() and not failed_model):
            raise ValueError("Empty replies require an explicit native model-failure outcome")
        if failed_model:
            if row["split"] != "pilot_review" or arena_pilot.outcome_status(row, True) != "model_no_final_reply":
                raise ValueError("Only verified native checkpoint failures may skip judging")
        allowed = label.get("allowed_margins")
        if (
            not isinstance(allowed, list) or (not allowed and not failed_model) or len(allowed) != len(set(allowed))
            or (failed_model and allowed)
            or any(type(value) is not int or value not in range(-2, 3) for value in allowed)
            or not isinstance(label.get("basis"), str) or not label["basis"].strip()
        ):
            raise ValueError("Every review requires explicit allowed candidate-relative margins and a basis")
    return cases, oracle


def comparison_profile(args):
    rubric = getattr(args, "compact_rubric", None)
    expected = getattr(args, "compact_rubric_sha256", None)
    if (rubric is None) != (expected is None):
        raise ValueError("Compact comparisons require both a rubric file and its frozen checksum")
    if rubric is None:
        return None, PROTOCOL, LIMIT
    verified_bytes(rubric, expected)
    text = Path(rubric).read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Compact comparison rubric must not be empty")
    return text, COMPACT_PROTOCOL, COMPACT_LIMIT


def plans(cases, assets, rubric=None):
    entries = [
        {"case_id": row["case"]["id"], **entry}
        for row in cases if row.get("status") != "model_no_final_reply"
        for entry in open_benchmarks.arena_messages(row["case"], row["answer"], assets)
    ]
    if rubric is not None:
        categories = {row["case"]["id"]: row["case"]["category"] for row in cases}
        for entry in entries:
            entry["messages"][0]["content"] = rubric + "\nCategory: " + categories[entry["case_id"]]
    return entries


def request(entry, model, limit=LIMIT):
    return {"model": model, "messages": entry["messages"], "temperature": 0,
            "max_tokens": limit, "seed": 1234}


def results(cases, oracle, responses, protocol=PROTOCOL):
    expected_orders = 2 * sum(row.get("status") != "model_no_final_reply" for row in cases)
    if len(responses) != expected_orders:
        raise ValueError("Alignment requires every actual judge response; native model failures must not be judged")
    rows = []
    offset = 0
    for index, (case, expected) in enumerate(zip(cases, oracle["cases"])):
        failed_model = case.get("status") == "model_no_final_reply"
        sample = {**case, "status": "model_no_final_reply" if failed_model else "generated"}
        count = 0 if failed_model else 2
        row = arena_pilot.judgment(sample, responses[offset:offset + count])
        offset += count
        row["split"] = case["split"]
        row["review"] = expected
        row["alignment_match"] = (
            row["status"] == "succeeded"
            and all(order["margin"] in expected["allowed_margins"] for order in row["orders"])
        )
        rows.append(row)
    split_results = {}
    for split, selected in SPLITS.items():
        group = [row for row in rows if row["split"] == split]
        valid = [row for row in group if row["status"] == "succeeded"]
        split_results[split] = {
            "selected_cases": selected, "valid_cases": len(valid),
            "complete_case_matches": sum(row["alignment_match"] for row in group),
            "position_exact_matches": sum(row["orders"][0]["margin"] == row["orders"][1]["margin"] for row in valid),
            "position_exact_denominator": len(valid),
            "failed_case_ids": [row["case"]["id"] for row in group if not row["alignment_match"]],
        }
    valid_orders = sum(order["status"] == "succeeded" for row in rows for order in row["orders"])
    gate = valid_orders == expected_orders and all(value["complete_case_matches"] >= 17 for value in split_results.values())
    pilot = [row for row in rows if row["split"] == "pilot_review"]
    pilot_metrics = arena_pilot.summarize(pilot)
    if protocol != PROTOCOL:
        pilot_metrics["protocol"] = protocol
    summary = {
        "protocol": protocol, "split_results": split_results,
        "valid_orders": valid_orders, "selected_orders": 80, "expected_judge_orders": expected_orders,
        "omitted_model_failure_orders": 80 - expected_orders, "alignment_gate_passed": gate,
        "checkpoint_reply_coverage_complete": expected_orders == 80,
        "automatic_scoring_ready": False, "independent_human_gold": False,
        "interpretation": "Agreement with frozen AI review intervals and constructive controls, not general judge accuracy",
        "pilot_metrics": pilot_metrics,
    }
    return rows, summary


def replay(args, cases, oracle, calls, assets):
    rubric, protocol, limit = comparison_profile(args)
    expected = plans(cases, assets, rubric)
    if len(calls) != 2 * len(expected):
        raise ValueError("Incomplete alignment request/response evidence")
    responses = []
    for entry, requested, returned in zip(expected, calls[::2], calls[1::2]):
        if (
            requested.get("phase") != "requested" or returned.get("phase") != "returned"
            or requested.get("request") != request(entry, args.model, limit)
            or returned.get("response", {}).get("model") != args.model
        ):
            raise ValueError("Alignment raw messages/decoding/model binding changed")
        responses.append(returned["response"])
    return results(cases, oracle, responses, protocol)


def preflight(args):
    cases, oracle = load_inputs(args)
    rubric, protocol, limit = comparison_profile(args)
    assets = open_benchmarks.staged_assets("arena_hard_v2", args.upstream)
    identity = read_json(args.identity)
    reject_closed(args.model)
    reject_closed(identity["public_model"])
    if (
        identity["served_model"] != args.model or identity["base_url"] != args.base_url
        or identity["cases_sha256"] != args.cases_sha256 or identity["oracle_sha256"] != args.oracle_sha256
        or identity.get("license") != "Apache-2.0" or not identity.get("public_repository_revision")
        or identity.get("protocol") != protocol
        or identity.get("compact_rubric_sha256") != getattr(args, "compact_rubric_sha256", None)
    ):
        raise ValueError("Alignment judge identity is not bound to frozen inputs and protocol")
    return cases, oracle, assets, {
        "protocol": protocol, "cases_sha256": args.cases_sha256, "oracle_sha256": args.oracle_sha256,
        "judge_identity": identity, "judge_messages": plans(cases, assets, rubric), "max_tokens": limit,
        "upstream_sha256": assets["sha256"], "rubric_identity": digest(assets["sha256"]),
    }


def run(args):
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    report = {"status": "running", "protocol": comparison_profile(args)[1], "completed_calls": 0,
              "automatic_scoring_ready": False, "target_generation_calls": 0}
    write_json(args.output / "report.json", report)
    try:
        cases, oracle, assets, checked = preflight(args)
        report["preflight"] = {key: value for key, value in checked.items() if key != "judge_messages"}
        teacher_smoke.wait_for_server(args.base_url, args.model, 1200)
        from openai import OpenAI

        with (
            record_sdk_calls(args.output / "judge-calls.jsonl", args.base_url, args.model,
                             checked["judge_messages"]),
            OpenAI(base_url=args.base_url, api_key="allocation-local-no-auth",
                   max_retries=0, timeout=180) as sdk,
        ):
            for entry in checked["judge_messages"]:
                sdk.chat.completions.create(**request(entry, args.model, checked["max_tokens"]))
                report["completed_calls"] += 1
                write_json(args.output / "report.json", report)
        rows, summary = replay(args, cases, oracle, read_jsonl(args.output / "judge-calls.jsonl"), assets)
        for row in rows:
            append_jsonl(args.output / "judgments.jsonl", row)
        report.update(summary=summary, status="succeeded" if summary["alignment_gate_passed"] else "rejected")
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(args.output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run"))
    for name in ("cases", "oracle", "identity", "upstream", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("cases-sha256", "oracle-sha256", "model"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--compact-rubric", type=Path)
    parser.add_argument("--compact-rubric-sha256")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    args = parser.parse_args(argv)
    if args.action == "preflight":
        if args.output.exists():
            raise FileExistsError("Preflight already exists")
        write_json(args.output, preflight(args)[3])
    else:
        report = run(args)
        if report["status"] != "succeeded":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
