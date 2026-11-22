"""Frozen WildBench v2 checkpoint generation and source-bound rubric diagnostics."""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import re

from scripts.helpfulness import open_benchmarks as native, teacher_smoke
from scripts.helpfulness.abstention_pilot import outcome_status
from scripts.helpfulness.benchmark_pilot import record_sdk_calls, verified_bytes
from scripts.helpfulness.core import append_jsonl, read_json, read_jsonl, write_json
from scripts.helpfulness.open_models import reject_closed

BENCHMARK = "wildbench_v2"
PROTOCOL = "adapted-wildbench-v2-native-rubric-pilot-v1"
SOURCE_SHA256 = "3f905245ab4ecd9af4dbdb8c833e5138c0da09bd566c6ca96002cca4133c5a4d"
DATA_REVISION = "26c49eb39d7d5ce2099b0bbafed5a88dcce954ec"
SEED = "wildbench-v2-pilot20-v1:"
GROUPS = tuple(dict.fromkeys(native.WILD_GROUPS.values()))
ASSESSMENT_PATTERN = r"^[^\s]{1,32}(?: [^\s]{1,32}){0,23}$"


def select(rows):
    cases = native.normalize(BENCHMARK, rows, {})
    if len(cases) != 1024 or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Expected the pinned 1024 unique WildBench v2 test cases")
    selected = []
    for group in GROUPS:
        pool = [c for c in cases if native.WILD_GROUPS[c["category"]] == group]
        ranked = sorted(pool, key=lambda c: (hashlib.sha256((SEED + c["id"]).encode()).hexdigest(), c["id"]))
        if len(ranked) < 4:
            raise ValueError("WildBench task group has fewer than four cases")
        selected.extend(ranked[:4])
    return selected


def load_selection(args):
    verified_bytes(args.source, SOURCE_SHA256)
    selection = json.loads(verified_bytes(args.selection, args.selection_sha256))
    cases = select(read_jsonl(args.source))
    if (
        selection.get("benchmark") != BENCHMARK or selection.get("config") != "v2"
        or selection.get("data_revision") != DATA_REVISION
        or selection.get("source_sha256") != SOURCE_SHA256
        or selection.get("case_ids") != [c["id"] for c in cases]
        or selection.get("frozen_before_generation") is not True
        or selection.get("full_benchmark") is not False
    ):
        raise ValueError("WildBench source or frozen selection differs")
    return cases


def load_samples(args, cases):
    rows = read_jsonl(args.samples)
    verified_bytes(args.samples, args.samples_sha256)
    if [r["case"] for r in rows] != cases:
        raise ValueError("WildBench checkpoint samples differ from frozen selection")
    for row in rows:
        if row["status"] != outcome_status(row, True):
            raise ValueError("WildBench generation status differs from native evidence")
        row.update(evaluation_id="pilot-" + row["case"]["id"], split="pilot")
    return rows


def request(sample, assets, model):
    return {
        "model": model, "temperature": 0, "seed": 1234, "max_tokens": 2048,
        "messages": [
            {"role": "system", "content": sample["calibration_prompt"]},
            {"role": "user", "content": native.wildbench_prompt(sample["case"], sample["answer"], assets)},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "wildbench_native_rubric", "strict": True, "schema": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "strengths": {"type": "string", "pattern": ASSESSMENT_PATTERN},
                    "weaknesses": {"type": "string", "pattern": ASSESSMENT_PATTERN},
                    "score": {"type": "number", "minimum": 1, "maximum": 10},
                }, "required": ["strengths", "weaknesses", "score"],
            },
        }},
    }


def deployed_metadata(identity, tokenizer):
    checkpoint = Path(identity["checkpoint_directory"]).resolve()
    metadata = identity["local_metadata_sha256"]
    canonical = identity["canonical_tokenizer"]
    if (set(metadata) != {"config.json", "model.safetensors.index.json"}
        or canonical["public_revision"] != identity["public_repository_revision"]
        or set(canonical["files"]) != {"config.json", "tokenizer_config.json", "chat_template.jinja", "tokenizer.json"}):
        raise ValueError("WildBench judge metadata identity is incomplete")
    for name, expected in metadata.items():
        verified_bytes(checkpoint / name, expected)
    for name, record in canonical["files"].items():
        path = (tokenizer / name).resolve()
        if not path.is_relative_to(tokenizer.resolve()):
            raise ValueError("Judge tokenizer metadata escapes staged directory")
        if len(verified_bytes(path, record["sha256"])) != record["bytes"]:
            raise ValueError("Judge tokenizer byte count differs")
    return {"local_metadata_sha256": metadata, "canonical_tokenizer_verified": True,
            "full_deployed_weight_contents_verified": False}


def judge_preflight(args, cases, assets):
    samples = load_samples(args, cases)
    if args.controls:
        verified_bytes(args.controls, args.controls_sha256)
        for control in read_jsonl(args.controls):
            normalized = native.normalize(BENCHMARK, [control["source_document"]], {})[0]
            if (control["case"] != normalized or control["split"] not in ("development", "transfer")
                or control["evaluation_id"] != control["case"]["id"] or control["status"] != "generated"
                or not isinstance(control["answer"], str) or not control["answer"].strip()):
                raise ValueError("Invalid explicitly constructed WildBench control")
            samples.append(control)
    if len({s["evaluation_id"] for s in samples}) != len(samples):
        raise ValueError("Duplicate WildBench evaluation identity")
    prompt = verified_bytes(args.calibration_prompt, args.calibration_prompt_sha256).decode("utf-8")
    if not prompt.strip():
        raise ValueError("WildBench calibration prompt is empty")
    identity = read_json(args.identity)
    reject_closed(args.model)
    reject_closed(identity["public_model"])
    if (
        identity["served_model"] != args.model or identity["base_url"] != args.base_url
        or identity["benchmark"] != BENCHMARK or not identity["public_repository_revision"]
        or identity["source_samples_sha256"] != args.samples_sha256
        or identity["selection_sha256"] != args.selection_sha256
        or identity.get("controls_sha256") != args.controls_sha256
        or identity["calibration_prompt_sha256"] != args.calibration_prompt_sha256
        or identity["structured_outputs_backend"] != "xgrammar"
    ):
        raise ValueError("WildBench judge identity is not bound to source, prompt and structured backend")
    for sample in samples:
        sample["calibration_prompt"] = prompt
    plans = [{"case_id": s["case"]["id"], "messages": request(s, assets, args.model)["messages"]}
             for s in samples if s["status"] != "model_no_final_reply"]
    return samples, identity, plans


def judgment(sample, response, assets):
    row = {key: sample[key] for key in ("case", "evaluation_id", "split")}
    row.update(status="failed", scores={})
    if sample["status"] == "model_no_final_reply":
        row["error"] = "model_no_final_reply"
        return row
    row["response"] = response
    try:
        choices = response["choices"]
        if len(choices) != 1 or choices[0]["finish_reason"] != "stop":
            raise ValueError("WildBench judge output is missing or truncated")
        text = choices[0]["message"]["content"]
        data = json.loads(text)
        if (
            not isinstance(data, dict) or set(data) != {"strengths", "weaknesses", "score"}
            or any(not isinstance(data[k], str) or not re.fullmatch(ASSESSMENT_PATTERN, data[k])
                   for k in ("strengths", "weaknesses"))
            or type(data["score"]) not in (int, float)
        ):
            raise ValueError("WildBench compact output violates exact numeric JSON or finite assessment bounds")

        class Returned:
            def complete(self, messages, **kwargs):
                if messages != [{"role": "user", "content": native.wildbench_prompt(
                    sample["case"], sample["answer"], assets)}]:
                    raise ValueError("WildBench native prompt changed during scoring replay")
                return text

        scores, _ = native.score(BENCHMARK, sample["case"], sample["answer"], Returned(), assets,
                                 {"judge_max_tokens": 2048}, {})
        row.update(status="succeeded", scores=scores, assessment=data)
    except (ValueError, KeyError, TypeError) as error:
        row["error"] = str(error)
    return row


def summarize(rows):
    pilot = [r for r in rows if r["split"] == "pilot"]
    report = native.summarize(BENCHMARK, pilot, 20)
    report.update(protocol=PROTOCOL, full_benchmark=False, automatic_scoring_ready=False,
                  independent_human_accuracy_measured=False, judged_items=len(rows),
                  valid_items=sum(r["status"] == "succeeded" for r in rows),
                  data_config="v2", data_revision=DATA_REVISION, pilot_primary_group_counts=4)
    return report


def replay(samples, calls, assets, model, *, partial=False):
    planned = [s for s in samples if s["status"] != "model_no_final_reply"]
    pairs, pending = divmod(len(calls), 2)
    if pairs > len(planned) or (pending and pairs == len(planned)):
        raise ValueError("Extra WildBench raw calls")
    if not partial and (pending or pairs != len(planned)):
        raise ValueError("Incomplete WildBench raw calls")
    for sample, asked, returned in zip(planned, calls[::2], calls[1::2]):
        if (asked.get("phase") != "requested" or returned.get("phase") != "returned"
            or asked.get("request") != request(sample, assets, model)
            or returned.get("response", {}).get("model") != model):
            raise ValueError("WildBench raw request or response model differs")
    if pending and (calls[-1].get("phase") != "requested"
                    or calls[-1].get("request") != request(planned[pairs], assets, model)):
        raise ValueError("WildBench pending request differs")
    rows, offset = [], 0
    for sample in samples:
        if sample["status"] != "model_no_final_reply":
            if offset == pairs:
                break
            response = calls[2 * offset + 1]["response"]
            offset += 1
        else:
            response = None
        rows.append(judgment(sample, response, assets))
    report = summarize(rows)
    report["status"] = "succeeded" if len(rows) == len(samples) and report["valid_items"] == len(samples) else "incomplete"
    if partial:
        report.update(raw_returned_calls=pairs, unreturned_request=bool(pending),
                      unattempted_calls=len(planned) - pairs - pending,
                      complete_raw_evidence=not pending and pairs == len(planned))
    return rows, report


def run(args):
    cases = load_selection(args)
    assets = native.staged_assets(BENCHMARK, args.upstream)
    generating = args.action in ("generate", "preflight-target")
    checked = {"benchmark": BENCHMARK, "protocol": PROTOCOL, "selected_cases": 20,
               "upstream": {k: v for k, v in assets.items() if k != "files"}}
    if generating:
        checked["native_preflight"] = teacher_smoke.preflight(args.checkpoint)
    else:
        samples, identity, plans = judge_preflight(args, cases, assets)
        checked.update(judge_identity=identity, judge_messages=plans)
        checked["deployed_metadata"] = deployed_metadata(identity, args.identity.parent / "tokenizer")
    if args.action.startswith("preflight"):
        write_json(args.output, checked)
        return checked
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "benchmark": BENCHMARK, "preflight": checked, "recorded_cases": 0}
    write_json(args.output / "report.json", report)
    try:
        if generating:
            decoding = {"temperature": 0, "max_gen_toks": 2048, "until": [],
                        "reasoning_effort": "low", "prefill": "after_think", "seed": 1234}
            report.update(decoding=decoding, selected_cases=20, full_benchmark=False,
                          generation_limit_adaptation=True, automatic_scoring_ready=False,
                          completed_cases=0, model_failure_cases=0, truncated_cases=0)
            teacher_smoke.wait_for_server(args.base_url, args.model, args.ready_timeout)
            model = teacher_smoke.native_model(args)
            for index, case in enumerate(cases):
                row = teacher_smoke.generate_native(model, case, case["messages"], BENCHMARK, index, decoding)
                append_jsonl(args.output / "generation-evidence.jsonl", row)
                row["status"] = outcome_status(row, True)
                append_jsonl(args.output / "samples.jsonl", row)
                report["recorded_cases"] += 1
                report["completed_cases"] += row["status"] != "model_no_final_reply"
                report["model_failure_cases"] += row["status"] == "model_no_final_reply"
                report["truncated_cases"] += row["status"] == "generated_truncated"
                write_json(args.output / "report.json", report)
        else:
            from openai import OpenAI
            rows = []
            with (record_sdk_calls(args.output / "judge-calls.jsonl", args.base_url, args.model, plans),
                  OpenAI(base_url=args.base_url, api_key="allocation-local-no-auth",
                         max_retries=0, timeout=180) as client):
                for sample in samples:
                    response = None
                    if sample["status"] != "model_no_final_reply":
                        response = client.chat.completions.create(
                            **request(sample, assets, args.model)).model_dump(mode="json")
                    row = judgment(sample, response, assets)
                    rows.append(row)
                    append_jsonl(args.output / "judgments.jsonl", row)
                    report = summarize(rows)
                    report["status"] = "running"
                    write_json(args.output / "report.json", report)
            rebuilt, report = replay(samples, read_jsonl(args.output / "judge-calls.jsonl"), assets, args.model)
            if rebuilt != rows:
                raise ValueError("WildBench raw replay differs from saved judgments")
        if generating:
            report["status"] = "succeeded" if report["completed_cases"] == len(cases) else "incomplete"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        logging.exception("WildBench pilot failed")
        raise
    finally:
        write_json(args.output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "judge", "preflight-target", "preflight-judge"))
    for key in ("source", "selection", "upstream", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-model-len", type=int, choices=(16384,), default=16384)
    parser.add_argument("--max-tokens", type=int, choices=(2048,), default=2048)
    parser.add_argument("--ready-timeout", type=int, default=1200)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--samples-sha256")
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--controls-sha256")
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--calibration-prompt", type=Path)
    parser.add_argument("--calibration-prompt-sha256")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
