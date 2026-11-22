"""Bounded native EQ-Bench 3 scenarios, separated checkpoint generation and judging."""

import argparse
import ast
import hashlib
import importlib
import json
import logging
import math
import os
from pathlib import Path
import re
import statistics
import sys
import typing
from types import SimpleNamespace

from scripts.helpfulness import arena_pilot, teacher_smoke
from scripts.helpfulness.benchmark_pilot import record_sdk_calls, verified_bytes
from scripts.helpfulness.core import append_jsonl, read_json, read_jsonl, write_json
from scripts.helpfulness.open_models import reject_closed


REVISION = "affc5a000bbdb965b807e09ca7875f8530fa8f56"
PROTOCOL = "adapted-eqbench3-native-scenario-rubric-pilot-v1"
ALLOCATION = {"roleplay_or_direct": 11, "message_drafting": 1, "analysis": 8}


def native(upstream, manifest):
    upstream = Path(upstream).resolve()
    record = read_json(manifest)
    if record["revision"] != REVISION or record["license"] != "MIT":
        raise ValueError("EQ-Bench pilot requires the reviewed MIT source pin")
    for name, expected in record["files_sha256"].items():
        path = (upstream / name).resolve()
        if not path.is_relative_to(upstream):
            raise ValueError("Upstream manifest path escapes source directory")
        verified_bytes(path, expected)
    sys.path.insert(0, str(upstream))
    C = importlib.import_module("utils.constants")
    Task = importlib.import_module("core.conversation").ScenarioTask
    if not Path(sys.modules["utils.constants"].__file__).resolve().is_relative_to(upstream):
        raise ValueError("Native EQ-Bench modules loaded from a different source")
    for key, value in vars(C).copy().items():
        if key.endswith("_FILE") and isinstance(value, str):
            setattr(C, key, str(upstream / value))
    # Execute only the two pinned native functions, avoiding unused Elo imports.
    tree = ast.parse((upstream / "core/benchmark.py").read_text(encoding="utf-8"))
    names = {"parse_scenario_prompts", "calculate_final_rubric_score"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(nodes) != 2:
        raise ValueError("Pinned native scenario/aggregation functions are missing")
    namespace = {**vars(typing), "C": C, "logging": logging, "re": re, "statistics": statistics}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(upstream / "core/benchmark.py"), "exec"), namespace)
    return SimpleNamespace(C=C, Task=Task, **{name: namespace[name] for name in names})


def category(sid, C):
    return ("analysis" if sid in C.ANALYSIS_SCENARIO_IDS else
            "message_drafting" if sid in C.MESSAGE_DRAFTING_SCENARIO_IDS else "roleplay_or_direct")


def select(n):
    scenarios = n.parse_scenario_prompts(n.C.STANDARD_SCENARIO_PROMPTS_FILE)
    result = []
    for group, count in ALLOCATION.items():
        ids = [sid for sid in scenarios if category(sid, n.C) == group]
        ids.sort(key=lambda sid: hashlib.sha256(("eqbench3-pilot20-v1:" + sid).encode()).hexdigest())
        if len(ids) < count:
            raise ValueError("Insufficient native scenarios for pilot allocation")
        result.extend({"id": sid, "category": group, "prompts": scenarios[sid]} for sid in ids[:count])
    return result


def criteria(n, sid):
    file = n.C.ANALYSIS_RUBRIC_CRITERIA_FILE if sid in n.C.ANALYSIS_SCENARIO_IDS else n.C.STANDARD_RUBRIC_CRITERIA_FILE
    return [line.strip() for line in Path(file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]


def task(n, case, model):
    analysis = case["id"] in n.C.ANALYSIS_SCENARIO_IDS
    master = (n.C.ANALYSIS_MASTER_PROMPT_FILE if analysis else
              n.C.MESSAGE_DRAFTING_MASTER_PROMPT_FILE if case["id"] in n.C.MESSAGE_DRAFTING_SCENARIO_IDS
              else n.C.STANDARD_MASTER_PROMPT_FILE)
    return n.Task(
        case["id"], case["prompts"],
        None if analysis else Path(n.C.STANDARD_DEBRIEF_PROMPT_FILE).read_text(encoding="utf-8"),
        1, model, Path(master).read_text(encoding="utf-8"),
    )


def rubric(n, saved):
    t = n.Task.from_dict(saved)
    file = n.C.ANALYSIS_RUBRIC_PROMPT_FILE if t.scenario_id in n.C.ANALYSIS_SCENARIO_IDS else n.C.STANDARD_RUBRIC_PROMPT_FILE
    output = {"chain_of_thought_reasoning": "detailed chain of thought reasoning about the coming scoring decisions"}
    output.update({key: 0 for key in criteria(n, t.scenario_id)})
    prompt = t.prepare_rubric_prompt_text(
        Path(file).read_text(encoding="utf-8"),
        json.dumps(output, indent=2).replace(": 0", ": 0-20"), True,
    )
    if prompt is None:
        raise ValueError("Native rubric preparation failed: " + str(t.error))
    return prompt.replace("*", "").replace("#", "")


def preflight(args):
    n = native(args.upstream, args.manifest)
    verified_bytes(args.selection, args.selection_sha256)
    cases = read_json(args.selection)["cases"]
    if cases != select(n):
        raise ValueError("Frozen EQ-Bench scenario selection differs from native source")
    return n, cases


class NativeClient:
    def __init__(self, model, args, case):
        self.model, self.args, self.case = model, args, case
        self.model_failure = False
        self.fatal_error = None

    def generate(self, model, messages, temperature, max_tokens, min_p):
        decoding = {"temperature": temperature, "max_gen_toks": self.args.max_tokens,
                    "seed": 1234, "until": []}
        if min_p is not None:
            decoding["min_p"] = min_p
        try:
            row = teacher_smoke.generate_native(
                self.model, self.case, messages, "eqbench3_pilot", 0, decoding,
            )
            row["messages"] = json.loads(json.dumps(messages))
            row["requested_native_max_tokens"] = max_tokens
            append_jsonl(self.args.output / "generation-evidence.jsonl", row)
            if arena_pilot.outcome_status(row, True) == "model_no_final_reply":
                self.model_failure = True
                raise ValueError("Explicit native model_no_final_reply")
            return row["answer"]
        except Exception as error:
            if not self.model_failure:
                self.fatal_error = error
            raise


def generate(args):
    n, cases = preflight(args)
    args.output.mkdir(exist_ok=False, parents=True)
    report = {"protocol": PROTOCOL, "status": "running", "selected_cases": 20, "recorded_cases": 0,
              "generated_cases": 0, "elo_computed": False, "automatic_scoring_ready": False,
              "max_tokens": args.max_tokens, "full_benchmark": False}
    write_json(args.output / "report.json", report)
    try:
        teacher_smoke.wait_for_server(args.base_url, args.model, 1200)
        model = teacher_smoke.native_model(args)
        for case in cases:
            t = task(n, case, args.model)
            client = NativeClient(model, args, case)
            t.run_scenario({"test": client}, None, None, args.model)
            if t.status != "error":
                t.run_debrief({"test": client}, None, None, args.model)
            saved = t.to_dict()
            status = "model_no_final_reply" if client.model_failure else "generated"
            if client.fatal_error is not None:
                raise client.fatal_error
            if t.status == "error" and not client.model_failure:
                raise RuntimeError("Native scenario failure: " + str(t.error))
            prompt = rubric(n, saved) if status == "generated" else None
            append_jsonl(args.output / "samples.jsonl", {
                "case": case, "task": saved, "status": status, "judge_prompt": prompt,
                "criteria": criteria(n, case["id"]),
            })
            report["recorded_cases"] += 1
            report["generated_cases"] += status == "generated"
            write_json(args.output / "report.json", report)
        report["status"] = "succeeded" if report["generated_cases"] == 20 else "incomplete"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(args.output / "report.json", report)


def request(sample, model, max_tokens=2048):
    return {"model": model, "messages": [{"role": "user", "content": sample["judge_prompt"]}],
            "temperature": 0, "max_tokens": max_tokens, "seed": 1234}


def judge_preflight(args, n, cases):
    verified_bytes(args.samples, args.samples_sha256)
    samples = read_jsonl(args.samples)
    if [s["case"] for s in samples] != cases:
        raise ValueError("EQ-Bench samples do not match selection")
    for sample in samples:
        if sample["status"] not in ("generated", "model_no_final_reply"):
            raise ValueError("Unknown checkpoint outcome")
        if sample["criteria"] != criteria(n, sample["case"]["id"]):
            raise ValueError("Sample criteria differ from native source")
        if sample["status"] == "generated" and sample["judge_prompt"] != rubric(n, sample["task"]):
            raise ValueError("Frozen judge prompt differs from native task replay")
    identity = read_json(args.identity)
    reject_closed(args.model)
    reject_closed(identity["public_model"])
    if (
        identity["served_model"] != args.model or identity["base_url"] != args.base_url
        or identity["source_samples_sha256"] != args.samples_sha256
        or identity["selection_sha256"] != args.selection_sha256
        or identity["benchmark"] != "eqbench3" or identity["license"] != "Apache-2.0"
        or not identity["public_repository_revision"]
    ):
        raise ValueError("Judge requires bound open-model, source, selection and licensing evidence")
    plans = [{"case_id": s["case"]["id"], "messages": request(s, args.model)["messages"]}
             for s in samples if s["status"] == "generated"]
    if not plans:
        raise ValueError("No complete EQ-Bench scenarios to judge")
    return samples, identity, plans


def parse_scores(n, sample, response):
    choice = response["choices"]
    if len(choice) != 1 or choice[0]["finish_reason"] != "stop":
        raise ValueError("Native EQ-Bench judge output missing or truncated")
    scores = n.Task._parse_rubric_scores(choice[0]["message"]["content"])
    if not scores or set(scores) != set(sample["criteria"]) or any(
        type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 20
        for value in scores.values()
    ):
        raise ValueError("Judge must supply every exact native criterion with a finite 0–20 score")
    return scores


def judgment(n, sample, response):
    row = {"case": sample["case"], "status": "failed", "scores": {}}
    if sample["status"] != "generated":
        row["error"] = "model_no_final_reply"
        return row
    row["response"] = response
    try:
        scores = parse_scores(n, sample, response)
        t = {**sample["task"], "status": "rubric_scored", "rubric_scores": scores}
        scalar, error = n.calculate_final_rubric_score({"scenario_tasks": {"1": {sample["case"]["id"]: t}}})
        if error:
            raise ValueError(error)
        row.update(status="succeeded", scores=scores, native_rubric_score=scalar)
    except (ValueError, KeyError, TypeError) as error:
        row["error"] = str(error)
    return row


def summarize(n, samples, rows):
    tasks = {s["case"]["id"]: {**s["task"], "status": "rubric_scored", "rubric_scores": r["scores"]}
             for s, r in zip(samples, rows) if r["status"] == "succeeded"}
    score, error = n.calculate_final_rubric_score({"scenario_tasks": {"1": tasks}})
    return {
        "protocol": PROTOCOL, "status": "succeeded" if len(tasks) == 20 else "incomplete",
        "selected_cases": 20, "scored_cases": len(tasks), "native_rubric_score": score,
        "aggregation_error": error, "scale": [0, 20], "elo_computed": False,
        "automatic_scoring_ready": False,
    }


def replay(n, samples, calls, model):
    valid = [s for s in samples if s["status"] == "generated"]
    if len(calls) != 2 * len(valid):
        raise ValueError("Incomplete EQ-Bench raw judge evidence")
    responses = iter(calls[1::2])
    for sample, sent, returned in zip(valid, calls[::2], calls[1::2]):
        if (
            sent.get("phase") != "requested" or returned.get("phase") != "returned"
            or sent.get("request") != request(sample, model)
            or returned.get("response", {}).get("model") != model
        ):
            raise ValueError("Raw EQ-Bench request or response differs from declared protocol")
    rows = [judgment(n, s, next(responses)["response"] if s["status"] == "generated" else None)
            for s in samples]
    return rows, summarize(n, samples, rows)


def judge(args):
    n, cases = preflight(args)
    samples, identity, plans = judge_preflight(args, n, cases)
    from openai import OpenAI

    args.output.mkdir(exist_ok=False, parents=True)
    write_json(args.output / "report.json", {"status": "running", "judge_identity": identity,
                                           "automatic_scoring_ready": False})
    teacher_smoke.wait_for_server(args.base_url, args.model, 1200)
    with (
        record_sdk_calls(args.output / "judge-calls.jsonl", args.base_url, args.model, plans),
        OpenAI(base_url=args.base_url, api_key="allocation-local", max_retries=0, timeout=180) as sdk,
    ):
        rows = []
        for sample in samples:
            response = (sdk.chat.completions.create(**request(sample, args.model)).model_dump(mode="json")
                        if sample["status"] == "generated" else None)
            row = judgment(n, sample, response)
            rows.append(row)
            append_jsonl(args.output / "judgments.jsonl", row)
            write_json(args.output / "report.json", summarize(n, samples, rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "generate", "judge"))
    for key in ("upstream", "manifest", "selection", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--samples-sha256")
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=16384)
    args = parser.parse_args()
    if args.action == "preflight":
        n, cases = preflight(args)
        value = {"protocol": PROTOCOL, "selected_cases": 20,
                 "scenario_turns": sum(len(c["prompts"]) for c in cases),
                 "debrief_calls": sum(c["category"] != "analysis" for c in cases)}
        if args.checkpoint:
            value["native_preflight"] = teacher_smoke.preflight(args.checkpoint)
        else:
            _, identity, plans = judge_preflight(args, n, cases)
            value.update(judge_identity=identity, judge_messages=plans)
        write_json(args.output, value)
    elif args.action == "generate":
        generate(args)
    else:
        judge(args)


if __name__ == "__main__":
    main()
