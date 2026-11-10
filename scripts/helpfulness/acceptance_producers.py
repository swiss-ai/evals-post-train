"""Optional hosted producers; original/likelihood runners can supply artifacts instead."""

import importlib
import tomllib
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path

from scripts.helpfulness import acceptance as checks
from scripts.helpfulness import acceptance_scoring as scoring
from scripts.helpfulness import adapters, native
from scripts.helpfulness.core import (
    ChatClient, endpoint_key, endpoint_model, read_json, render_request as render, write_json,
)


class CapturedClient(ChatClient):
    def __init__(self, config):
        super().__init__(config)
        self.requests = []
        self.responses = []

    def generate(self, content, decoding):
        self.requests.append({"model": self.model, **deepcopy(content), **deepcopy(decoding)})
        answer = super().generate(content, decoding)
        self.responses.append(answer)
        return answer


def endpoint(config: dict, base: Path, pinned: dict) -> dict:
    if "api_key" in config:
        raise ValueError("Use api_key_env, never plaintext credentials")
    if endpoint_model(config) != pinned["model_id"]:
        raise ValueError("Endpoint model ID does not match the predeclared deployment")
    exported = checks.deployment(read_json((base / config["deployment_manifest"]).resolve()))
    if exported != pinned:
        raise ValueError("Operator-exported endpoint deployment does not match the pins")
    return config


def judge_case(name: str, row: dict, client: CapturedClient, decoding: dict) -> None:
    client.requests.clear()
    client.responses.clear()
    if name == "mt_bench":
        metric = importlib.import_module("lm_eval.tasks.mt_bench.metric")
        prompts = metric._load_judge_prompts()
        item = {**row["doc"], "responses": row["answers"]}
        for turn in (1, 2):
            system, user, _ = metric._build_judge_prompt(item, turn, prompts)
            client.generate({"messages": [
                {"role": "system", "content": system}, {"role": "user", "content": user},
            ]}, decoding)
        row["judgments"] = list(client.responses)
    elif name in ("truthfulqa_generation", "simpleqa", "summeval"):
        adapters.score_case(name, row["doc"], row["answers"], client)
        row["judgments"] = list(client.responses)
    else:
        raise ValueError("Use the native weighted annotator for AlpacaEval")
    row["judge_requests"] = list(client.requests)
    scoring.replay_case(name, row)


def judge_alpaca(rows: list[dict], config: dict, output: Path) -> None:
    if not endpoint_key(config):
        raise ValueError("Native SDK judging requires an explicit API-key environment variable")
    instructions = [row["doc"]["instruction"] for row in rows]
    if len(set(instructions)) != len(instructions):
        raise ValueError("Use a separate AlpacaEval calibration plan per candidate system; duplicate prompts are unsafe")
    items = [{
        "instruction": row["doc"]["instruction"], "completion": row["answers"][0],
        "reference_output": row["doc"]["reference_output"], "dataset": row["doc"]["dataset"],
    } for row in rows]
    native.alpaca_aggregation(items, config, output, length_controlled=False)
    annotations = read_json(output / "validated_annotations.json")
    by_instruction = {annotation["instruction"]: annotation for annotation in annotations}
    for row in rows:
        annotation = by_instruction[row["doc"]["instruction"]]
        if annotation["output_1"] != row["doc"]["reference_output"] or annotation["output_2"] != row["answers"][0]:
            raise ValueError("Weighted annotation was not returned in canonical reference/candidate order")
        annotation["index"] = row["annotation"]["index"]
        row.update({
            "annotation": annotation, "candidate_position": 2, "replay_stage": "annotation",
            "judgment_capture": "upstream_SDK_parsed_annotation_not_raw_logprob_replay",
            "status": "succeeded",
        })


def produce(plan_path: Path, endpoint_path: Path, output: Path, command: str) -> dict:
    if output.exists():
        raise ValueError("Candidate artifacts cannot be overwritten")
    if (output.parent / (output.stem + "-annotator")).exists():
        raise ValueError("Annotator cache already exists; use a new artifact name to preserve provenance")
    plan = checks.load_plan(plan_path)
    required = "judge_calibration" if command == "judge" else "reference_regeneration"
    if plan["kind"] != required:
        raise ValueError(f"{command} requires a {required} plan")
    if plan["benchmark"] == "sycophancy":
        raise ValueError("Generate likelihood artifacts with the pinned original/native likelihood runner")
    with endpoint_path.open("rb") as stream:
        config = tomllib.load(stream)
    pinned = endpoint(config["endpoint"], endpoint_path.parent, plan["deployment"])
    if command == "regenerate" and plan["benchmark"] in adapters.JUDGED:
        judge_config = endpoint(config["judge"], endpoint_path.parent, plan["judge"]["deployment"])
        if plan["judge"]["decoding"] != scoring.judge_decoding(
            plan["benchmark"], plan["judge"]["deployment"]["model_id"]
        ):
            raise ValueError("Regeneration scoring-judge decoding differs from the implemented producer")
    else:
        judge_config = pinned
    rows, _ = checks.load_evidence(plan)
    artifact = {
        "schema_version": 1, "status": "incomplete", "acceptance_plan_sha256": plan["sha256"],
        "inputs_sha256": plan["inputs"]["sha256"], "deployment": plan["deployment"],
        "deployment_assurance": plan["deployment_assurance"], "decoding": plan["decoding"],
        "cases": deepcopy(rows),
    }
    if command == "judge":
        artifact["rubric_sha256"] = plan["rubric_sha256"]
    elif "judge" in plan:
        artifact["judge"] = plan["judge"]
    for row in artifact["cases"]:
        row["status"] = "unprocessed"
        row["judgments"] = []
        if command == "regenerate":
            row["answers"] = []
            row.pop("generation_truncated", None)
    write_json(output, artifact)
    target = CapturedClient(pinned)
    judge = CapturedClient(judge_config) if plan["benchmark"] in adapters.JUDGED else None
    with ExitStack() as stack:
        checks.scorer_assets(plan, stack)
        for row in artifact["cases"]:
            if judge is not None:
                judge.requests.clear()
                judge.responses.clear()
            try:
                if command == "regenerate":
                    target.requests.clear()
                    count = 2 if plan["benchmark"] in ("mt_bench", "ifbench_multiturn") else 1
                    if len(row["requests"]) != count:
                        raise ValueError("Freeze one actual generation request template per evaluated turn")
                    for request in row["requests"]:
                        row["answers"].append(target.generate(render(request, row["answers"]), plan["decoding"]))
                    row["generation_requests"] = list(target.requests)
                if plan["benchmark"] == "alpaca_eval_2_lc":
                    row["status"] = "generated"
                elif plan["benchmark"] in adapters.JUDGED:
                    judge_case(
                        plan["benchmark"], row, judge,
                        plan["decoding"] if command == "judge" else plan["judge"]["decoding"],
                    )
                    row["status"] = "succeeded"
                else:
                    scoring.replay_case(plan["benchmark"], row)
                    row["status"] = "succeeded"
            except (ValueError, RuntimeError, KeyError, ImportError) as error:
                row.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
                if judge is not None:
                    row["judgments"] = list(judge.responses)
                    row["judge_requests"] = list(judge.requests)
            write_json(output, artifact)
        if plan["benchmark"] == "alpaca_eval_2_lc" and all(
            row["status"] == "generated" for row in artifact["cases"]
        ):
            try:
                judge_alpaca(artifact["cases"], judge_config, output.parent / (output.stem + "-annotator"))
            except (ValueError, RuntimeError, KeyError, ImportError) as error:
                artifact["error"] = f"{type(error).__name__}: {error}"
    artifact["status"] = (
        "succeeded" if all(row["status"] == "succeeded" for row in artifact["cases"]) else "incomplete"
    )
    checks.reject_secrets(artifact)
    write_json(output, artifact)
    return artifact
