"""Bounded native-format pilots with separate checkpoint generation and scoring."""

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import importlib
import json
from pathlib import Path
from unittest.mock import patch

from scripts.helpfulness import native, teacher_smoke
from scripts.helpfulness.core import append_jsonl, digest, read_json, read_jsonl, write_json
from scripts.helpfulness.open_models import reject_closed


ALPACA_SOURCE_SHA256 = "83db546b872ddebee8965fd05fa48461ee3c32bc695c62fb57f2d214ff741ec4"
ALPACA_FEATURES_SHA256 = "97aeec3f1c7de6dee6fd31fe66f1702623f4614c3efe1c6fc5f4927cd5fd674d"
ALPACA_PROMPT_SHA256 = "784227e6dc2832fc08c43d2c8ea3a308e7523780187a1aaad2f85e30bac85f62"
SELECTION_SEED = "alpaca-eval2-pilot20-v1:"


def verified_bytes(path: Path, expected: str) -> bytes:
    value = path.read_bytes()
    if hashlib.sha256(value).hexdigest() != expected:
        raise ValueError(f"Source checksum mismatch: {path.name}")
    return value


def load_selection(args) -> list[dict]:
    selection = json.loads(verified_bytes(args.selection, args.selection_sha256))
    if (
        selection.get("benchmark") != "alpaca_eval_2_lc"
        or selection.get("source_sha256") != ALPACA_SOURCE_SHA256
    ):
        raise ValueError("This pilot currently supports only the pinned AlpacaEval 2 source")
    documents = json.loads(verified_bytes(args.source, ALPACA_SOURCE_SHA256))
    by_id = {digest(row): row for row in documents}
    source_indices = {digest(row): index for index, row in enumerate(documents)}
    if len(by_id) != len(documents):
        raise ValueError("Duplicate original documents")
    ranked = sorted(by_id, key=lambda key: hashlib.sha256((SELECTION_SEED + key).encode()).hexdigest())
    ids = selection["case_ids"]
    if len(ids) != args.expected_cases or ids != ranked[:args.expected_cases]:
        raise ValueError("Selection differs from the frozen deterministic pilot")
    cases = []
    for key in ids:
        doc = by_id[key]
        if any(not isinstance(doc.get(field), str) or not doc[field].strip()
               for field in ("instruction", "output")):
            raise ValueError("AlpacaEval requires the original instruction and released reference")
        dataset = doc.get("dataset", "alpaca_eval")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("AlpacaEval dataset identity must be text")
        cases.append({
            "id": key, "question": doc["instruction"],
            "reference_answer": doc["output"], "dataset": dataset, "source_index": source_indices[key],
        })
    return cases


def load_samples(args, cases: list[dict], *, allow_truncated: bool = False) -> list[dict]:
    verified_bytes(args.samples, args.samples_sha256)
    samples = read_jsonl(args.samples)
    if [row["case"] for row in samples] != cases:
        raise ValueError("Saved generations do not match every frozen case in order")
    for row in samples:
        truncated = bool(row["generation_info"].get("generation_truncated"))
        if row.get("status") != ("generated_truncated" if truncated else "generated"):
            raise ValueError("Saved generation is not validated")
        teacher_smoke.validate_generation(
            row["answer"], row["generation_info"], allow_truncated=allow_truncated,
        )
    return samples


def resume_samples(args, cases: list[dict], preflight: dict) -> tuple[list[dict], dict | None]:
    if args.resume_samples is None:
        return [], None
    verified_bytes(args.resume_samples, args.resume_samples_sha256)
    source = json.loads(verified_bytes(args.resume_report, args.resume_report_sha256))
    samples = read_jsonl(args.resume_samples)
    if (
        source["status"] not in ("failed", "succeeded")
        or source["benchmark"] != "alpaca_eval_2_lc"
        or source["source_sha256"] != ALPACA_SOURCE_SHA256
        or source["selection_sha256"] != args.selection_sha256
        or source["model"] != args.model or source["preflight"] != preflight
        or source["completed_cases"] != len(samples)
        or not 0 < len(samples) <= len(cases)
        or [row["case"] for row in samples] != cases[:len(samples)]
    ):
        raise ValueError("Resume evidence does not match this checkpoint and frozen case prefix")
    for row in samples:
        if row.get("status") != "generated":
            raise ValueError("Resume accepts only previously validated, nontruncated answers")
        teacher_smoke.validate_generation(row["answer"], row["generation_info"])
    return samples, source["decoding"]


def judge_preflight(args, cases: list[dict]) -> tuple[list[dict], dict]:
    samples = load_samples(args, cases, allow_truncated=args.allow_truncated)
    identity = read_json(args.identity)
    if identity["served_model"] != args.model or identity["base_url"] != args.base_url:
        raise ValueError("Judge identity is not bound to this endpoint and model")
    reject_closed(identity["public_model"])
    reject_closed(args.model)
    if not identity.get("license") or not identity.get("public_repository_revision"):
        raise ValueError("Judge requires staged public revision and license evidence")
    verified_bytes(args.scorer_assets / "df_gamed.csv", ALPACA_FEATURES_SHA256)
    native.require_harness()
    importlib.import_module("alpaca_eval.metrics.glm_winrate")

    return samples, identity


def judge_messages(samples: list[dict]) -> list[dict]:
    import pandas as pd
    from alpaca_eval import constants, utils

    template = verified_bytes(
        constants.EVALUATORS_CONFIG_DIR / "alpaca_eval_clf_gpt4_turbo" / "alpaca_eval_clf.txt",
        ALPACA_PROMPT_SHA256,
    ).decode("utf-8")
    rendered = []
    for row in samples:
        for candidate_position in (1, 2):
            outputs = [row["answer"], row["case"]["reference_answer"]]
            if candidate_position == 2:
                outputs.reverse()
            prompts, _ = utils.make_prompts(pd.DataFrame([{
                "instruction": row["case"]["question"],
                "output_1": outputs[0], "output_2": outputs[1],
            }]), template=template, batch_size=1)
            rendered.append({
                "case_id": row["case"]["id"], "candidate_position": candidate_position,
                "messages": utils.prompt_to_chatml(prompts[0]),
            })
    return rendered


@contextmanager
def record_sdk_calls(path: Path, base_url: str, model: str, expected_messages: list[dict] | None = None):
    from openai.resources.chat.completions import Completions as ChatCompletions
    from openai.resources.completions import Completions

    allowed = {digest(row["messages"]) for row in expected_messages} if expected_messages is not None else None
    with ExitStack() as stack:
        for resource in (ChatCompletions, Completions):
            original = resource.create

            def create(self, *args, _original=original, **kwargs):
                if (
                    args or str(self._client.base_url).rstrip("/") != base_url
                    or kwargs.get("model") != model
                ):
                    raise ValueError("Native scorer attempted an undeclared endpoint or model")
                if allowed is not None and digest(kwargs.get("messages")) not in allowed:
                    raise ValueError("Native scorer messages differ from the tokenized preflight")
                append_jsonl(path, {"phase": "requested", "request": kwargs})
                response = _original(self, **kwargs)
                append_jsonl(path, {"phase": "returned", "response": response.model_dump(mode="json")})
                return response

            stack.enter_context(patch.object(resource, "create", create))
        yield


def run(args) -> dict:
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    generating = args.action == "generate"
    report = {
        "status": "running", "phase": "preflight", "benchmark": "alpaca_eval_2_lc",
        "scope": "bounded_checkpoint_pilot" if generating else "bounded_frozen_answer_scoring",
        "published_protocol_equivalence": False, "full_benchmark": False,
        "independent_judge_validation": False, "model_weights_downloaded": False,
        "selected_cases": args.expected_cases, "completed_cases": 0,
        "selection_sha256": args.selection_sha256, "source_sha256": ALPACA_SOURCE_SHA256,
        "model": args.model, "base_url": args.base_url, "target_regenerated": False,
        "truncation_policy": "include_and_flag" if args.allow_truncated else "reject",
        "truncated_cases": 0,
    }
    write_json(args.output / "report.json", report)
    try:
        cases = load_selection(args)
        if generating:
            report["preflight"] = teacher_smoke.preflight(args.checkpoint)
            report["decoding"] = {
                "temperature": 0, "max_gen_toks": args.max_tokens, "until": [],
                "reasoning_effort": "low", "prefill": "after_think", "seed": 1234,
            }
            report["generation_limit_adaptation"] = (
                f"Bounded {args.max_tokens}-token generation budget, not the pinned task's 16384-token ceiling"
            )
            saved, previous_decoding = resume_samples(args, cases, report["preflight"])
            report.update(resumed_cases=len(saved), new_generations=0, completed_cases=len(saved))
            report["decoding_segments"] = [{
                "case_ids": [case["id"] for case in cases[len(saved):]], "decoding": report["decoding"],
            }]
            if saved:
                report["resume_source"] = {
                    "samples_sha256": args.resume_samples_sha256, "report_sha256": args.resume_report_sha256,
                }
                report["decoding_segments"].insert(0, {
                    "case_ids": [row["case"]["id"] for row in saved], "decoding": previous_decoding,
                })
                for row in saved:
                    append_jsonl(args.output / "samples.jsonl", row)
        else:
            samples, identity = judge_preflight(args, cases)
            expected_messages = judge_messages(samples)
            report.update(
                judge_identity=identity, source_samples_sha256=args.samples_sha256,
                features_sha256=ALPACA_FEATURES_SHA256,
                protocol="adapted-swissai-alpacaeval2-length-controlled-replacement-judge",
                reasoning_available=False,
                reasoning_note="Native m/M logprob classification emits no written rationale",
                judge_prompt_sha256=ALPACA_PROMPT_SHA256,
                truncated_cases=sum(bool(row["generation_info"].get("generation_truncated")) for row in samples),
            )
        report["phase"] = "server_readiness"
        write_json(args.output / "report.json", report)
        teacher_smoke.wait_for_server(args.base_url, args.model, args.ready_timeout)
        report["phase"] = "generation" if generating else "scoring"
        write_json(args.output / "report.json", report)
        if generating:
            model = teacher_smoke.native_model(args)
            for index, case in enumerate(cases[len(saved):], len(saved)):
                record = teacher_smoke.generate_native(
                    model, case, [{"role": "user", "content": case["question"]}],
                    "alpaca_eval_2_lc_pilot", index,
                    {"temperature": 0, "max_gen_toks": args.max_tokens, "until": []},
                )
                append_jsonl(args.output / "generation-evidence.jsonl", record)
                teacher_smoke.validate_generation(
                    record["answer"], record["generation_info"], allow_truncated=args.allow_truncated,
                )
                truncated = bool(record["generation_info"].get("generation_truncated"))
                record["status"] = "generated_truncated" if truncated else "generated"
                append_jsonl(args.output / "samples.jsonl", record)
                report.update(
                    phase="generation", completed_cases=index + 1, target_regenerated=True,
                    new_generations=report["new_generations"] + 1,
                    truncated_cases=report["truncated_cases"] + int(truncated),
                )
                write_json(args.output / "report.json", report)
        else:
            items = [{
                "instruction": row["case"]["question"], "completion": row["answer"],
                "reference_output": row["case"]["reference_answer"], "dataset": row["case"]["dataset"],
                "index": row["case"]["source_index"],
            } for row in samples]
            config = {
                "base_url": args.base_url, "model": args.model,
                "api_key_env": "PILOT_JUDGE_API_KEY", "requests_per_minute": 0,
            }
            with (
                native.environment({
                    "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "WANDB_MODE": "disabled",
                    "PILOT_JUDGE_API_KEY": "allocation-local-no-auth",
                }),
                native.offline_alpaca_assets(args.scorer_assets),
                record_sdk_calls(args.output / "judge-calls.jsonl", args.base_url, args.model, expected_messages),
            ):
                native.alpaca_aggregation(items, config, args.output / "alpaca")
            report.update(
                phase="scoring", completed_cases=len(samples),
                metrics=read_json(args.output / "alpaca" / "validated_metrics.json"),
            )
        report.update(status="succeeded", phase="completed")
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(args.output / "report.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight-target", "generate", "preflight-judge", "judge"))
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, choices=range(1, 21), default=20)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-tokens", type=int, choices=range(1, 8193), default=8192)
    parser.add_argument("--max-model-len", type=int, choices=range(1, 16385), default=16384)
    parser.add_argument("--ready-timeout", type=int, choices=range(1, 1801), default=1200)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--samples-sha256")
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--scorer-assets", type=Path)
    parser.add_argument("--allow-truncated", action="store_true",
                        help="Include token-limit replies with generated_truncated status; other failures still reject")
    parser.add_argument("--resume-samples", type=Path)
    parser.add_argument("--resume-samples-sha256")
    parser.add_argument("--resume-report", type=Path)
    parser.add_argument("--resume-report-sha256")
    args = parser.parse_args(argv)
    target = args.action in ("preflight-target", "generate")
    if target and args.checkpoint is None:
        parser.error("Target actions require --checkpoint")
    resume = (args.resume_samples, args.resume_samples_sha256, args.resume_report, args.resume_report_sha256)
    if any(resume) and (not target or not all(resume)):
        parser.error("Resume requires target action, samples/report files and both SHA256 digests")
    if not target and not all((args.samples, args.samples_sha256, args.identity, args.scorer_assets)):
        parser.error("Judge actions require --samples, --samples-sha256, --identity and --scorer-assets")
    if args.action.startswith("preflight"):
        if args.output.exists():
            raise FileExistsError("Preflight output already exists")
        cases = load_selection(args)
        result = {"selected_cases": len(cases), "inference_performed": False}
        if target:
            result["native_preflight"] = teacher_smoke.preflight(args.checkpoint)
            saved, _ = resume_samples(args, cases, result["native_preflight"])
            result["resumed_cases"] = len(saved)
        else:
            samples, result["judge_identity"] = judge_preflight(args, cases)
            result["judge_messages"] = judge_messages(samples)
            result["judge_prompt_sha256"] = ALPACA_PROMPT_SHA256
        write_json(args.output, result)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
