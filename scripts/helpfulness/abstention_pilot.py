"""Bounded partial AbstentionBench pilot, separate from full-corpus preparation."""

import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.helpfulness import abstention_subset, open_benchmarks, teacher_smoke
from scripts.helpfulness.benchmark_pilot import record_sdk_calls, verified_bytes
from scripts.helpfulness.core import append_jsonl, read_json, read_jsonl, write_json
from scripts.helpfulness.open_models import reject_closed


BENCHMARK = "abstention_bench_partial_pilot"
COMPONENTS = (
    "big_bench_known_unknowns", "gsm8k_abstain", "known_unknown_questions",
    "mmlu_history_abstain", "mmlu_math_abstain", "moral_choice", "qasper",
)
SCENARIOS = {
    "answer unknown", "underspecified context", "underspecified intent", "false premise", "subjective",
}
OPTIONS = {"component_scope": "without_gpqa", "judge_max_tokens": 16}


def load_selection(args):
    manifest = json.loads(verified_bytes(args.selection, args.selection_sha256))
    verified_bytes(args.source, manifest["cases_sha256"])
    cases = read_jsonl(args.source)
    normalized = open_benchmarks.normalize("abstention_bench", cases, OPTIONS)
    if (
        manifest["benchmark"] != BENCHMARK or manifest["full_benchmark"] is not False
        or manifest["selection_frozen_before_generation"] is not True
        or normalized != cases or len(cases) != 20
        or len({case["id"] for case in cases}) != 20
        or set(case["dataset"] for case in cases) != set(COMPONENTS)
        or set(case["scenario"] for case in cases) != SCENARIOS
        or Counter(case["should_abstain"] for case in cases) != {False: 10, True: 10}
        or manifest["case_ids"] != [case["id"] for case in cases]
    ):
        raise ValueError("Partial pilot must preserve its frozen twenty cases, seven components and 10/10 labels")
    return cases


def load_samples(args, cases):
    verified_bytes(args.samples, args.samples_sha256)
    rows = read_jsonl(args.samples)
    if [row["case"] for row in rows] != cases:
        raise ValueError("Saved responses differ from the frozen cases")
    for row in rows:
        teacher_smoke.validate_generation(
            row["answer"], row["generation_info"], allow_truncated=True, allow_claim=True,
        )
        expected = "generated_truncated" if row["generation_info"].get("generation_truncated") else "generated"
        if row["status"] != expected:
            raise ValueError("Saved response has inconsistent generation status")
    return rows


class PromptCollector:
    def __init__(self):
        self.messages = []

    def complete(self, messages, **kwargs):
        self.messages.append(messages)
        return "no" if len(self.messages) == 1 else "incorrect"


def judge_messages(samples, assets):
    expected = []
    for sample in samples:
        collector = PromptCollector()
        open_benchmarks.score(
            "abstention_bench", sample["case"], sample["answer"], collector, assets, OPTIONS, {},
        )
        for index, messages in enumerate(collector.messages):
            expected.append({
                "case_id": sample["case"]["id"], "kind": "abstention" if index == 0 else "correctness",
                "messages": messages,
            })
    return expected


def preflight(args, cases):
    assets = open_benchmarks.staged_assets("abstention_bench", args.upstream)
    result = {
        "benchmark": BENCHMARK, "selected_cases": 20, "included_components": list(COMPONENTS),
        "missing_retained_components": sorted(set(abstention_subset.INCLUDED) - set(COMPONENTS)),
        "native_prompt_sha256": assets["sha256"], "full_benchmark": False,
    }
    if args.action in ("preflight-target", "generate"):
        result["native_preflight"] = teacher_smoke.preflight(args.checkpoint)
    else:
        samples = load_samples(args, cases)
        identity = read_json(args.identity)
        reject_closed(identity["public_model"])
        reject_closed(args.model)
        if (
            identity["served_model"] != args.model or identity["base_url"] != args.base_url
            or not identity.get("license") or not identity.get("public_repository_revision")
            or identity["source_samples_sha256"] != args.samples_sha256
        ):
            raise ValueError("Judge identity is not bound to this open model, endpoint and frozen response file")
        result.update(
            judge_identity=identity, judge_messages=judge_messages(samples, assets),
            source_samples_sha256=args.samples_sha256,
        )
    return result, assets


def replay_judgments(samples, judgments, calls, assets, model):
    expected = judge_messages(samples, assets)
    if len(calls) != 2 * len(expected) or len(judgments) != len(samples):
        raise ValueError("Incomplete raw judge evidence")
    responses = []
    for entry, requested, returned in zip(expected, calls[::2], calls[1::2]):
        if (
            requested["phase"] != "requested" or returned["phase"] != "returned"
            or requested["request"]["messages"] != entry["messages"]
            or requested["request"]["temperature"] != 0
            or requested["request"]["max_tokens"] != OPTIONS["judge_max_tokens"]
            or requested["request"]["model"] != model or returned["response"]["model"] != model
        ):
            raise ValueError("Raw judge request differs from the native prompt or frozen decoding")
        choices = returned["response"]["choices"]
        if len(choices) != 1 or choices[0]["finish_reason"] != "stop":
            raise ValueError("Judge response is missing or truncated")
        responses.append(choices[0]["message"]["content"])

    class Replay:
        def __init__(self):
            self.index = 0

        def complete(self, messages, **kwargs):
            if messages != expected[self.index]["messages"]:
                raise ValueError("Native judge prompt order changed")
            response = responses[self.index]
            self.index += 1
            return response

    client = Replay()
    for sample, row in zip(samples, judgments):
        scores, trace = open_benchmarks.score(
            "abstention_bench", sample["case"], sample["answer"], client, assets, OPTIONS, {},
        )
        if row != {"case": sample["case"], "answer": sample["answer"], "status": "succeeded",
                   "scores": scores, "trace": trace}:
            raise ValueError("Saved verdict differs from the actual native-prompt judge responses")
    return summary(judgments)


def summary(rows):
    result = open_benchmarks.summarize("abstention_bench", rows, 20, OPTIONS)
    tp = sum(row["case"]["should_abstain"] and row["scores"]["abstained"] for row in rows)
    fp = sum(not row["case"]["should_abstain"] and row["scores"]["abstained"] for row in rows)
    fn = sum(row["case"]["should_abstain"] and not row["scores"]["abstained"] for row in rows)
    tn = len(rows) - tp - fp - fn
    answerable = [row for row in rows if not row["case"]["should_abstain"]]
    result.update(
        benchmark=BENCHMARK, full_benchmark=False, pilot_only=True,
        missing_retained_components=sorted(set(abstention_subset.INCLUDED) - set(COMPONENTS)),
        pooled_diagnostics={
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "answerable_accuracy": {
                "value": sum(row["scores"]["correct"] for row in answerable) / len(answerable) if answerable else None,
                "denominator": len(answerable),
            },
        },
    )
    return result


def run(args):
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    generating = args.action == "generate"
    report = {
        "benchmark": BENCHMARK, "status": "running", "phase": "preflight",
        "full_benchmark": False, "published_protocol_equivalence": False,
        "independent_judge_validation": False, "model": args.model, "base_url": args.base_url,
        "selection_sha256": args.selection_sha256, "completed_cases": 0,
        "model_weights_downloaded": False, "target_regenerated": generating,
        "target_max_tokens": 2048, "judge_max_tokens": OPTIONS["judge_max_tokens"],
        "truncation_policy": "retain_nonempty_final_reply_and_flag; block_scoring_empty_reply",
    }
    write_json(args.output / "report.json", report)
    try:
        cases = load_selection(args)
        checked, assets = preflight(args, cases)
        report["preflight"] = {key: value for key, value in checked.items() if key != "judge_messages"}
        teacher_smoke.wait_for_server(args.base_url, args.model, args.ready_timeout)
        report["phase"] = "generation" if generating else "judging"
        write_json(args.output / "report.json", report)
        if generating:
            model = teacher_smoke.native_model(args)
            for index, case in enumerate(cases):
                row = teacher_smoke.generate_native(
                    model, case, [{"role": "user", "content": case["question"]}], BENCHMARK, index,
                    {"temperature": 0, "max_gen_toks": 2048, "until": [], "seed": 1234},
                )
                append_jsonl(args.output / "generation-evidence.jsonl", row)
                teacher_smoke.validate_generation(
                    row["answer"], row["generation_info"], allow_truncated=True, allow_claim=True,
                )
                row["status"] = "generated_truncated" if row["generation_info"].get("generation_truncated") else "generated"
                append_jsonl(args.output / "samples.jsonl", row)
                report["completed_cases"] += 1
                write_json(args.output / "report.json", report)
        else:
            from openai import OpenAI

            samples = load_samples(args, cases)
            expected = checked["judge_messages"]
            with (
                record_sdk_calls(args.output / "judge-calls.jsonl", args.base_url, args.model, expected),
                OpenAI(base_url=args.base_url, api_key="allocation-local-no-auth", max_retries=0, timeout=90) as sdk,
            ):
                class Judge:
                    def complete(self, messages, *, temperature, max_tokens):
                        response = sdk.chat.completions.create(
                            model=args.model, messages=messages, temperature=temperature, max_tokens=max_tokens,
                        )
                        if len(response.choices) != 1 or response.choices[0].finish_reason != "stop":
                            raise ValueError("Judge did not finish its binary verdict")
                        return response.choices[0].message.content

                judge = Judge()
                for sample in samples:
                    scores, trace = open_benchmarks.score(
                        "abstention_bench", sample["case"], sample["answer"], judge, assets, OPTIONS, {},
                    )
                    append_jsonl(args.output / "judgments.jsonl", {
                        "case": sample["case"], "answer": sample["answer"], "status": "succeeded",
                        "scores": scores, "trace": trace,
                    })
                    report["completed_cases"] += 1
                    write_json(args.output / "report.json", report)
            report["summary"] = replay_judgments(
                samples, read_jsonl(args.output / "judgments.jsonl"),
                read_jsonl(args.output / "judge-calls.jsonl"), assets, args.model,
            )
        report.update(status="succeeded", phase="completed")
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
    if args.action in ("preflight-target", "generate"):
        if args.checkpoint is None:
            parser.error("Target actions require --checkpoint")
    elif not all((args.samples, args.samples_sha256, args.identity)):
        parser.error("Judge actions require frozen samples, SHA256 and judge identity")
    if args.action.startswith("preflight"):
        if args.output.exists():
            raise FileExistsError("Preflight output already exists")
        value, _ = preflight(args, load_selection(args))
        write_json(args.output, value)
    else:
        run(args)


if __name__ == "__main__":
    main()
