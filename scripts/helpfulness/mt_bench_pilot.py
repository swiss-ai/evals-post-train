"""Split, native-format two-turn MT-Bench non-code checkpoint pilot."""

import argparse
from copy import deepcopy
import hashlib
import importlib
import math
from pathlib import Path

from scripts.helpfulness import acceptance_scoring, native, teacher_smoke
from scripts.helpfulness.benchmark_pilot import record_sdk_calls, verified_bytes
from scripts.helpfulness.core import append_jsonl, digest, read_json, read_jsonl, write_json
from scripts.helpfulness.open_models import reject_closed


SOURCE_SHA256 = "119565adbab82227089cefdb44c8d7e2cf04dc0a0ec233634c82e7d4e2a944f7"
REFERENCES_SHA256 = "f957a5bc977badb66885ec970e6cd08527845780313f0995764260e5777b9b3f"
CATEGORIES = ("writing", "roleplay", "reasoning", "math", "extraction", "stem", "humanities")
SEED = "mt-bench-noncode-pilot20-v1:"
NATIVE_HASHES = {
    "metric.py": "2e43d87c64049ef24e47fd7f80b049eb6a9b644fd315e67c9ee5b319ceef050f",
    "task.py": "70ec939f5ebcfcc9b7a15140a5f3568ad2ad808c2fb3eb904afe3d886484c9ad",
    "data/judge_prompts.jsonl": "fd283293406d024f44c174b094ef48031d0687a4682fd3a56b29b138f80281b6",
    "data/mt_bench/question.jsonl": SOURCE_SHA256,
}


def select_documents(documents: list[dict]) -> list[dict]:
    ranked = []
    for category in CATEGORIES:
        group = [doc for doc in documents if doc["category"] == category]
        if len(group) < 3:
            raise ValueError("MT-Bench selection requires at least three cases per non-code category")
        ranked.extend(sorted(group, key=lambda doc: hashlib.sha256((SEED + digest(doc)).encode()).hexdigest())[:3])
    ranked.sort(key=lambda doc: hashlib.sha256((SEED + digest(doc)).encode()).hexdigest())
    return sorted(ranked[:20], key=lambda doc: doc["question_id"])


def load_selection(args) -> list[dict]:
    import json

    documents = [json.loads(line) for line in verified_bytes(args.source, SOURCE_SHA256).decode("utf-8").splitlines()
                 if line.strip()]
    references = [json.loads(line) for line in verified_bytes(args.references, REFERENCES_SHA256).decode("utf-8").splitlines()
                  if line.strip()]
    if len({doc["question_id"] for doc in documents}) != len(documents):
        raise ValueError("Duplicate original MT-Bench question IDs")
    by_id = {}
    for row in references:
        if row["question_id"] in by_id or row["model_id"] != "gpt-4":
            raise ValueError("Duplicate or unexpected released reference identity")
        choices = row["choices"]
        if len(choices) != 1 or choices[0]["index"] != 0:
            raise ValueError("Expected the original single reference conversation")
        by_id[row["question_id"]] = choices[0]["turns"]
    selected = select_documents(documents)
    manifest = json.loads(verified_bytes(args.selection, args.selection_sha256))
    if (
        manifest["benchmark"] != "mt_bench_noncode"
        or manifest["question_ids"] != [doc["question_id"] for doc in selected]
        or manifest["source_sha256"] != SOURCE_SHA256
        or manifest["references_sha256"] != REFERENCES_SHA256
    ):
        raise ValueError("MT-Bench cases differ from the frozen deterministic selection")
    cases = []
    for doc in selected:
        turns = doc["turns"]
        if len(turns) != 2 or any(not isinstance(value, str) or not value.strip() for value in turns):
            raise ValueError("MT-Bench requires both original nonempty user turns")
        reference = by_id.get(doc["question_id"], [])
        if doc["category"] in ("math", "reasoning") and (
            len(reference) != 2 or any(not isinstance(value, str) or not value.strip() for value in reference)
        ):
            raise ValueError("Reference-based MT-Bench judging requires both released answers")
        cases.append({**doc, "id": digest(doc), "reference": reference})
    return cases


def runtime():
    native.require_harness()
    metric = importlib.import_module("lm_eval.tasks.mt_bench.metric")
    root = Path(metric.__file__).parent
    for name, expected in NATIVE_HASHES.items():
        verified_bytes(root / name, expected)
    if metric.NEED_REF_CATS.intersection(CATEGORIES) != {"math", "reasoning"}:
        raise ValueError("Native reference-category contract changed")
    task = native.load_tasks("mt_bench")["mt_bench"]
    return task, metric


def load_samples(args, cases: list[dict]) -> list[dict]:
    verified_bytes(args.samples, args.samples_sha256)
    samples = read_jsonl(args.samples)
    if [row["case"] for row in samples] != cases:
        raise ValueError("Saved MT-Bench conversations differ from the frozen cases")
    for row in samples:
        if len(row["answers"]) != 2 or len(row["generation_info"]) != 2:
            raise ValueError("MT-Bench requires two saved outputs and native evidence records")
        for answer, info in zip(row["answers"], row["generation_info"]):
            teacher_smoke.validate_generation(answer, info, allow_truncated=args.allow_truncated)
        expected = []
        for question, answer in zip(row["case"]["turns"], row["answers"]):
            expected.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
        truncated = any(info.get("generation_truncated") for info in row["generation_info"])
        if row["history"] != expected or row["status"] != ("generated_truncated" if truncated else "generated"):
            raise ValueError("Saved conversation history or truncation status is inconsistent")
    return samples


def judge_messages(samples, metric):
    prompts = metric._load_judge_prompts()
    result = []
    for sample in samples:
        item = {**sample["case"], "responses": sample["answers"]}
        for turn in (1, 2):
            system, user, name = metric._build_judge_prompt(item, turn, prompts)
            result.append({
                "case_id": sample["case"]["id"], "question_id": item["question_id"], "turn": turn,
                "judge_prompt": name,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            })
    return result


def preflight(args, cases):
    task, metric = runtime()
    result = {"selected_conversations": 20, "selected_turns": 40, "native_files_sha256": NATIVE_HASHES}
    if args.action in ("preflight-target", "generate"):
        result["native_preflight"] = teacher_smoke.preflight(args.checkpoint)
        if task.config.generation_kwargs["max_gen_toks"] != 1024:
            raise ValueError("Pinned MT-Bench native generation ceiling changed")
    else:
        samples = load_samples(args, cases)
        identity = read_json(args.identity)
        if identity["served_model"] != args.model or identity["base_url"] != args.base_url:
            raise ValueError("Judge identity differs from declared model/endpoint")
        reject_closed(identity["public_model"])
        reject_closed(args.model)
        if not identity.get("license") or not identity.get("public_repository_revision"):
            raise ValueError("Judge requires staged open-weight identity evidence")
        result.update(
            judge_identity=identity, judge_messages=judge_messages(samples, metric),
            source_samples_sha256=args.samples_sha256,
        )
    return result, task, metric


def run(args):
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    generating = args.action == "generate"
    report = {
        "benchmark": "mt_bench_noncode", "status": "running", "phase": "preflight",
        "published_protocol_equivalence": False, "full_benchmark": False,
        "independent_judge_validation": False, "model": args.model, "base_url": args.base_url,
        "selected_conversations": 20, "selected_turns": 40, "completed_conversations": 0,
        "completed_turns": 0, "truncated_turns": 0, "model_weights_downloaded": False,
        "selection_sha256": args.selection_sha256, "source_sha256": SOURCE_SHA256,
        "references_sha256": REFERENCES_SHA256,
        "truncation_policy": "include_and_flag" if args.allow_truncated else "reject",
        "target_regenerated": generating,
    }
    write_json(args.output / "report.json", report)
    try:
        cases = load_selection(args)
        checked, task, metric = preflight(args, cases)
        report["preflight"] = {key: value for key, value in checked.items() if key != "judge_messages"}
        teacher_smoke.wait_for_server(args.base_url, args.model, args.ready_timeout)
        report["phase"] = "generation" if generating else "judging"
        write_json(args.output / "report.json", report)
        if generating:
            model = teacher_smoke.native_model(args)
            report["decoding_by_category"] = {}
            for index, case in enumerate(cases):
                state = task.init_multiturn_state(
                    case, "", {"until": [], "max_gen_toks": args.max_tokens, "seed": 1234},
                    apply_chat_template=True, chat_template=model.apply_chat_template,
                )
                infos = []
                report["decoding_by_category"][case["category"]] = state["gen_kwargs"]
                while not task.multiturn_is_done(state):
                    context, decoding = task.multiturn_next_request(state)
                    history = deepcopy(state["history"])
                    record = teacher_smoke.generate_native(
                        model, case, history, "mt_bench_noncode_pilot", index,
                        decoding, rendered_context=context,
                    )
                    record.update(history=history, turn=len(infos) + 1, decoding=decoding)
                    append_jsonl(args.output / "generation-evidence.jsonl", record)
                    teacher_smoke.validate_generation(
                        record["answer"], record["generation_info"], allow_truncated=args.allow_truncated,
                    )
                    task.multiturn_consume_response(state, record["answer"])
                    infos.append(record["generation_info"])
                    report["completed_turns"] += 1
                    report["truncated_turns"] += int(bool(record["generation_info"].get("generation_truncated")))
                    write_json(args.output / "report.json", report)
                result = task.multiturn_result(state)
                if len(result["responses"]) != 2:
                    raise ValueError("Native MT-Bench state did not complete two turns")
                append_jsonl(args.output / "samples.jsonl", {
                    "case": case, "answers": result["responses"], "history": result["history"],
                    "generation_info": infos,
                    "status": "generated_truncated" if any(info.get("generation_truncated") for info in infos) else "generated",
                })
                report["completed_conversations"] += 1
        else:
            from openai import OpenAI
            from lm_eval.api.rate_limiter import JUDGE_REQUESTS_PER_MINUTE_ENV

            samples = load_samples(args, cases)
            rows = []
            with (
                native.environment({
                    "MT_BENCH_JUDGE_API_BASE": args.base_url,
                    "MT_BENCH_JUDGE_MAX_TOKENS": "2048",
                    JUDGE_REQUESTS_PER_MINUTE_ENV: "",
                }),
                record_sdk_calls(args.output / "judge-calls.jsonl", args.base_url, args.model, checked["judge_messages"]),
                OpenAI(base_url=args.base_url, api_key="allocation-local-no-auth", max_retries=0, timeout=90) as client,
            ):
                prompts = metric._load_judge_prompts()
                for sample in samples:
                    item = {**sample["case"], "responses": sample["answers"]}
                    for turn in (1, 2):
                        row = metric._judge_one(client, args.model, item, turn, prompts)
                        append_jsonl(args.output / "judgments.jsonl", {
                            **row, "score": row["score"] if math.isfinite(row["score"]) else None,
                        })
                        rows.append(row)
            if len(rows) != 40 or any(
                row["judge_error"] or not math.isfinite(row["score"]) or not 1 <= row["score"] <= 10
                or metric._parse_score(row["judgment"]) != row["score"] for row in rows
            ):
                raise ValueError("MT-Bench requires forty valid native judgments; no partial-score fallback")
            by_key = {(row["question_id"], row["turn"]): row for row in rows}
            scores = [{"turn_1": by_key[(row["case"]["question_id"], 1)]["score"],
                       "turn_2": by_key[(row["case"]["question_id"], 2)]["score"]} for row in samples]
            aggregation = acceptance_scoring.aggregate("mt_bench", [{"doc": row["case"]} for row in samples], scores)
            report.update(
                completed_conversations=20, completed_turns=40, metrics=aggregation["metrics"],
                denominators=aggregation["denominators"], judge_success_rate=1.0,
                source_samples_sha256=args.samples_sha256, reasoning_available=True,
                truncated_turns=sum(bool(info.get("generation_truncated")) for row in samples for info in row["generation_info"]),
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
    for name in ("source", "references", "selection", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-tokens", type=int, choices=(1024,), default=1024)
    parser.add_argument("--max-model-len", type=int, choices=(16384,), default=16384)
    parser.add_argument("--ready-timeout", type=int, default=1200)
    parser.add_argument("--allow-truncated", action="store_true")
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
        value, _, _ = preflight(args, load_selection(args))
        write_json(args.output, value)
    else:
        run(args)


if __name__ == "__main__":
    main()
