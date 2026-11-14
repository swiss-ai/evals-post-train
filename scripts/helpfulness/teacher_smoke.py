"""Bounded Apertus 2 load/generation diagnostic; not full teacher-suite acceptance."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

from scripts.helpfulness import native, open_benchmarks
from scripts.helpfulness.core import append_jsonl, read_json, write_json


FAILURE_FLAGS = (
    "generation_request_failed", "generation_prompt_too_long",
    "generation_parse_failed", "generation_map_failed", "generation_think_truncated",
    "generation_truncated", "generation_prompt_mismatch",
    "generation_unknown_output", "generation_has_claim",
    "generation_call_counter_reused", "generation_unexpected_call",
)


def checkpoint_inventory(directory: Path) -> dict:
    config = read_json(directory / "config.json")
    if config.get("architectures") != ["Apertus2KDAForCausalLM"]:
        raise ValueError("This smoke path requires the Apertus2KDAForCausalLM checkpoint")
    index = read_json(directory / "model.safetensors.index.json")
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Checkpoint requires a nonempty sharded safetensors index")
    shards = sorted(set(mapping.values()))
    for name in shards:
        if (
            not isinstance(name, str) or Path(name).name != name
            or "\\" in name or not name.endswith(".safetensors")
        ):
            raise ValueError("Checkpoint index contains an unsafe shard path")
        path = directory / name
        if not path.is_file() or path.stat().st_size <= 8:
            raise ValueError(f"Missing or empty checkpoint shard: {name}")
    files = ("config.json", "model.safetensors.index.json", "tokenizer.json",
             "tokenizer_config.json", "generation_config.json", "configuration_apertus2.py")
    return {
        "directory": str(directory.resolve()), "architecture": config["architectures"][0],
        "shards": len(shards),
        "shard_bytes": sum((directory / name).stat().st_size for name in shards),
        "metadata_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in files
        },
        "weight_contents_hashed": False,
        "chat_template_present": bool(read_json(directory / "tokenizer_config.json").get("chat_template")),
    }


def validate_generation(answer: str, info: dict) -> None:
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Native generation produced no final reply")
    required = FAILURE_FLAGS[:5]
    if not isinstance(info, dict) or any(flag not in info for flag in required):
        raise ValueError("Native generation did not retain required failure evidence")
    failed = [flag for flag in FAILURE_FLAGS if info.get(flag)]
    if failed:
        raise ValueError("Native generation failed: " + ", ".join(failed))
    if not isinstance(info.get("apertus2_output"), str) or not info["apertus2_output"]:
        raise ValueError("Missing raw native output evidence")


def preflight(checkpoint: Path) -> dict:
    inventory = checkpoint_inventory(checkpoint)
    native.require_harness()
    from apertus_common import load_encoding
    from lm_eval.models.apertus2 import Apertus2Completions

    encoding = load_encoding(checkpoint)
    return {
        "checkpoint": inventory, "harness_commit": native.harness_pin(),
        "apertus_common_version": importlib.metadata.version("apertus-common"),
        "native_client": Apertus2Completions.__name__,
        "vocab_size": encoding.vocab_size, "bos_token_id": encoding.bos_token_id,
        "wait_token_id": encoding.controls.wait,
        "weights_loaded": False, "inference_performed": False,
    }


def wait_for_server(base_url: str, model_name: str, seconds: int) -> None:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
        or parsed.path != "/v1" or parsed.query or parsed.fragment
        or parsed.username or parsed.password
    ):
        raise ValueError("Smoke inference must use the allocation-local loopback /v1 endpoint")
    deadline = time.monotonic() + seconds
    last_error = "server not checked"
    while True:
        try:
            with urlopen(base_url + "/models", timeout=10) as response:
                data = json.load(response)
            models = {row["id"] for row in data["data"]}
            if model_name not in models:
                raise ValueError(f"Ready server does not advertise requested model {model_name}")
            return
        except (URLError, TimeoutError) as error:
            last_error = str(error)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"Server readiness timed out: {last_error}")
        time.sleep(min(5, remaining))


def run(args) -> dict:
    output = args.output
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    report = {
        "schema_version": 1, "scope": "bounded_teacher_infrastructure_and_or_bench_smoke",
        "status": "running", "phase": "preflight",
        "full_teacher_benchmark": False, "scientific_acceptance": "not_performed",
        "judge_used": False, "model_weights_downloaded": False,
        "selected_cases": args.limit, "scored_cases": 0,
    }
    write_json(output / "report.json", report)
    try:
        report["preflight"] = preflight(args.checkpoint)
        assets = open_benchmarks.staged_assets("or_bench", args.upstream)
        with args.source.open(encoding="utf-8-sig", newline="") as stream:
            cases = open_benchmarks.normalize("or_bench", list(csv.DictReader(stream)), {})
        if len(cases) < args.limit:
            raise ValueError("OR-Bench source has fewer cases than the declared limit")
        report.update(
            source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
            source_cases=len(cases), upstream=assets, phase="server_readiness",
            served_model=args.model, base_url=args.base_url,
            decoding={"temperature": 0.6, "top_p": 0.95, "max_gen_toks": args.max_tokens,
                      "reasoning_effort": "low", "prefill": "after_think", "seed": 1234},
        )
        write_json(output / "report.json", report)
        wait_for_server(args.base_url, args.model, args.ready_timeout)
        from lm_eval.api.instance import Instance
        from lm_eval.models.apertus2 import Apertus2Completions

        model = Apertus2Completions(
            base_url=args.base_url + "/completions", model=args.model,
            tokenizer=str(args.checkpoint), max_model_len=args.max_model_len,
            max_length=args.max_model_len, max_gen_toks=args.max_tokens,
            num_concurrent=1, batch_size=1, max_retries=1, timeout=180,
            reasoning_effort="low", log_native="full",
        )
        records = []
        report["phase"] = "native_generation_and_keyword_scoring"
        write_json(output / "report.json", report)
        for index, case in enumerate(cases[:args.limit]):
            context = model.apply_chat_template([{"role": "user", "content": case["question"]}])
            request = Instance(
                request_type="generate_until", doc=case, idx=0,
                arguments=(context, {"temperature": 0.6, "top_p": 0.95,
                                     "max_gen_toks": args.max_tokens, "until": []}),
                metadata=("or_bench_smoke", index, 1),
            )
            answers = model.generate_until([request], disable_tqdm=True)
            info = request.length_info[0] if request.length_info else {}
            answer = answers[0] if len(answers) == 1 else ""
            record = {"case": case, "answer": answer, "generation_info": info,
                      "status": "generated_unvalidated"}
            append_jsonl(output / "generation-evidence.jsonl", record)
            validate_generation(answer, info)
            scores, _ = open_benchmarks.score(
                "or_bench", case, answer, None, assets, {"keyword_profile": "opensource"}, {},
            )
            record.update(status="succeeded", scores=scores)
            append_jsonl(output / "samples.jsonl", record)
            records.append(record)
            report["scored_cases"] = len(records)
            write_json(output / "report.json", report)
        report.update(
            status="succeeded", phase="completed",
            benchmark=open_benchmarks.summarize("or_bench", records, args.limit),
            native_format_generation_verified=True,
        )
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(output / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--limit", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--max-tokens", type=int, choices=range(1, 16385), default=8192)
    parser.add_argument("--max-model-len", type=int, choices=range(1, 16385), default=16384)
    parser.add_argument("--ready-timeout", type=int, choices=range(1, 1801), default=1200)
    args = parser.parse_args(argv)
    if args.command == "run" and not all((args.source, args.upstream, args.model)):
        parser.error("run requires --source, --upstream and --model")
    try:
        if args.command == "preflight":
            if args.output.exists():
                raise ValueError("Preflight output already exists")
            write_json(args.output, preflight(args.checkpoint))
        else:
            run(args)
    except (ValueError, RuntimeError, OSError, ImportError, KeyError) as error:
        print(f"Teacher smoke failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
