"""Prepare genuine original IFEval evidence, independently of our scorer."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import urllib.request

from scripts.helpfulness.core import digest, read_jsonl, write_json


REVISION = "b7f222394aeb06af2f8f8511b910bf8a413f9339"
SOURCE = f"https://raw.githubusercontent.com/google-research/google-research/{REVISION}"
RELEASE_CASES = 541
RESPONSE_FILE = "input_response_data_gpt4_20231107_145030.jsonl"


def original_metrics(stdout: str) -> dict:
    prompts = re.findall(r"^prompt-level: ([0-9.eE+-]+)$", stdout, re.MULTILINE)
    instructions = re.findall(r"^instruction-level: ([0-9.eE+-]+)$", stdout, re.MULTILINE)
    if len(prompts) != 2 or len(instructions) != 2:
        raise ValueError("Original scorer must print strict and loose aggregation")
    return {
        "prompt_level_strict_acc": float(prompts[0]),
        "inst_level_strict_acc": float(instructions[0]),
        "prompt_level_loose_acc": float(prompts[1]),
        "inst_level_loose_acc": float(instructions[1]),
    }


def unique_by(items: list[dict], key: str) -> dict:
    result = {item[key]: item for item in items}
    if len(result) != len(items):
        raise ValueError(f"Duplicate original {key}")
    return result


def normalize_outputs(docs, responses, strict, loose, metrics):
    """Only align original outputs; never invoke or aggregate our scorer."""
    prompts = unique_by(docs, "prompt")
    unique_by(docs, "key")
    mappings = [unique_by(items, "prompt") for items in (responses, strict, loose)]
    if any(set(mapping) != set(prompts) for mapping in mappings):
        raise ValueError("Original response/strict/loose coverage differs from prompts")
    response_map, strict_map, loose_map = mappings
    rows, cases = [], []
    for doc in docs:
        prompt = doc["prompt"]
        response = response_map[prompt]["response"]
        scores = {}
        for label, mapping in (("strict", strict_map), ("loose", loose_map)):
            output = mapping[prompt]
            flags = output["follow_instruction_list"]
            if (
                output["response"] != response
                or output["instruction_id_list"] != doc["instruction_id_list"]
                or len(flags) != len(doc["instruction_id_list"])
                or any(type(value) is not bool for value in flags)
                or type(output["follow_all_instructions"]) is not bool
                or output["follow_all_instructions"] != all(flags)
            ):
                raise ValueError("Original output/document/response alignment failed")
            scores[f"prompt_level_{label}_acc"] = output["follow_all_instructions"]
            scores[f"inst_level_{label}_acc"] = flags
        row = {
            "id": str(doc["key"]), "cluster_id": str(doc["key"]),
            "stratum": "full-original-release", "doc": doc,
            # Only the released literal input is known, not a GPT-4 chat template.
            "requests": [{"input_text": prompt}], "answers": [response],
        }
        rows.append(row)
        cases.append({"id": row["id"], "input_sha256": digest(row), "scores": scores})
    prompt_count = len(strict)
    instruction_count = sum(len(item["instruction_id_list"]) for item in strict)
    denominators = {
        key: instruction_count if key.startswith("inst_") else prompt_count
        for key in metrics
    }
    return rows, cases, {"metrics": metrics, "denominators": denominators}


def seed_scorers():
    import langdetect

    random.seed(0)
    langdetect.DetectorFactory.seed = 0


def prepare(directory: Path, nltk_directory: Path):
    directory.mkdir(parents=True, exist_ok=False)
    upstream = directory / "upstream"
    package = upstream / "instruction_following_eval"
    package.mkdir(parents=True)
    files = [
        "README.md", "requirements.txt", "evaluation_main.py", "instructions.py",
        "instructions_registry.py", "instructions_util.py",
        "data/input_data.jsonl", f"data/{RESPONSE_FILE}",
    ]
    manifest = {}
    for relative in ["LICENSE", *[f"instruction_following_eval/{name}" for name in files]]:
        url = f"{SOURCE}/{relative}"
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        destination = upstream / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        manifest[relative] = {"url": url, "sha256": hashlib.sha256(data).hexdigest()}
    docs = read_jsonl(package / "data/input_data.jsonl")
    responses = read_jsonl(package / f"data/{RESPONSE_FILE}")
    if len(docs) != RELEASE_CASES or len(responses) != RELEASE_CASES:
        raise ValueError("Frozen original IFEval release must contain all 541 prompts/responses")
    original_output = directory / "original-output"
    original_output.mkdir()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(upstream)
    environment["NLTK_DATA"] = str(nltk_directory)
    environment["PYTHONHASHSEED"] = "0"
    command = [
        sys.executable, "-c",
        "import random, runpy, langdetect; random.seed(0); "
        "langdetect.DetectorFactory.seed=0; "
        "runpy.run_module('instruction_following_eval.evaluation_main',run_name='__main__')",
        f"--input_data={package / 'data/input_data.jsonl'}",
        f"--input_response_data={package / f'data/{RESPONSE_FILE}'}",
        f"--output_dir={original_output}",
    ]
    result = subprocess.run(command, cwd=upstream, env=environment, text=True,
                            capture_output=True, timeout=1200)
    (directory / "original.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (directory / "original.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Original IFEval scorer failed ({result.returncode}); inspect original.stderr.txt")
    rows, cases, aggregate = normalize_outputs(
        docs, responses, read_jsonl(original_output / "eval_results_strict.jsonl"),
        read_jsonl(original_output / "eval_results_loose.jsonl"), original_metrics(result.stdout),
    )
    inputs = directory / "inputs.jsonl"
    inputs.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    oracle = {
        "schema_version": 1, "inputs_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
        "source": f"https://github.com/google-research/google-research/tree/{REVISION}/instruction_following_eval",
        "revision": REVISION, "protocol": "original-google-ifeval-2023-scorer-fixed-rng-0",
        "label_origin": "independent_original_scorer_replay",
        "cases": cases, "aggregate": aggregate,
    }
    write_json(directory / "oracle.json", oracle)
    write_json(directory / "source-manifest.json", {
        "revision": REVISION, "files": manifest,
        "normalization": "Exact prompt/response join; original booleans unchanged; aggregates parsed from original stdout",
        "scorer_settings": {"random_seed": 0, "langdetect_factory_seed": 0},
        "generation_request_template_verified": False,
        "reference_model_regeneration_performed": False,
    })
    from scripts.helpfulness import native

    config = (
        'schema_version = 1\nkind = "scorer_replay"\nbenchmark = "ifeval"\n'
        'protocol = "swissai-harness-ifeval-strict-and-loose"\n'
        'scope = "Full 541 released GPT-4 responses; original 2023 scorer; fixed scorer RNG 0; not model regeneration"\n'
        'data_license = "Apache-2.0 repository distribution; retain upstream notices and provider response terms"\n'
        f'[inputs]\npath = "inputs.jsonl"\nsha256 = "{oracle["inputs_sha256"]}"\n'
        f'[expected]\npath = "oracle.json"\nsha256 = "{hashlib.sha256((directory / "oracle.json").read_bytes()).hexdigest()}"\n'
        f'[assets.nltk]\ndirectory = {json.dumps(str(nltk_directory))}\n'
        f'sha256 = "{digest(native.asset_hashes(nltk_directory))}"\n'
        '[policy]\nmin_cases = 541\nfloat_tolerance = 0.000000000001\n'
    )
    (directory / "acceptance.toml").write_text(config, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--nltk-directory", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if args.nltk_directory is None:
            parser.error("prepare requires --nltk-directory")
        prepare(args.directory.resolve(), args.nltk_directory.resolve())
    else:
        from scripts.helpfulness import acceptance

        seed_scorers()
        plan = args.directory / "plan.json"
        acceptance.create_plan(args.directory / "acceptance.toml", plan)
        return acceptance.main(["run", "--plan", str(plan), "--output", str(args.directory / "report")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
