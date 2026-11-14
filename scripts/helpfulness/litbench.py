"""Validate a locally staged LitBench scalar verifier; this never scores the teacher."""

import argparse
import csv
import hashlib
import importlib.metadata
import math
from pathlib import Path
import sys

from scripts.helpfulness.core import append_jsonl, digest, read_json, read_jsonl, write_json
from scripts.helpfulness.open_models import provenance

MODEL_ID = "SAA-Lab/Llama8B-CreativeWritingVerifier"
MODEL_REVISION = "81e7bc150b17012c856eb39172c1b15fccd239b5"
DATA_REVISION = "21e2739a2bce58bf0bc9317be412e8f33ed8f370"


def load_pairs(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = read_jsonl(path)
    pairs = []
    for row in rows:
        keys = ("chosen_story", "rejected_story", "chosen_comment_id", "rejected_comment_id")
        if any(not isinstance(row.get(key), str) or not row[key].strip() for key in keys):
            raise ValueError("Rehydrate held-out story text/IDs first; IDs-only data cannot be scored")
        pairs.append({key: row[key] for key in keys})
    ids = [(row["chosen_comment_id"], row["rejected_comment_id"]) for row in pairs]
    if not pairs or len(set(ids)) != len(ids):
        raise ValueError("Held-out pairs must be nonempty and unique")
    return pairs


def validate(pairs: list[dict], reward, output: Path, identity: dict, resume: bool = False) -> dict:
    fingerprint = digest({"pairs": pairs, "identity": identity, "protocol": "litbench-raw-scalar-heldout-v1"})
    marker = output / "run.json"
    if output.exists():
        if not resume or not marker.is_file() or read_json(marker)["fingerprint"] != fingerprint:
            raise ValueError("Use a fresh output directory or resume exactly the same validation")
    else:
        write_json(marker, {"fingerprint": fingerprint, "identity": identity})
    saved = output / "pairs.jsonl"
    records = read_jsonl(saved) if saved.exists() else []
    expected = {digest(pair): pair for pair in pairs}
    by_id = {}
    for record in records:
        if record.get("id") not in expected or record["id"] in by_id:
            raise ValueError("Saved reward scores do not match unique frozen pairs")
        chosen, rejected = record.get("chosen_reward"), record.get("rejected_reward")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in (chosen, rejected)):
            raise ValueError("Saved rewards must be finite scalar logits")
        if record.get("correct") != (chosen > rejected) or record.get("tie") != (chosen == rejected):
            raise ValueError("Saved pair verdicts disagree with scalar rewards")
        by_id[record["id"]] = record
    for key, pair in expected.items():
        if key in by_id:
            continue
        chosen, rejected = reward(pair["chosen_story"]), reward(pair["rejected_story"])
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in (chosen, rejected)):
            raise ValueError("Reward model must return finite scalar logits")
        row = {"id": key, "chosen_reward": chosen, "rejected_reward": rejected,
               "correct": chosen > rejected, "tie": chosen == rejected}
        append_jsonl(saved, row)
        by_id[key] = row
    report = {
        "status": "succeeded", "scope": "scorer_validation_not_teacher_quality",
        "protocol": "litbench-raw-scalar-heldout-v1", "published_protocol_equivalence": False,
        "scientific_acceptance": "not_performed",
        "pairs": len(pairs), "correct": sum(row["correct"] for row in by_id.values()),
        "ties": sum(row["tie"] for row in by_id.values()),
        "accuracy": sum(row["correct"] for row in by_id.values()) / len(pairs),
        "fingerprint": fingerprint,
    }
    write_json(output / "report.json", report)
    return report


def local_reward(directory: Path, config: dict, device: str):
    profile = provenance(config, {"scalar_reward_raw_text"})
    if profile["model_id"] != MODEL_ID or profile["revision"] != MODEL_REVISION:
        raise ValueError("This adapter implements the pinned published scalar verifier, not arbitrary chat judges")
    checksums = profile.get("local_files")
    if not isinstance(checksums, dict) or not checksums:
        raise ValueError("Provide SHA256 hashes of every file in the local reward snapshot")
    actual = {
        str(path.relative_to(directory)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*") if path.is_file()
    }
    if actual != checksums:
        raise ValueError("The complete local reward snapshot differs from its reviewed manifest")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False,
    ).to(device).eval()
    if model.config.num_labels != 1:
        raise ValueError("LitBench requires scalar logits, not class probabilities")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    def reward(story):
        inputs = tokenizer(story.strip(), return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        with torch.inference_mode():
            return model(**inputs).logits.squeeze(-1).float().item()

    return reward, profile


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--heldout-ids", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        data = read_json(args.data_manifest)
        if data.get("source") != "SAA-Lab/LitBench-Test" or data.get("revision") != DATA_REVISION:
            raise ValueError("Use the pinned held-out LitBench-Test IDs, not training/rationale pairs")
        if data.get("rehydrated_sha256") != hashlib.sha256(args.pairs.read_bytes()).hexdigest():
            raise ValueError("Rehydrated held-out pair checksum mismatch")
        if not isinstance(data.get("terms"), str) or not data["terms"].strip() or data["terms"].startswith("REPLACE_WITH"):
            raise ValueError("Review source/Reddit data terms before validating the scorer")
        pairs = load_pairs(args.pairs)
        if data.get("ids_sha256") != hashlib.sha256(args.heldout_ids.read_bytes()).hexdigest():
            raise ValueError("Pinned held-out ID export checksum mismatch")
        ids = read_jsonl(args.heldout_ids)
        if any(not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("chosen_comment_id", "rejected_comment_id")
        ) for row in ids):
            raise ValueError("Held-out IDs must contain both nonempty chosen/rejected comment IDs")
        expected = {(row["chosen_comment_id"], row["rejected_comment_id"]) for row in ids}
        actual = {(row["chosen_comment_id"], row["rejected_comment_id"]) for row in pairs}
        if len(ids) != 2381 or len(expected) != 2381 or actual != expected:
            raise ValueError("Require complete coverage of the pinned 2381 held-out story pairs, not train pairs/subsets")
        config = {"model": MODEL_ID, "weights_directory": str(args.model_directory.resolve()),
                  "provenance_file": str(args.provenance.resolve())}
        reward, profile = local_reward(args.model_directory, config, args.device)
        runtime = {
            "python": sys.version, "device": args.device,
            "dependencies": {
                distribution.metadata["Name"]: distribution.version
                for distribution in importlib.metadata.distributions()
                if distribution.metadata["Name"].lower().replace("_", "-")
                in {"torch", "transformers", "tokenizers", "huggingface-hub", "numpy", "safetensors"}
            },
            "implementation": {
                filename: hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()
                for filename in ("litbench.py", "open_models.py", "core.py")
            },
        }
        report = validate(
            pairs, reward, args.output, {"model": profile, "data": data, "runtime": runtime}, args.resume,
        )
        print(f"Scorer validation: {report['correct']}/{report['pairs']}; not a teacher score")
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        print(f"LitBench validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
