"""Opt-in teacher policy; model identity remains an operator assertion."""

import hashlib
import re
from pathlib import Path
from urllib.parse import urlsplit

from scripts.helpfulness.core import digest, endpoint_model, endpoint_url, read_json, read_jsonl


def identity(config: dict) -> tuple[Path, dict]:
    path = config.get("provenance_file")
    if not path:
        raise ValueError("Open-weight teacher models require a provenance_file")
    source = Path(path)
    record = read_json(source)
    for key in ("model_id", "license", "license_url", "weights_url",
                "company", "display_name", "reviewed_by"):
        if not isinstance(record.get(key), str) or not record[key].strip() or "REPLACE_WITH" in record[key]:
            raise ValueError(f"Open-weight provenance requires verified {key}")
    if record.get("open_weights") is not True or record.get("license_reviewed") is not True:
        raise ValueError("Weights availability and license review must be explicitly confirmed")
    reject_closed(record["model_id"])
    for key in ("weights_url", "license_url"):
        parsed = urlsplit(record[key])
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query:
            raise ValueError(f"{key} must be a public, credential-free HTTPS location")
    card = source.parent / record.get("model_card_file", "")
    if not card.is_file() or hashlib.sha256(card.read_bytes()).hexdigest() != record.get("model_card_sha256"):
        raise ValueError("A staged, checksummed model card is required")
    return source, record


def reject_closed(model: str) -> None:
    if re.search(r"(^|[/:\s_-])(gpt-[345]|chatgpt|claude|gemini|o[134](?:-|$))", model.lower()):
        raise ValueError("A closed-model family cannot satisfy the teacher open-weight policy")


def provenance(config: dict, capabilities: set[str]) -> dict:
    _, record = identity(config)
    if record.get("kind") == "released_answers":
        raise ValueError("Released-answer provenance is reference-only, not a target/judge deployment")
    if not re.fullmatch(r"[a-f0-9]{40,64}", str(record.get("revision", ""))):
        raise ValueError("Model revision must be an immutable weights commit/hash")
    model = endpoint_model(config)
    reject_closed(model)
    if "weights_directory" in config:
        if Path(record.get("weights_directory", "")).resolve() != Path(config["weights_directory"]).resolve():
            raise ValueError("Provenance is not bound to the local reward snapshot")
    elif record.get("served_model") != model or record.get("base_url") != endpoint_url(config):
        raise ValueError("Provenance is not bound to this exact endpoint and served model")
    available = record.get("capabilities")
    if not isinstance(available, list) or not all(isinstance(item, str) for item in available):
        raise ValueError("Model capabilities must be an explicit list")
    if capabilities - set(available):
        raise ValueError(f"Model lacks required capabilities: {sorted(capabilities - set(available))}")
    return {**record, "identity_assurance": "operator_assertion_not_verified_server_state"}


def reference_provenance(config: dict) -> dict:
    source, record = identity(config)
    if record.get("kind") != "released_answers":
        return provenance(config, set())
    from scripts.helpfulness import open_benchmarks

    if record.get("reference_model") != record["model_id"]:
        raise ValueError("Released reference must use its canonical open-model identity")
    revision = record.get("answers_revision", "")
    url = urlsplit(record.get("answers_url", ""))
    if (
        not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40,64}", revision)
        or url.scheme != "https" or not url.hostname or url.username or url.password or url.query
        or revision not in url.path.split("/")
    ):
        raise ValueError("Released answers require a public URL pinned to their immutable data revision")
    paths = {}
    for key in ("answers", "cases"):
        relative = record.get(key + "_file")
        if not isinstance(relative, str) or not relative.strip() or Path(relative).is_absolute():
            raise ValueError("Released reference files must be explicit profile-relative paths")
        path = (source.parent / relative).resolve()
        if (
            not path.is_relative_to(source.parent.resolve()) or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != record.get(key + "_sha256")
        ):
            raise ValueError(f"Released reference {key} checksum/path mismatch")
        paths[key] = path
    cases = read_jsonl(paths["cases"])
    verified = open_benchmarks.arena_references(cases, read_jsonl(paths["answers"]), record["model_id"])
    if [row.get("reference_answer") for row in cases] != [row["reference_answer"] for row in verified]:
        raise ValueError("Ported reference answers differ from the released native final answers")
    if any(row.get("reference_model") != record["model_id"] for row in cases):
        raise ValueError("Ported reference model differs from its released source")
    return {
        **record, "cases_digest": digest(open_benchmarks.normalize("arena_hard_v2", verified, {})),
        "generation_weights_revision_verified": False,
        "identity_assurance": "operator_review_of_publisher_attribution_not_generation_attestation",
    }


def validate_suite(suite: dict, checkpoints: list[dict], benchmarks: list[dict]) -> dict:
    policy = suite.get("model_policy")
    if policy is None:
        return {}
    if policy != "open_weights_only" or suite.get("purpose") != "helpfulness_chitchat_teacher":
        raise ValueError("Unknown/scoped incorrectly model policy")
    names = {item["name"] for item in benchmarks}
    requirements = {"chat"}
    if "sycophancy" in names:
        requirements |= {"text_completions", "token_logprobs"}
    if names & {"eqbench3", "eqbench_creative_v3"}:
        requirements.add("min_p")
    targets = {item["name"]: provenance(item["endpoint"], requirements) for item in checkpoints}
    from scripts.helpfulness.adapters import JUDGED

    judge_requirements = {"chat", "min_p"} if "eqbench_creative_v3" in names else {"chat"}
    judge = provenance(suite["judge"], judge_requirements) if names & JUDGED else None
    if judge:
        for target in targets.values():
            if target["model_id"] == judge["model_id"]:
                raise ValueError("An independent judge must not be another alias/revision of the target model")
    references = {
        model: reference_provenance(config)
        for model, config in suite.get("reference_models", {}).items()
    } if "arena_hard_v2" in names else {}
    if any(model != config.get("reference_model", config.get("served_model")) for model, config in references.items()):
        raise ValueError("Reference provenance keys must match the served model identities")
    return {"targets": targets, "judge": judge, "references": references}
