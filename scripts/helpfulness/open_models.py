"""Opt-in teacher policy; model identity remains an operator assertion."""

import hashlib
import re
from pathlib import Path
from urllib.parse import urlsplit

from scripts.helpfulness.core import endpoint_model, endpoint_url, read_json


def provenance(config: dict, capabilities: set[str]) -> dict:
    path = config.get("provenance_file")
    if not path:
        raise ValueError("Open-weight teacher models require a provenance_file")
    source = Path(path)
    record = read_json(source)
    for key in ("model_id", "revision", "license", "license_url", "weights_url",
                "company", "display_name", "reviewed_by"):
        if not isinstance(record.get(key), str) or not record[key].strip() or "REPLACE_WITH" in record[key]:
            raise ValueError(f"Open-weight provenance requires verified {key}")
    if record.get("open_weights") is not True or record.get("license_reviewed") is not True:
        raise ValueError("Weights availability and license review must be explicitly confirmed")
    if not re.fullmatch(r"[a-f0-9]{40,64}", record["revision"]):
        raise ValueError("Model revision must be an immutable weights commit/hash")
    model = endpoint_model(config)
    if "weights_directory" in config:
        if Path(record.get("weights_directory", "")).resolve() != Path(config["weights_directory"]).resolve():
            raise ValueError("Provenance is not bound to the local reward snapshot")
    elif record.get("served_model") != model or record.get("base_url") != endpoint_url(config):
        raise ValueError("Provenance is not bound to this exact endpoint and served model")
    for identity in (record["model_id"], model):
        if re.search(r"(^|[/:\s_-])(gpt-[345]|chatgpt|claude|gemini|o[134](?:-|$))", identity.lower()):
            raise ValueError("A closed-model family cannot satisfy the teacher open-weight policy")
    for key in ("weights_url", "license_url"):
        parsed = urlsplit(record[key])
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query:
            raise ValueError(f"{key} must be a public, credential-free HTTPS location")
    card = source.parent / record.get("model_card_file", "")
    if not card.is_file() or hashlib.sha256(card.read_bytes()).hexdigest() != record.get("model_card_sha256"):
        raise ValueError("A staged, checksummed model card is required")
    available = record.get("capabilities")
    if not isinstance(available, list) or not all(isinstance(item, str) for item in available):
        raise ValueError("Model capabilities must be an explicit list")
    if capabilities - set(available):
        raise ValueError(f"Model lacks required capabilities: {sorted(capabilities - set(available))}")
    return {**record, "identity_assurance": "operator_assertion_not_verified_server_state"}


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
        model: provenance(config, set())
        for model, config in suite.get("reference_models", {}).items()
    }
    if any(model != config["served_model"] for model, config in references.items()):
        raise ValueError("Reference provenance keys must match the served model identities")
    return {"targets": targets, "judge": judge, "references": references}
