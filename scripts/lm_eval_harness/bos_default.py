#!/usr/bin/env python3
"""Whether a model needs the BOS token prepended by default (BOS=auto).

Gemma 4's tokenizer defines <bos> but never adds it (add_bos_token unset in its
tokenizer_config; Gemma 3 has true), and the model reads a prompt without it as
nonsense: plain completion prompts and log-likelihood tasks without a chat template
score around chance. Chat templates write <bos> themselves, and the tokenizer never adds it,
so prepending it gives exactly one.

Without a chat template there is a second rule: prepend when the tokenizer defines a
BOS token (tokenizer_config.json `bos_token`) and says it does not add it itself
(`add_bos_token: false`), as the Apertus 2 instruct tokenizer: no post-processor. Pretrained
models expect the BOS, and without it log-likelihood tasks score the prompt without its
start token. A tokenizer that adds BOS itself (`add_bos_token: true`: Llama 3, Mistral, Gemma 3,
the Chonk eval tokenizer) is left alone: the harness writes the BOS it decoded from its own
ids into log-likelihood prompts, so --bos there would reach a server that adds one as well
as two. A tokenizer that says nothing (add_bos_token unset) is left alone too, and so is one
without a BOS token (Qwen); bos_probe.py catches a model that needed it anyway. With a chat
template the template writes the BOS, so only the Gemma 4 rule applies.

Standard library only: run_lm_eval.sbatch calls this on the host, before the
evaluation environment is up. A local model directory is read (config.json
model_type, tokenizer_config.json); a Hugging Face repo id is matched by name and
its tokenizer_config.json fetched (HF_TOKEN is sent when set; any failure means "no BOS token").
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from pathlib import Path

# model_type values whose tokenizer never adds the BOS token the model needs.
BOS_MODEL_TYPES = {"gemma4", "gemma4_text"}
# "gemma-4-31B-it", "gemma-4-E2B-it", "gemma4": a 4 not followed by a letter or digit, so
# Gemma-3-based names like "medgemma-4b-it" don't match.
_NAME_RE = re.compile(r"gemma[-_]?4(?![0-9a-z])", re.IGNORECASE)


def _read_tokenizer_config(tokenizer: str) -> dict:
    """tokenizer_config.json of a local directory or a public Hugging Face repo, {} if unreadable."""
    try:
        local = Path(tokenizer) / "tokenizer_config.json"
        if local.is_file():
            return json.loads(local.read_text())
        if tokenizer.startswith("/") or tokenizer.startswith("."):
            return {}
        # The gateway's provider prefix is not part of the repo id (RCP-AIaaS/Qwen/Qwen3-8B).
        repo = "/".join(tokenizer.split("/")[-2:])
        url = f"https://huggingface.co/{repo}/raw/main/tokenizer_config.json"
        headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=10) as resp:
            return json.loads(resp.read())
    except (OSError, ValueError):
        return {}


def prepends_bos(tokenizer: str) -> bool:
    """Whether the tokenizer defines a BOS token but is configured not to add it."""
    config = _read_tokenizer_config(tokenizer)
    bos = config.get("bos_token")
    if isinstance(bos, dict):
        bos = bos.get("content")
    return bool(bos) and config.get("add_bos_token") is False


def needs_bos(model: str, chat_template: bool = True) -> bool:
    if not chat_template and prepends_bos(model):
        return True
    return _model_needs_bos(model)


def _model_needs_bos(model: str) -> bool:
    config = Path(model) / "config.json"
    if config.is_file():
        try:
            model_type = json.loads(config.read_text()).get("model_type")
        except (OSError, ValueError):
            model_type = None
        if model_type:
            return model_type in BOS_MODEL_TYPES
    return bool(_NAME_RE.search(Path(model).name if model.startswith("/") else model))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="tokenizer: Hugging Face repo id or local directory")
    parser.add_argument(
        "--no-chat-template", action="store_true", help="the run sends prompts without a chat template"
    )
    args = parser.parse_args()
    print("true" if needs_bos(args.model, chat_template=not args.no_chat_template) else "false")


if __name__ == "__main__":
    main()
