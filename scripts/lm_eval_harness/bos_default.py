#!/usr/bin/env python3
"""Whether a model needs the BOS token prepended by default (BOS=auto).

Gemma 4's tokenizer defines <bos> but never adds it (add_bos_token unset in its
tokenizer_config; Gemma 3 has true), and the model reads a prompt without it as
nonsense: plain completion prompts and log-likelihood tasks without a chat template
score around chance. Chat templates write <bos> themselves, and the harness never
doubles one (swiss-ai/lm-evaluation-harness add_bos_token), so prepending it is safe
for every task.

Standard library only: run_lm_eval.sbatch calls this on the host, before the
evaluation environment is up. A local model directory is read (config.json
model_type); a Hugging Face repo id is matched by name.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# model_type values whose tokenizer never adds the BOS token the model needs.
BOS_MODEL_TYPES = {"gemma4", "gemma4_text"}
_NAME_RE = re.compile(r"gemma[-_]?4", re.IGNORECASE)


def needs_bos(model: str) -> bool:
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
    print("true" if needs_bos(parser.parse_args().model) else "false")


if __name__ == "__main__":
    main()
