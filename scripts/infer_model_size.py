#!/usr/bin/env python3
"""Infer an approximate parameter count from a model name or path."""

from __future__ import annotations

import argparse
import re


_SIZE_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[bB](?![A-Za-z])")
_EXPERT_RE = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\s*[bB](?![A-Za-z])"
)


def infer_parameter_billions(model: str) -> float | None:
    """Return the largest plausible ``B`` parameter marker in ``model``.

    Expert notation is treated as total parameters, so ``Mixtral-8x7B`` is
    conservatively classified as 56B rather than 7B. For names such as
    ``30B-A3B``, taking the largest marker selects total rather than active
    parameters.
    """

    candidates = [float(match.group(1)) for match in _SIZE_RE.finditer(model)]
    candidates.extend(
        float(match.group(1)) * float(match.group(2))
        for match in _EXPERT_RE.finditer(model)
    )
    return max(candidates) if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()
    size = infer_parameter_billions(args.model)
    if size is not None:
        print(f"{size:g}")


if __name__ == "__main__":
    main()
