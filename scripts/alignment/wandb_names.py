"""Stable W&B identifiers that leave room for generated artifact suffixes."""

from __future__ import annotations

import hashlib


WANDB_ARTIFACT_NAME_LIMIT = 128
WANDB_ARTIFACT_SUFFIX_RESERVE = 24
WANDB_RUN_ID_MAX_LENGTH = 72
_RUN_ARTIFACT_PREFIX = "run-"
_SAMPLE_KEY_PREFIX = "samples/"


def _shorten_with_hash(value: str, max_length: int) -> str:
    """Shorten a value without making distinct long values collide."""
    if len(value) <= max_length:
        return value
    digest = hashlib.sha1(value.encode()).hexdigest()[:10]
    prefix_length = max_length - len(digest) - 1
    if prefix_length < 1:
        raise ValueError(f"Cannot safely shorten a value to {max_length} characters")
    prefix = value[:prefix_length].rstrip("-_/")
    return f"{prefix}-{digest}"


def make_wandb_run_id(model_name: str, suffix: str = "-001") -> str:
    """Return a stable run ID while preserving the full W&B display name."""
    return _shorten_with_hash(
        f"{model_name}{suffix}",
        WANDB_RUN_ID_MAX_LENGTH,
    )


def make_sample_table_key(wandb_id: str, task_name: str) -> str:
    """Build a task-specific key whose generated artifact name remains safe.

    W&B derives a table artifact name from ``run-{run_id}-{key}`` and appends
    an internal suffix. The reserve keeps the final name below its 128-character
    limit without depending on the suffix's current implementation.
    """
    key_budget = (
        WANDB_ARTIFACT_NAME_LIMIT
        - WANDB_ARTIFACT_SUFFIX_RESERVE
        - len(_RUN_ARTIFACT_PREFIX)
        - len(wandb_id)
        - 1  # separator between the run ID and key
    )
    task_budget = key_budget - len(_SAMPLE_KEY_PREFIX)
    return _SAMPLE_KEY_PREFIX + _shorten_with_hash(task_name, task_budget)


def maximum_generated_artifact_name_length(wandb_id: str, key: str) -> int:
    """Return the conservative length used by tests and diagnostics."""
    return (
        len(_RUN_ARTIFACT_PREFIX)
        + len(wandb_id)
        + 1
        + len(key)
        + WANDB_ARTIFACT_SUFFIX_RESERVE
    )
