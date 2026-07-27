"""Alignment evaluation utilities for W&B integration."""

__all__ = [
    "create_model_evaluation_from_results",
    "create_wandb_table",
    "find_all_eval_dirs",
    "upload_multi_model_results",
]


def __getattr__(name: str):
    """Load W&B-dependent helpers only when an upload helper is requested."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import wandb_alignment_utils

    return getattr(wandb_alignment_utils, name)
