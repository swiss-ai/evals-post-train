"""Send the sampling and thinking settings with every request bfcl-eval's OpenAI handler makes.

The handler builds its request from the model name and the temperature alone, so the rest of the generation
config never reaches the server: ``top_p``, ``max_tokens``, ``top_k``, ``repetition_penalty``, ``reasoning_effort`` and
``extra_body`` (the chat-template kwargs of scripts/common/thinking.py, ``min_p``). ``install`` adds them to every
``chat.completions.create`` call, function-calling and prompt mode alike, without replacing anything the handler
sets. ``top_p``, ``max_tokens`` and ``reasoning_effort`` are fields of the OpenAI client; ``top_k`` and ``repetition_penalty`` are
not, so they travel in ``extra_body``, which the OpenAI-compatible server reads (as EvalScope's own client does).
"""

from typing import Any

FIELDS = ("top_p", "max_tokens", "reasoning_effort")
EXTRA_BODY_FIELDS = ("top_k", "repetition_penalty")


def request_settings(generation_config: dict[str, Any]) -> dict[str, Any]:
    """The part of EvalScope's generation config that this patch sends, as chat.completions.create kwargs."""
    settings = {key: generation_config[key] for key in FIELDS if generation_config.get(key) is not None}
    body = {
        **(generation_config.get("extra_body") or {}),
        **{key: generation_config[key] for key in EXTRA_BODY_FIELDS if generation_config.get(key) is not None},
    }
    if body:
        settings["extra_body"] = body
    return settings


def install(generation_config: dict[str, Any]) -> None:
    """Add request_settings(generation_config) to the handler's requests (idempotent)."""
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler

    settings = request_settings(generation_config)
    if not settings or getattr(OpenAICompletionsHandler, "_bfcl_request_installed", False):
        return
    generate = OpenAICompletionsHandler.generate_with_backoff

    def generate_with_settings(self, **kwargs):
        for key, value in settings.items():
            kwargs.setdefault(key, value)
        return generate(self, **kwargs)

    OpenAICompletionsHandler.generate_with_backoff = generate_with_settings
    OpenAICompletionsHandler._bfcl_request_installed = True
