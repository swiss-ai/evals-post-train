"""Send the thinking settings with every request bfcl-eval's OpenAI handler makes.

The handler builds its request from the model name and the temperature alone, so the generation config's
``reasoning_effort`` and ``extra_body`` (the chat-template kwargs of scripts/common/thinking.py, ``min_p``) never
reach the server. ``install`` adds them to every ``chat.completions.create`` call, function-calling and prompt
mode alike, without replacing anything the handler sets.
"""

from typing import Any

SETTINGS = ("reasoning_effort", "extra_body")


def request_settings(generation_config: dict[str, Any]) -> dict[str, Any]:
    """The part of EvalScope's generation config that this patch sends."""
    return {key: generation_config[key] for key in SETTINGS if generation_config.get(key) is not None}


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
