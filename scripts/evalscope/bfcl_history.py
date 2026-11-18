"""Keep null fields out of the assistant turns bfcl-eval replays to the server.

bfcl-eval stores each reply object as it came back and sends it with the next request. A vLLM reply carries
explicit nulls (``tool_calls``, ``refusal``, ``reasoning``, ...), the OpenAI SDK keeps them on the object, and vLLM's
request validation then rejects a text-only or empty assistant turn with ``tool_calls: null`` (HTTP 400, "Input
should be iterable"): that multi-turn conversation stops and the sample is lost. ``clean_message`` stores the
turn without the nulls, with ``content`` set (empty if the reply had none) when there are no tool calls.
"""

from typing import Any


def clean_message(message: Any) -> dict[str, Any]:
    """An assistant turn (an SDK message object or a dict) as a dict without null fields."""
    data = message.model_dump() if hasattr(message, "model_dump") else dict(message)
    cleaned = {key: value for key, value in data.items() if value is not None}
    if not cleaned.get("tool_calls"):
        cleaned.pop("tool_calls", None)
        cleaned.setdefault("content", "")
    return cleaned


def install() -> None:
    """Clean every assistant turn bfcl-eval's OpenAI handler adds to the history (function-calling and prompt mode)."""
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler

    if getattr(OpenAICompletionsHandler, "_bfcl_history_installed", False):
        return

    def wrap(add_assistant_message):
        def add_clean(self, inference_data, model_response_data):
            key = "model_responses_message_for_chat_history"
            model_response_data = {**model_response_data, key: clean_message(model_response_data[key])}
            return add_assistant_message(self, inference_data, model_response_data)

        return add_clean

    for name in ("_add_assistant_message_FC", "_add_assistant_message_prompting"):
        setattr(OpenAICompletionsHandler, name, wrap(getattr(OpenAICompletionsHandler, name)))
    OpenAICompletionsHandler._bfcl_history_installed = True
