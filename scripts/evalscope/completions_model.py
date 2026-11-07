"""EvalScope model backend for a plain-text /v1/completions endpoint (run.py --completions).

EvalScope's `openai_api` backend only calls /v1/chat/completions, so the server wraps every prompt in
the model's chat template. A base model has none (or a template it was never trained on). This backend,
registered as `openai_completions`, sends the prompt as raw text to /v1/completions instead and builds
the same ModelOutput the chat backend does, so the benchmarks need no change. It reuses
OpenAICompatibleAPI's client, retries, bad-request handling and sampling parameters.

- System messages are dropped: a base model has no system turn (LiveCodeBench's base prompt has none).
- The other messages' text is joined as the prompt, so a single user message is sent verbatim.
- Streaming is not used. Parameters /v1/completions does not take (top_k, repetition_penalty,
  min_p ...) go in extra_body, as the chat backend does for the chat endpoint.
- max_tokens must be set: /v1/completions defaults to 16 tokens.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Tuple

from evalscope.api.messages import ChatMessage
from evalscope.api.messages.perf_metrics import PerformanceMetrics
from evalscope.api.model import GenerateConfig, ModelOutput, ModelUsage
from evalscope.api.registry import register_model_api
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.models.openai_compatible import (
    NON_RETRYABLE_OPENAI_ERRORS,
    OpenAICompatibleAPI,
    raise_if_error_payload,
)
from evalscope.utils import get_logger
from evalscope.utils.argument_utils import get_supported_params
from evalscope.utils.function_utils import async_retry_call, retry_call

logger = get_logger()

# Chat-only request fields that a completions request does not take.
_CHAT_ONLY = ("stream", "stream_options", "logprobs", "top_logprobs", "response_format", "tools", "tool_choice")
_STOP_REASONS = {"stop": "stop", "length": "max_tokens"}


def prompt_text(messages: List[ChatMessage]) -> str:
    """The messages' text as one prompt, system turns dropped."""
    return "".join(m.text for m in messages if m.role != "system")


@register_model_api("openai_completions")
class OpenAICompletionsAPI(OpenAICompatibleAPI):
    def request(self, input: List[ChatMessage], config: GenerateConfig) -> Dict[str, Any]:
        params = self.completion_params(config=config, tools=False)
        for key in _CHAT_ONLY:
            params.pop(key, None)
        if not hasattr(self, "_valid_completion_params"):
            self._valid_completion_params = get_supported_params(self.client.completions.create)
        extra = dict(params.pop("extra_body", None) or {})
        for key in list(params):
            if key not in self._valid_completion_params:
                extra[key] = params.pop(key)
        if extra:
            params["extra_body"] = extra
        if params.get("max_tokens") is None:
            raise ValueError("openai_completions needs max_tokens: /v1/completions defaults to 16 tokens")
        return dict(prompt=prompt_text(input), **params)

    def output(self, completion: Any, elapsed: float) -> ModelOutput:
        raise_if_error_payload(completion)
        choice = completion.choices[0]
        output = ModelOutput.from_content(
            model=completion.model or self.model_name,
            content=choice.text or "",
            stop_reason=_STOP_REASONS.get(choice.finish_reason, "unknown"),
        )
        usage = getattr(completion, "usage", None)
        if usage is not None:
            output.usage = ModelUsage(
                input_tokens=usage.prompt_tokens or 0,
                output_tokens=usage.completion_tokens or 0,
                total_tokens=usage.total_tokens or 0,
            )
        output.time = elapsed
        output.message.perf_metrics = PerformanceMetrics(
            latency=elapsed,
            ttft=None,
            input_tokens=output.usage.input_tokens if output.usage else 0,
            output_tokens=output.usage.output_tokens if output.usage else 0,
        )
        return output

    def generate(
        self, input: List[ChatMessage], tools: List[ToolInfo], tool_choice: ToolChoice, config: GenerateConfig
    ) -> ModelOutput:
        request = self.request(input, config)
        started = time.monotonic()
        try:
            completion = retry_call(
                lambda: self.client.completions.create(**request),
                retries=config.retries,
                sleep_interval=config.retry_interval,
                no_retry_exceptions=NON_RETRYABLE_OPENAI_ERRORS,
            )
            return self.output(completion, time.monotonic() - started)
        except NON_RETRYABLE_OPENAI_ERRORS as ex:
            return self.handle_bad_request(ex)
        except ValueError as ex:
            logger.error(f"Model [{self.model_name}] returned an invalid response: {ex}")
            raise

    async def generate_async(
        self, input: List[ChatMessage], tools: List[ToolInfo], tool_choice: ToolChoice, config: GenerateConfig
    ) -> ModelOutput:
        request = self.request(input, config)
        started = time.monotonic()
        try:

            async def create():
                return await self.async_client.completions.create(**request)

            completion = await async_retry_call(
                create,
                retries=config.retries,
                sleep_interval=config.retry_interval,
                no_retry_exceptions=NON_RETRYABLE_OPENAI_ERRORS,
            )
            return self.output(completion, time.monotonic() - started)
        except NON_RETRYABLE_OPENAI_ERRORS as ex:
            return self.handle_bad_request(ex)
        except ValueError as ex:
            logger.error(f"Model [{self.model_name}] returned an invalid response: {ex}")
            raise
