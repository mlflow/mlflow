from __future__ import annotations

import json
from typing import Any, AsyncIterable

from mlflow.gateway.config import EndpointConfig, LiteLLMConfig
from mlflow.gateway.providers.base import BaseProvider, PassthroughAction, ProviderAdapter
from mlflow.gateway.schemas import chat, embeddings
from mlflow.gateway.utils import parse_sse_lines
from mlflow.tracing.constant import TokenUsageKey


class LiteLLMAdapter(ProviderAdapter):
    @classmethod
    def _get_litellm_model_name(cls, config: EndpointConfig) -> str:
        litellm_config = config.model.config
        if litellm_config.litellm_provider:
            return f"{litellm_config.litellm_provider}/{config.model.name}"
        return config.model.name

    @classmethod
    def chat_to_model(cls, payload: dict[str, Any], config: EndpointConfig) -> dict[str, Any]:
        return {"model": cls._get_litellm_model_name(config), **payload}

    @classmethod
    def embeddings_to_model(cls, payload: dict[str, Any], config: EndpointConfig) -> dict[str, Any]:
        return {"model": cls._get_litellm_model_name(config), **payload}

    @classmethod
    def model_to_chat(cls, resp: dict[str, Any], config: EndpointConfig) -> chat.ResponsePayload:
        return chat.ResponsePayload.model_validate(resp)

    @classmethod
    def model_to_chat_streaming(
        cls, resp: dict[str, Any], config: EndpointConfig
    ) -> chat.StreamResponsePayload:
        return chat.StreamResponsePayload.model_validate(resp)

    @classmethod
    def model_to_embeddings(
        cls, resp: dict[str, Any], config: EndpointConfig
    ) -> embeddings.ResponsePayload:
        return embeddings.ResponsePayload.model_validate(resp)


class LiteLLMProvider(BaseProvider):
    """
    Provider that uses LiteLLM library to support any LLM provider.
    This serves as a fallback for providers not natively supported.
    """

    NAME = "LiteLLM"
    CONFIG_TYPE = LiteLLMConfig

    PASSTHROUGH_PROVIDER_PATHS = {
        PassthroughAction.OPENAI_CHAT: "chat/completions",
        PassthroughAction.OPENAI_EMBEDDINGS: "embeddings",
        PassthroughAction.OPENAI_RESPONSES: "responses",
        PassthroughAction.ANTHROPIC_MESSAGES: "messages",
        PassthroughAction.GEMINI_GENERATE_CONTENT: "{model}:generateContent",
        PassthroughAction.GEMINI_STREAM_GENERATE_CONTENT: "{model}:streamGenerateContent",
    }

    def __init__(self, config: EndpointConfig, enable_tracing: bool = False) -> None:
        super().__init__(config, enable_tracing=enable_tracing)
        if config.model.config is None or not isinstance(config.model.config, LiteLLMConfig):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.litellm_config: LiteLLMConfig = config.model.config

    def get_provider_name(self) -> str:
        """
        Return the actual underlying provider name instead of "LiteLLM".

        For example, if litellm_provider is "anthropic", returns "anthropic"
        instead of "LiteLLM" for more accurate tracing and metrics.
        """
        if self.litellm_config.litellm_provider:
            return self.litellm_config.litellm_provider
        return self.NAME

    @property
    def adapter_class(self):
        return LiteLLMAdapter

    def _build_litellm_kwargs(self, payload: dict[str, Any]) -> dict[str, Any]:
        kwargs = {**payload}

        if self.litellm_config.litellm_auth_config:
            kwargs.update(self.litellm_config.litellm_auth_config)

        return kwargs

    async def _chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        import litellm
        from fastapi.encoders import jsonable_encoder

        payload_dict = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload_dict)

        kwargs = self._build_litellm_kwargs(
            self.adapter_class.chat_to_model(payload_dict, self.config)
        )

        response = await litellm.acompletion(**kwargs)

        # Convert to dict for adapter processing
        resp_dict = {
            "id": response.id,
            "object": response.object,
            "created": response.created,
            "model": response.model,
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content,
                        "tool_calls": (
                            [
                                {
                                    "id": tc.id,
                                    "type": tc.type,
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in choice.message.tool_calls
                            ]
                            if choice.message.tool_calls
                            else None
                        ),
                    },
                    "finish_reason": choice.finish_reason,
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }

        return self.adapter_class.model_to_chat(resp_dict, self.config)

    async def _chat_stream(
        self, payload: chat.RequestPayload
    ) -> AsyncIterable[chat.StreamResponsePayload]:
        import litellm
        from fastapi.encoders import jsonable_encoder

        payload_dict = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload_dict)

        kwargs = self._build_litellm_kwargs(
            self.adapter_class.chat_to_model(payload_dict, self.config)
        )
        kwargs["stream"] = True

        if self._enable_tracing:
            if kwargs.get("stream_options") is None:
                kwargs["stream_options"] = {"include_usage": True}
            elif "include_usage" not in kwargs["stream_options"]:
                kwargs["stream_options"]["include_usage"] = True

        response = await litellm.acompletion(**kwargs)

        async for chunk in response:
            # Convert chunk to dict for adapter processing
            resp_dict = {
                "id": chunk.id,
                "object": chunk.object,
                "created": chunk.created,
                "model": chunk.model,
                "choices": [
                    {
                        "index": choice.index,
                        "finish_reason": choice.finish_reason,
                        "delta": {
                            "role": getattr(choice.delta, "role", None),
                            "content": getattr(choice.delta, "content", None),
                            "tool_calls": (
                                [
                                    {
                                        "index": tc_idx,
                                        "id": getattr(tc, "id", None),
                                        "type": getattr(tc, "type", None),
                                        "function": {
                                            "name": getattr(tc.function, "name", None),
                                            "arguments": getattr(tc.function, "arguments", None),
                                        }
                                        if hasattr(tc, "function")
                                        else None,
                                    }
                                    for tc_idx, tc in enumerate(choice.delta.tool_calls)
                                ]
                                if getattr(choice.delta, "tool_calls", None)
                                else None
                            ),
                        },
                    }
                    for choice in chunk.choices
                ],
            }

            yield self.adapter_class.model_to_chat_streaming(resp_dict, self.config)

    async def _embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        import litellm
        from fastapi.encoders import jsonable_encoder

        payload_dict = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload_dict)

        kwargs = self._build_litellm_kwargs(
            self.adapter_class.embeddings_to_model(payload_dict, self.config)
        )

        response = await litellm.aembedding(**kwargs)

        # Convert to dict for adapter processing
        resp_dict = {
            "data": [
                {"embedding": data["embedding"], "index": idx}
                for idx, data in enumerate(response.data)
            ],
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }

        return self.adapter_class.model_to_embeddings(resp_dict, self.config)

    def _extract_passthrough_token_usage(
        self, action: PassthroughAction, result: dict[str, Any]
    ) -> dict[str, int] | None:
        """
        Extract token usage from LiteLLM passthrough response.

        LiteLLM normalizes responses to different formats depending on the action:
        - OpenAI-style actions: usage.prompt_tokens, completion_tokens, total_tokens
        - Anthropic-style actions: usage.input_tokens, output_tokens
        - Gemini-style actions: usageMetadata.promptTokenCount, candidatesTokenCount,
          totalTokenCount
        """
        usage = result.get("usage")

        # Try OpenAI format first (most common)
        if token_usage := self._extract_token_usage_from_dict(
            usage,
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            cache_read_key="prompt_tokens_details.cached_tokens",
        ):
            return token_usage

        # Try Anthropic format
        if token_usage := self._extract_token_usage_from_dict(
            usage,
            "input_tokens",
            "output_tokens",
            cache_read_key="cache_read_input_tokens",
            cache_creation_key="cache_creation_input_tokens",
        ):
            return token_usage

        # Try Gemini format
        return self._extract_token_usage_from_dict(
            result.get("usageMetadata"),
            "promptTokenCount",
            "candidatesTokenCount",
            "totalTokenCount",
            cache_read_key="cachedContentTokenCount",
        )

    def _extract_streaming_token_usage(self, chunk: Any) -> dict[str, int]:
        """
        Extract token usage from LiteLLM streaming chunks.

        LiteLLM handles multiple provider formats:
        - OpenAI: {"usage": {"prompt_tokens": X, "completion_tokens": Y, "total_tokens": Z}}
        - Anthropic: message_start/message_delta events with usage data
        - Gemini: {"usageMetadata": {"promptTokenCount": X, ...}}

        Chunk can be:
        - bytes: SSE-formatted data (for Anthropic passthrough)
        - dict: Direct response dict
        - object with model_dump: LiteLLM response object (OpenAI/Gemini)

        Returns:
            A dictionary with token usage found in this chunk.
        """
        usage: dict[str, int] = {}
        try:
            # Handle bytes (SSE format from Anthropic)
            if isinstance(chunk, bytes):
                for data in parse_sse_lines(chunk):
                    usage.update(self._extract_usage_from_data(data))
            elif isinstance(chunk, dict):
                usage.update(self._extract_usage_from_data(chunk))
            elif hasattr(chunk, "model_dump"):
                data = chunk.model_dump()
                usage.update(self._extract_usage_from_data(data))

        except AttributeError:
            pass

        return usage

    def _extract_usage_from_data(self, data: dict[str, Any]) -> dict[str, int]:
        """Extract token usage from a parsed data dictionary."""
        usage = data.get("usage")

        # OpenAI format (in chunk.usage)
        if token_usage := self._extract_token_usage_from_dict(
            usage,
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            cache_read_key="prompt_tokens_details.cached_tokens",
        ):
            return token_usage

        # OpenAI Responses API format (usage nested in response object)
        resp_usage = data.get("response", {}).get("usage")
        if token_usage := self._extract_token_usage_from_dict(
            resp_usage,
            "input_tokens",
            "output_tokens",
            "total_tokens",
            cache_read_key="input_tokens_details.cached_tokens",
        ):
            return token_usage

        # Anthropic format (in chunk.usage)
        if token_usage := self._extract_token_usage_from_dict(
            usage,
            "input_tokens",
            "output_tokens",
            cache_read_key="cache_read_input_tokens",
            cache_creation_key="cache_creation_input_tokens",
        ):
            return token_usage

        # Anthropic message_start format (input_tokens in message.usage)
        if data.get("type") == "message_start":
            msg_usage = data.get("message", {}).get("usage")
            if token_usage := self._extract_token_usage_from_dict(
                msg_usage,
                "input_tokens",
                "output_tokens",
                cache_read_key="cache_read_input_tokens",
                cache_creation_key="cache_creation_input_tokens",
            ):
                return token_usage

        # Gemini format
        if token_usage := self._extract_token_usage_from_dict(
            data.get("usageMetadata"),
            "promptTokenCount",
            "candidatesTokenCount",
            "totalTokenCount",
            cache_read_key="cachedContentTokenCount",
        ):
            return token_usage

        return {}

    async def _stream_passthrough_with_usage(
        self, stream: AsyncIterable[Any]
    ) -> AsyncIterable[bytes]:
        """
        Stream passthrough response while accumulating token usage.

        LiteLLM streaming methods yield different types:
        - Anthropic: bytes (SSE format)
        - OpenAI/Gemini: objects with model_dump

        Token usage is only extracted when tracing is enabled.
        """
        accumulated_usage: dict[str, int] = {} if self._enable_tracing else None
        try:
            async for chunk in stream:
                if accumulated_usage is not None:
                    chunk_usage = self._extract_streaming_token_usage(chunk)
                    accumulated_usage.update(chunk_usage)
                if isinstance(chunk, bytes):
                    yield chunk
                else:
                    data = json.dumps(self._response_to_dict(chunk))
                    yield f"data: {data}\n\n".encode()
        finally:
            if accumulated_usage is not None:
                # Calculate total if we have input and output but no total
                if (
                    TokenUsageKey.INPUT_TOKENS in accumulated_usage
                    and TokenUsageKey.OUTPUT_TOKENS in accumulated_usage
                    and TokenUsageKey.TOTAL_TOKENS not in accumulated_usage
                ):
                    accumulated_usage[TokenUsageKey.TOTAL_TOKENS] = (
                        accumulated_usage[TokenUsageKey.INPUT_TOKENS]
                        + accumulated_usage[TokenUsageKey.OUTPUT_TOKENS]
                    )
                self._set_span_token_usage(accumulated_usage)

    async def _passthrough(
        self,
        action: PassthroughAction,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | AsyncIterable[Any]:
        """
        Passthrough endpoint for raw API requests using LiteLLM.

        Routes requests to the appropriate LiteLLM SDK method based on the action.
        The headers parameter is unused because LiteLLM handles auth via kwargs.
        """
        self._validate_passthrough_action(action)

        model_name = self.adapter_class._get_litellm_model_name(self.config)
        kwargs = self._build_litellm_kwargs(payload)
        kwargs["model"] = model_name

        match action:
            case PassthroughAction.OPENAI_RESPONSES:
                result = await self._passthrough_openai_responses(kwargs)
            case PassthroughAction.ANTHROPIC_MESSAGES:
                result = await self._passthrough_anthropic_messages(kwargs)
            case PassthroughAction.GEMINI_GENERATE_CONTENT:
                result = await self._passthrough_gemini_generate_content(kwargs)
            case PassthroughAction.GEMINI_STREAM_GENERATE_CONTENT:
                result = self._passthrough_gemini_stream_generate_content(kwargs)
            case PassthroughAction.OPENAI_CHAT:
                result = await self._passthrough_openai_chat(kwargs)
            case PassthroughAction.OPENAI_EMBEDDINGS:
                result = await self._passthrough_openai_embeddings(kwargs)
            case _:
                raise ValueError(f"Unsupported passthrough action: {action!r}")

        # Wrap streaming responses with token usage extraction
        if not isinstance(result, dict):
            return self._stream_passthrough_with_usage(result)

        return result

    async def _passthrough_openai_responses(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Passthrough for OpenAI Response API using litellm.aresponses()."""
        import litellm

        if kwargs.pop("stream", False):
            return self._stream_openai_responses(kwargs)

        response = await litellm.aresponses(**kwargs)
        return self._response_to_dict(response)

    def _stream_openai_responses(self, kwargs: dict[str, Any]) -> AsyncIterable[Any]:
        """Stream OpenAI Response API responses."""
        # Inject stream_options.include_usage=true to get usage in final chunk (only when tracing)
        if self._enable_tracing:
            if kwargs.get("stream_options") is None:
                kwargs["stream_options"] = {"include_usage": True}
            elif "include_usage" not in kwargs["stream_options"]:
                kwargs["stream_options"]["include_usage"] = True

        async def stream_generator():
            import litellm

            response = await litellm.aresponses(**kwargs, stream=True)
            async for chunk in response:
                yield chunk

        return stream_generator()

    async def _passthrough_anthropic_messages(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Passthrough for Anthropic Messages API using litellm.anthropic.messages.acreate()."""
        import litellm

        if kwargs.pop("stream", False):
            return self._stream_anthropic_messages(kwargs)

        response = await litellm.anthropic.messages.acreate(**kwargs)
        return self._response_to_dict(response)

    def _stream_anthropic_messages(self, kwargs: dict[str, Any]) -> AsyncIterable[bytes]:
        """Stream Anthropic Messages API responses."""

        async def stream_generator():
            import litellm

            response = await litellm.anthropic.messages.acreate(**kwargs, stream=True)
            async for chunk in response:
                # LiteLLM returns bytes directly, so we can yield them directly
                yield chunk

        return stream_generator()

    async def _passthrough_gemini_generate_content(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Passthrough for Gemini generateContent API."""
        from litellm.google_genai import agenerate_content

        response = await agenerate_content(**kwargs)
        return self._response_to_dict(response)

    def _passthrough_gemini_stream_generate_content(
        self, kwargs: dict[str, Any]
    ) -> AsyncIterable[Any]:
        """Passthrough for Gemini streamGenerateContent API."""

        async def stream_generator():
            from litellm.google_genai import agenerate_content

            response = await agenerate_content(**kwargs, stream=True)
            async for chunk in response:
                yield chunk

        return stream_generator()

    async def _passthrough_openai_chat(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Passthrough for OpenAI Chat Completions API."""
        import litellm

        if kwargs.pop("stream", False):
            return self._stream_openai_chat(kwargs)

        response = await litellm.acompletion(**kwargs)
        return self._response_to_dict(response)

    def _stream_openai_chat(self, kwargs: dict[str, Any]) -> AsyncIterable[Any]:
        """Stream OpenAI Chat Completions API responses."""
        # Inject stream_options.include_usage=true to get usage in final chunk (only when tracing)
        if self._enable_tracing:
            if kwargs.get("stream_options") is None:
                kwargs["stream_options"] = {"include_usage": True}
            elif "include_usage" not in kwargs["stream_options"]:
                kwargs["stream_options"]["include_usage"] = True

        async def stream_generator():
            import litellm

            response = await litellm.acompletion(**kwargs, stream=True)
            async for chunk in response:
                yield chunk

        return stream_generator()

    async def _passthrough_openai_embeddings(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Passthrough for OpenAI Embeddings API."""
        import litellm

        response = await litellm.aembedding(**kwargs)
        return self._response_to_dict(response)

    def _response_to_dict(self, response: Any) -> dict[str, Any]:
        """Convert a LiteLLM response object to a dictionary."""
        match response:
            case dict():
                return response
            case _ if hasattr(response, "model_dump"):
                return response.model_dump()
            case _:
                raise TypeError(f"Unexpected response type: {type(response).__name__}")
