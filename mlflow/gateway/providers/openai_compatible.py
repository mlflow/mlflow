"""
Base provider for OpenAI-compatible APIs.

Many LLM providers (Groq, DeepSeek, xAI, etc.) expose APIs that follow the
OpenAI chat/completions/embeddings format. This module provides a reusable
base class so that adding a new such provider requires only a config class,
a NAME, and a default base URL.
"""

from typing import Any, AsyncIterable

from mlflow.gateway.config import EndpointConfig
from mlflow.gateway.providers.base import BaseProvider, PassthroughAction, ProviderAdapter
from mlflow.gateway.providers.utils import send_request, send_stream_request
from mlflow.gateway.schemas import chat, embeddings
from mlflow.gateway.utils import stream_sse_data


class OpenAICompatibleAdapter(ProviderAdapter):
    """Adapter for providers that use the OpenAI request/response format.

    This is also the base class for ``OpenAIAdapter``, which adds Azure-specific
    payload handling on top.
    """

    @classmethod
    def chat_to_model(cls, payload: dict[str, Any], config: EndpointConfig) -> dict[str, Any]:
        return {"model": config.model.name, **payload}

    @classmethod
    def embeddings_to_model(cls, payload: dict[str, Any], config: EndpointConfig) -> dict[str, Any]:
        return {"model": config.model.name, **payload}

    @classmethod
    def model_to_chat(cls, resp: dict[str, Any], config: EndpointConfig) -> chat.ResponsePayload:
        # Response example (https://platform.openai.com/docs/api-reference/chat/create)
        # ```
        # {
        #    "id":"chatcmpl-abc123",
        #    "object":"chat.completion",
        #    "created":1677858242,
        #    "model":"gpt-4o-mini",
        #    "usage":{
        #       "prompt_tokens":13,
        #       "completion_tokens":7,
        #       "total_tokens":20
        #    },
        #    "choices":[
        #       {
        #          "message":{
        #             "role":"assistant",
        #             "content":"\n\nThis is a test!"
        #          },
        #          "finish_reason":"stop",
        #          "index":0
        #       }
        #    ]
        # }
        # ```
        return chat.ResponsePayload(
            id=resp["id"],
            object=resp["object"],
            created=resp["created"],
            model=resp["model"],
            choices=[
                chat.Choice(
                    index=idx,
                    message=chat.ResponseMessage(
                        role=c["message"]["role"],
                        content=c["message"].get("content"),
                        tool_calls=(
                            (calls := c["message"].get("tool_calls"))
                            and [chat.ToolCall(**tc) for tc in calls]
                        ),
                    ),
                    finish_reason=c.get("finish_reason"),
                )
                for idx, c in enumerate(resp["choices"])
            ],
            usage=cls._build_chat_usage(resp.get("usage", {})),
        )

    @classmethod
    def _build_chat_usage(cls, usage_data: dict[str, Any]) -> chat.ChatUsage:
        prompt_tokens_details = None
        if details := usage_data.get("prompt_tokens_details"):
            prompt_tokens_details = chat.PromptTokensDetails(**details)
        return chat.ChatUsage(
            prompt_tokens=usage_data.get("prompt_tokens"),
            completion_tokens=usage_data.get("completion_tokens"),
            total_tokens=usage_data.get("total_tokens"),
            prompt_tokens_details=prompt_tokens_details,
        )

    @classmethod
    def model_to_chat_streaming(
        cls, resp: dict[str, Any], config: EndpointConfig
    ) -> chat.StreamResponsePayload:
        usage = None
        if usage_data := resp.get("usage"):
            usage = cls._build_chat_usage(usage_data)

        return chat.StreamResponsePayload(
            id=resp["id"],
            object=resp["object"],
            created=resp["created"],
            model=resp["model"],
            choices=[
                chat.StreamChoice(
                    index=c["index"],
                    finish_reason=c["finish_reason"],
                    delta=chat.StreamDelta(
                        role=c["delta"].get("role"),
                        content=c["delta"].get("content"),
                        tool_calls=(
                            (calls := c["delta"].get("tool_calls"))
                            and [chat.ToolCallDelta(**tc) for tc in calls]
                        ),
                    ),
                )
                for c in resp["choices"]
            ],
            usage=usage,
        )

    @classmethod
    def model_to_embeddings(
        cls, resp: dict[str, Any], config: EndpointConfig
    ) -> embeddings.ResponsePayload:
        # Response example (https://platform.openai.com/docs/api-reference/embeddings/create):
        # ```
        # {
        #   "object": "list",
        #   "data": [
        #     {
        #       "object": "embedding",
        #       "embedding": [
        #         0.0023064255,
        #         -0.009327292,
        #         .... (1536 floats total for ada-002)
        #         -0.0028842222,
        #       ],
        #       "index": 0
        #     }
        #   ],
        #   "model": "text-embedding-ada-002",
        #   "usage": {
        #     "prompt_tokens": 8,
        #     "total_tokens": 8
        #   }
        # }
        # ```
        return embeddings.ResponsePayload(
            data=[
                embeddings.EmbeddingObject(
                    embedding=d["embedding"],
                    index=idx,
                )
                for idx, d in enumerate(resp["data"])
            ],
            model=resp["model"],
            usage=embeddings.EmbeddingsUsage(
                prompt_tokens=resp["usage"]["prompt_tokens"],
                total_tokens=resp["usage"]["total_tokens"],
            ),
        )


class OpenAICompatibleProvider(BaseProvider):
    """
    Base provider for APIs that follow the OpenAI format.

    Subclasses must set:
        - NAME: Provider display name (e.g., "Groq")
        - CONFIG_TYPE: The provider's config class
        - DEFAULT_API_BASE: Default base URL for the API (e.g., "https://api.groq.com/openai/v1")

    The config class must expose an ``api_key`` property and an optional
    ``api_base`` property. See ``_OpenAICompatibleConfig`` in config.py.
    """

    DEFAULT_API_BASE: str = ""

    PASSTHROUGH_PROVIDER_PATHS = {
        PassthroughAction.OPENAI_CHAT: "chat/completions",
        PassthroughAction.OPENAI_EMBEDDINGS: "embeddings",
    }

    def __init__(self, config: EndpointConfig, enable_tracing: bool = False) -> None:
        super().__init__(config, enable_tracing=enable_tracing)
        if config.model.config is None or not isinstance(config.model.config, self.CONFIG_TYPE):
            raise TypeError(
                f"Expected config type {self.CONFIG_TYPE.__name__}, "
                f"got {type(config.model.config).__name__}"
            )
        self._provider_config = config.model.config

    @property
    def _api_base(self) -> str:
        return getattr(self._provider_config, "api_base", None) or self.DEFAULT_API_BASE

    @property
    def _api_key(self) -> str:
        return self._provider_config.api_key

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    @property
    def adapter_class(self) -> type[ProviderAdapter]:
        return OpenAICompatibleAdapter

    def _get_headers(
        self,
        headers: dict[str, str] | None = None,
    ) -> dict[str, str]:
        result_headers = self.headers.copy()
        if headers:
            client_headers = {
                k: v
                for k, v in headers.items()
                if k.lower() not in ("host", "content-length", "authorization")
            }
            result_headers = client_headers | result_headers
        return result_headers

    # ---- chat ----

    async def _chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload_dict = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload_dict)
        resp = await send_request(
            headers=self.headers,
            base_url=self._api_base,
            path="chat/completions",
            payload=self.adapter_class.chat_to_model(payload_dict, self.config),
        )
        return self.adapter_class.model_to_chat(resp, self.config)

    async def _chat_stream(
        self, payload: chat.RequestPayload
    ) -> AsyncIterable[chat.StreamResponsePayload]:
        from fastapi.encoders import jsonable_encoder

        payload_dict = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload_dict)

        if payload_dict.get("stream_options") is None:
            payload_dict["stream_options"] = {"include_usage": True}
        elif "include_usage" not in payload_dict["stream_options"]:
            payload_dict["stream_options"]["include_usage"] = True

        stream = send_stream_request(
            headers=self.headers,
            base_url=self._api_base,
            path="chat/completions",
            payload=self.adapter_class.chat_to_model(payload_dict, self.config),
        )
        async for resp in stream_sse_data(stream):
            yield self.adapter_class.model_to_chat_streaming(resp, self.config)

    # ---- embeddings ----

    async def _embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload_dict = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload_dict)
        resp = await send_request(
            headers=self.headers,
            base_url=self._api_base,
            path="embeddings",
            payload=self.adapter_class.embeddings_to_model(payload_dict, self.config),
        )
        return self.adapter_class.model_to_embeddings(resp, self.config)

    # ---- passthrough ----

    def _extract_passthrough_token_usage(
        self, action: PassthroughAction, result: dict[str, Any]
    ) -> dict[str, int] | None:
        usage = result.get("usage")
        if not usage:
            return None
        return self._extract_token_usage_from_dict(
            usage,
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            cache_read_key="prompt_tokens_details.cached_tokens",
        )

    def _extract_streaming_token_usage(self, chunk: bytes) -> dict[str, int]:
        from mlflow.gateway.utils import parse_sse_lines

        for data in parse_sse_lines(chunk):
            if chat_usage := data.get("usage"):
                if token_usage := self._extract_token_usage_from_dict(
                    chat_usage,
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    cache_read_key="prompt_tokens_details.cached_tokens",
                ):
                    return token_usage
        return {}

    async def _passthrough(
        self,
        action: PassthroughAction,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | AsyncIterable[Any]:
        payload_with_model = {"model": self.config.model.name, **payload}
        provider_path = self._validate_passthrough_action(action)
        request_headers = self._get_headers(headers)

        supports_streaming = action != PassthroughAction.OPENAI_EMBEDDINGS

        if supports_streaming and payload_with_model.get("stream"):
            if self._enable_tracing:
                if payload_with_model.get("stream_options") is None:
                    payload_with_model["stream_options"] = {"include_usage": True}
                elif "include_usage" not in payload_with_model["stream_options"]:
                    payload_with_model["stream_options"]["include_usage"] = True

            stream = send_stream_request(
                headers=request_headers,
                base_url=self._api_base,
                path=provider_path,
                payload=payload_with_model,
            )
            return self._stream_passthrough_with_usage(stream)
        else:
            return await send_request(
                headers=request_headers,
                base_url=self._api_base,
                path=provider_path,
                payload=payload_with_model,
            )
