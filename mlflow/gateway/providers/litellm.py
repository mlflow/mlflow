from __future__ import annotations

from typing import Any, AsyncIterable

from mlflow.gateway.config import EndpointConfig, LiteLLMConfig
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.schemas import chat, embeddings


class LiteLLMAdapter(ProviderAdapter):
    @classmethod
    def _get_model_name(cls, config):
        litellm_config = config.model.config
        if litellm_config.litellm_provider:
            return f"{litellm_config.litellm_provider}/{config.model.name}"
        return config.model.name

    @classmethod
    def chat_to_model(cls, payload, config):
        return {"model": cls._get_model_name(config), **payload}

    @classmethod
    def embeddings_to_model(cls, payload, config):
        return {"model": cls._get_model_name(config), **payload}

    @classmethod
    def model_to_chat(cls, resp, config):
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
            usage=chat.ChatUsage(
                prompt_tokens=resp["usage"]["prompt_tokens"],
                completion_tokens=resp["usage"]["completion_tokens"],
                total_tokens=resp["usage"]["total_tokens"],
            ),
        )

    @classmethod
    def model_to_chat_streaming(cls, resp, config):
        return chat.StreamResponsePayload(
            id=resp["id"],
            object=resp["object"],
            created=resp["created"],
            model=resp["model"],
            choices=[
                chat.StreamChoice(
                    index=c["index"],
                    finish_reason=c.get("finish_reason"),
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
        )

    @classmethod
    def model_to_embeddings(cls, resp, config):
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

    @classmethod
    def completions_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    def model_to_completions(cls, resp, config):
        raise NotImplementedError


class LiteLLMProvider(BaseProvider):
    """
    Provider that uses LiteLLM library to support any LLM provider.
    This serves as a fallback for providers not natively supported.
    """

    NAME = "LiteLLM"
    CONFIG_TYPE = LiteLLMConfig

    def __init__(self, config: EndpointConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, LiteLLMConfig):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.litellm_config: LiteLLMConfig = config.model.config

    @property
    def adapter_class(self):
        return LiteLLMAdapter

    def _build_litellm_kwargs(self, payload: dict[str, Any]) -> dict[str, Any]:
        kwargs = {**payload}

        if self.litellm_config.litellm_api_key:
            kwargs["api_key"] = self.litellm_config.litellm_api_key

        if self.litellm_config.litellm_api_base:
            kwargs["api_base"] = self.litellm_config.litellm_api_base
        return kwargs

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
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

    async def chat_stream(
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

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
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
            "data": [{"embedding": data["embedding"]} for data in response.data],
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }

        return self.adapter_class.model_to_embeddings(resp_dict, self.config)
