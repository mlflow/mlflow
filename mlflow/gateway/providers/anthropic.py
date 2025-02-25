import json
import time
from typing import AsyncIterable

from mlflow.gateway.config import AnthropicConfig, RouteConfig
from mlflow.gateway.constants import (
    MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS,
    MLFLOW_AI_GATEWAY_ANTHROPIC_MAXIMUM_MAX_TOKENS,
)
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.utils import rename_payload_keys, send_request, send_stream_request
from mlflow.gateway.schemas import chat, completions


class AnthropicAdapter(ProviderAdapter):
    @classmethod
    def chat_to_model(cls, payload, config):
        key_mapping = {"stop": "stop_sequences"}
        payload["model"] = config.model.name
        payload = rename_payload_keys(payload, key_mapping)

        if "top_p" in payload and "temperature" in payload:
            raise AIGatewayException(
                status_code=422, detail="Cannot set both 'temperature' and 'top_p' parameters."
            )

        max_tokens = payload.get("max_tokens", MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS)
        if max_tokens > MLFLOW_AI_GATEWAY_ANTHROPIC_MAXIMUM_MAX_TOKENS:
            raise AIGatewayException(
                status_code=422,
                detail="Invalid value for max_tokens: cannot exceed "
                f"{MLFLOW_AI_GATEWAY_ANTHROPIC_MAXIMUM_MAX_TOKENS}.",
            )
        payload["max_tokens"] = max_tokens

        if payload.pop("n", 1) != 1:
            raise AIGatewayException(
                status_code=422,
                detail="'n' must be '1' for the Anthropic provider. Received value: '{n}'.",
            )

        # Cohere uses `system` to set the system message
        # we concatenate all system messages from the user with a newline
        system_messages = [m for m in payload["messages"] if m["role"] == "system"]
        if system_messages:
            payload["system"] = "\n".join(m["content"] for m in system_messages)

        # remaining messages are chat history
        # we want to include only user and assistant messages
        payload["messages"] = [m for m in payload["messages"] if m["role"] in ("user", "assistant")]

        # The range of Anthropic's temperature is 0-1, but ours is 0-2, so we halve it
        if "temperature" in payload:
            payload["temperature"] = 0.5 * payload["temperature"]

        return payload

    @classmethod
    def model_to_chat(cls, resp, config):
        # API reference: https://docs.anthropic.com/en/api/messages#body-messages
        #
        # Example response:
        # ```
        # {
        #   "content": [
        #     {
        #       "text": "Blue is often seen as a calming and soothing color.",
        #       "type": "text"
        #     },
        #     {
        #       "source": {
        #       "type": "base64",
        #       "media_type": "image/jpeg",
        #       "data": "/9j/4AAQSkZJRg...",
        #       "type": "image",
        #       }
        #     }
        #   ],
        #   "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
        #   "model": "claude-2.1",
        #   "role": "assistant",
        #   "stop_reason": "end_turn",
        #   "stop_sequence": null,
        #   "type": "message",
        #   "usage": {
        #     "input_tokens": 10,
        #     "output_tokens": 25
        #   }
        # }
        # ```
        from mlflow.anthropic.chat import convert_message_to_mlflow_chat

        stop_reason = "length" if resp["stop_reason"] == "max_tokens" else "stop"

        return chat.ResponsePayload(
            id=resp["id"],
            created=int(time.time()),
            object="chat.completion",
            model=resp["model"],
            choices=[
                chat.Choice(
                    index=0,
                    # TODO: Remove this casting once
                    # https://github.com/mlflow/mlflow/pull/14160 is merged
                    message=chat.ResponseMessage(
                        **convert_message_to_mlflow_chat(resp).model_dump_compat()
                    ),
                    finish_reason=stop_reason,
                )
            ],
            usage=chat.ChatUsage(
                prompt_tokens=resp["usage"]["input_tokens"],
                completion_tokens=resp["usage"]["output_tokens"],
                total_tokens=resp["usage"]["input_tokens"] + resp["usage"]["output_tokens"],
            ),
        )

    @classmethod
    def chat_streaming_to_model(cls, payload, config):
        return cls.chat_to_model(payload, config)

    @classmethod
    def model_to_chat_streaming(cls, resp, config):
        content = resp.get("delta") or resp.get("content_block") or {}
        if (stop_reason := content.get("stop_reason")) is not None:
            stop_reason = "length" if stop_reason == "max_tokens" else "stop"
        return chat.StreamResponsePayload(
            id=resp["id"],
            created=int(time.time()),
            model=resp["model"],
            choices=[
                chat.StreamChoice(
                    index=resp["index"],
                    finish_reason=stop_reason,
                    delta=chat.StreamDelta(
                        role=None,
                        content=content.get("text"),
                    ),
                )
            ],
        )

    @classmethod
    def model_to_completions(cls, resp, config):
        stop_reason = "stop" if resp["stop_reason"] == "stop_sequence" else "length"

        return completions.ResponsePayload(
            created=int(time.time()),
            object="text_completion",
            model=resp["model"],
            choices=[
                completions.Choice(
                    index=0,
                    text=resp["completion"],
                    finish_reason=stop_reason,
                )
            ],
            usage=completions.CompletionsUsage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            ),
        )

    @classmethod
    def completions_to_model(cls, payload, config):
        key_mapping = {"max_tokens": "max_tokens_to_sample", "stop": "stop_sequences"}

        payload["model"] = config.model.name

        if "top_p" in payload:
            raise AIGatewayException(
                status_code=422,
                detail="Cannot set both 'temperature' and 'top_p' parameters. "
                "Please use only the temperature parameter for your query.",
            )
        max_tokens = payload.get("max_tokens", MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS)

        if max_tokens > MLFLOW_AI_GATEWAY_ANTHROPIC_MAXIMUM_MAX_TOKENS:
            raise AIGatewayException(
                status_code=422,
                detail="Invalid value for max_tokens: cannot exceed "
                f"{MLFLOW_AI_GATEWAY_ANTHROPIC_MAXIMUM_MAX_TOKENS}.",
            )

        payload["max_tokens"] = max_tokens

        if payload.get("stream", False):
            raise AIGatewayException(
                status_code=422,
                detail="Setting the 'stream' parameter to 'true' is not supported with the MLflow "
                "Gateway.",
            )
        n = payload.pop("n", 1)
        if n != 1:
            raise AIGatewayException(
                status_code=422,
                detail=f"'n' must be '1' for the Anthropic provider. Received value: '{n}'.",
            )

        payload = rename_payload_keys(payload, key_mapping)

        if payload["prompt"].startswith("Human: "):
            payload["prompt"] = "\n\n" + payload["prompt"]

        if not payload["prompt"].startswith("\n\nHuman: "):
            payload["prompt"] = "\n\nHuman: " + payload["prompt"]

        if not payload["prompt"].endswith("\n\nAssistant:"):
            payload["prompt"] = payload["prompt"] + "\n\nAssistant:"

        # The range of Anthropic's temperature is 0-1, but ours is 0-2, so we halve it
        if "temperature" in payload:
            payload["temperature"] = 0.5 * payload["temperature"]

        return payload

    @classmethod
    def embeddings_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    def model_to_embeddings(cls, resp, config):
        raise NotImplementedError


class AnthropicProvider(BaseProvider, AnthropicAdapter):
    NAME = "Anthropic"
    CONFIG_TYPE = AnthropicConfig

    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, AnthropicConfig):
            raise TypeError(f"Invalid config type {config.model.config}")
        self.anthropic_config: AnthropicConfig = config.model.config

    @property
    def headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.anthropic_config.anthropic_api_key,
            "anthropic-version": self.anthropic_config.anthropic_version,
        }

    @property
    def base_url(self) -> str:
        return "https://api.anthropic.com/v1"

    @property
    def adapter_class(self) -> type[ProviderAdapter]:
        return AnthropicAdapter

    def get_endpoint_url(self, route_type: str) -> str:
        if route_type == "llm/v1/chat":
            return f"{self.base_url}/messages"
        elif route_type == "llm/v1/completions":
            return f"{self.base_url}/complete"
        else:
            raise ValueError(f"Invalid route type {route_type}")

    async def chat_stream(
        self, payload: chat.RequestPayload
    ) -> AsyncIterable[chat.StreamResponsePayload]:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        stream = send_stream_request(
            headers=self.headers,
            base_url=self.base_url,
            path="messages",
            payload=AnthropicAdapter.chat_streaming_to_model(payload, self.config),
        )

        indices = []
        metadata = {}
        async for chunk in stream:
            chunk = chunk.strip()
            if not chunk:
                continue

            # No handling on "event" lines
            prefix, content = chunk.split(b":", 1)
            if prefix != b"data":
                continue

            # See https://docs.anthropic.com/claude/reference/messages-streaming
            resp = json.loads(content.decode("utf-8"))

            # response id and model are only present in `message_start`
            if resp["type"] == "message_start":
                metadata["id"] = resp["message"]["id"]
                metadata["model"] = resp["message"]["model"]
                continue

            if resp["type"] not in (
                "message_delta",
                "content_block_start",
                "content_block_delta",
            ):
                continue

            index = resp.get("index")
            if index is not None and index not in indices:
                indices.append(index)

            resp.update(metadata)
            if resp["type"] == "message_delta":
                for index in indices:
                    yield AnthropicAdapter.model_to_chat_streaming(
                        {**resp, "index": index},
                        self.config,
                    )
            else:
                yield AnthropicAdapter.model_to_chat_streaming(resp, self.config)

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        resp = await send_request(
            headers=self.headers,
            base_url=self.base_url,
            path="messages",
            payload=AnthropicAdapter.chat_to_model(payload, self.config),
        )
        return AnthropicAdapter.model_to_chat(resp, self.config)

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)

        resp = await send_request(
            headers=self.headers,
            base_url=self.base_url,
            path="complete",
            payload=AnthropicAdapter.completions_to_model(payload, self.config),
        )

        # Example response:
        # Documentation: https://docs.anthropic.com/claude/reference/complete_post
        # ```
        # {
        #     "completion": " Hello! My name is Claude."
        #     "stop_reason": "stop_sequence",
        #     "model": "claude-instant-1.1",
        #     "truncated": False,
        #     "stop": None,
        #     "log_id": "dee173f87ddf1357da639dee3c38d833",
        #     "exception": None,
        # }
        # ```

        return AnthropicAdapter.model_to_completions(resp, self.config)
