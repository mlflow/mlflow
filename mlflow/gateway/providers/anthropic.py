import json
import logging
import time
from typing import Any, AsyncIterable

from mlflow.gateway.config import AnthropicConfig, EndpointConfig
from mlflow.gateway.constants import (
    MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS,
    MLFLOW_AI_GATEWAY_ANTHROPIC_MAXIMUM_MAX_TOKENS,
)
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.providers.base import (
    BaseProvider,
    PassthroughAction,
    ProviderAdapter,
)
from mlflow.gateway.providers.utils import rename_payload_keys, send_request, send_stream_request
from mlflow.gateway.schemas import chat, completions
from mlflow.types.chat import Function, ToolCallDelta

_logger = logging.getLogger(__name__)

_ANTHROPIC_STRUCTURED_OUTPUTS_HEADER = "structured-outputs-2025-11-13"


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

        max_completion_tokens = payload.pop("max_completion_tokens", None)
        max_tokens = payload.get("max_tokens") or max_completion_tokens
        if max_tokens is None:
            max_tokens = MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS
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
        if system_messages := [m for m in payload["messages"] if m["role"] == "system"]:
            payload["system"] = "\n".join(m["content"] for m in system_messages)

        # remaining messages are chat history
        # we want to include only user, assistant or tool messages
        # Anthropic format of tool related messages example
        # https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview#tool-use-examples
        converted_messages = []
        for m in payload["messages"]:
            if m["role"] == "user":
                converted_messages.append(m)
            elif m["role"] == "assistant":
                if m.get("tool_calls") is not None:
                    tool_use_contents = [
                        {
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "input": json.loads(tool_call["function"]["arguments"]),
                        }
                        for tool_call in m["tool_calls"]
                    ]
                    m["content"] = tool_use_contents
                    m.pop("tool_calls")
                converted_messages.append(m)
            elif m["role"] == "tool":
                converted_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m["tool_call_id"],
                                "content": m["content"],
                            }
                        ],
                    }
                )
            else:
                _logger.info(f"Discarded unknown message: {m}")

        payload["messages"] = converted_messages

        # The range of Anthropic's temperature is 0-1, but ours is 0-2, so we halve it
        if "temperature" in payload:
            payload["temperature"] = 0.5 * payload["temperature"]

        # convert tool definition to Anthropic format
        if tools := payload.pop("tools", None):
            converted_tools = []
            for tool in tools:
                if tool["type"] != "function":
                    raise AIGatewayException(
                        status_code=422,
                        detail=(
                            "Only function calling tool is supported, but received tool type "
                            f"{tool['type']}"
                        ),
                    )

                tool_function = tool["function"]
                converted_tools.append(
                    {
                        "name": tool_function["name"],
                        "description": tool_function["description"],
                        "input_schema": tool_function["parameters"],
                    }
                )

            payload["tools"] = converted_tools

        # convert tool_choice to Anthropic format
        # OpenAI format: "none", "auto", "required", {"type": "...", "function": {"name": "..."}}
        # Anthropic format: {"type": "auto"}, {"type": "tool", "name": "..."}
        if tool_choice := payload.pop("tool_choice", None):
            match tool_choice:
                case "none":
                    payload["tool_choice"] = {"type": "none"}
                case "auto":
                    payload["tool_choice"] = {"type": "auto"}
                case "required":
                    payload["tool_choice"] = {"type": "any"}
                case {"type": "function", "function": {"name": name}}:
                    payload["tool_choice"] = {"type": "tool", "name": name}

        # Transform response_format for Anthropic structured outputs
        # Anthropic uses output_format with {"type": "json_schema", "schema": {...}}
        if response_format := payload.pop("response_format", None):
            if response_format.get("type") == "json_schema" and "json_schema" in response_format:
                payload["output_format"] = {
                    "type": "json_schema",
                    "schema": response_format["json_schema"],
                }

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
        #       "type": "tool_use",
        #       "id": "toolu_011UYCoc...",
        #       "name": "get_weather",
        #       "input": { "city": "Singapore" }
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
                        **convert_message_to_mlflow_chat(resp).model_dump()
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

        # example of function calling delta message format:
        # https://platform.openai.com/docs/guides/function-calling#streaming
        if content.get("type") == "tool_use":
            delta = chat.StreamDelta(
                tool_calls=[
                    ToolCallDelta(
                        index=0,
                        id=content.get("id"),
                        type="function",
                        function=Function(name=content.get("name")),
                    )
                ]
            )
        elif content.get("type") == "input_json_delta":
            delta = chat.StreamDelta(
                tool_calls=[
                    ToolCallDelta(index=0, function=Function(arguments=content.get("partial_json")))
                ]
            )
        else:
            delta = chat.StreamDelta(
                role=None,
                content=content.get("text"),
            )

        return chat.StreamResponsePayload(
            id=resp["id"],
            created=int(time.time()),
            model=resp["model"],
            choices=[
                chat.StreamChoice(
                    index=resp["index"],
                    finish_reason=stop_reason,
                    delta=delta,
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

    PASSTHROUGH_PROVIDER_PATHS = {
        PassthroughAction.ANTHROPIC_MESSAGES: "messages",
    }

    def __init__(self, config: EndpointConfig) -> None:
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

    def _get_headers(
        self,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """
        Generate headers for Anthropic API requests.

        Args:
            payload: Request payload (used for conditional headers like anthropic-beta)
            headers: Optional headers from client request to propagate

        Returns:
            Merged headers with provider headers taking precedence
        """
        result_headers = self.headers.copy()

        # Add conditional beta header based on payload
        if payload and payload.get("output_format"):
            if payload["output_format"].get("type") == "json_schema":
                if "anthropic-beta" not in result_headers:
                    result_headers["anthropic-beta"] = _ANTHROPIC_STRUCTURED_OUTPUTS_HEADER
                else:
                    if _ANTHROPIC_STRUCTURED_OUTPUTS_HEADER not in result_headers["anthropic-beta"]:
                        result_headers["anthropic-beta"] = (
                            f"{result_headers['anthropic-beta']},{_ANTHROPIC_STRUCTURED_OUTPUTS_HEADER}"
                        )

        if headers:
            client_headers = headers.copy()
            client_headers.pop("host", None)
            client_headers.pop("content-length", None)
            # Don't override api key or version headers
            result_headers = client_headers | result_headers

        return result_headers

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
        payload = AnthropicAdapter.chat_streaming_to_model(payload, self.config)

        headers = self._get_headers(payload)

        stream = send_stream_request(
            headers=headers,
            base_url=self.base_url,
            path="messages",
            payload=payload,
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
        payload = AnthropicAdapter.chat_to_model(payload, self.config)

        headers = self._get_headers(payload)

        resp = await send_request(
            headers=headers,
            base_url=self.base_url,
            path="messages",
            payload=payload,
        )
        return AnthropicAdapter.model_to_chat(resp, self.config)

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)

        resp = await send_request(
            headers=self._get_headers(payload),
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

    async def passthrough(
        self,
        action: PassthroughAction,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | AsyncIterable[bytes]:
        provider_path = self._validate_passthrough_action(action)

        # Add model name from config
        payload["model"] = self.config.model.name

        request_headers = self._get_headers(payload, headers)

        if payload.get("stream"):
            return send_stream_request(
                headers=request_headers,
                base_url=self.base_url,
                path=provider_path,
                payload=payload,
            )
        else:
            return await send_request(
                headers=request_headers,
                base_url=self.base_url,
                path=provider_path,
                payload=payload,
            )
