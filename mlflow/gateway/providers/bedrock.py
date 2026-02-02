from __future__ import annotations

import json
import time
from enum import Enum
from typing import AsyncIterable

from mlflow.gateway.config import AmazonBedrockConfig, AWSIdAndKey, AWSRole, EndpointConfig
from mlflow.gateway.constants import (
    MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS,
)
from mlflow.gateway.exceptions import AIGatewayConfigException, AIGatewayException
from mlflow.gateway.providers.anthropic import AnthropicAdapter
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.cohere import CohereAdapter
from mlflow.gateway.providers.utils import rename_payload_keys
from mlflow.gateway.schemas import chat, completions

AWS_BEDROCK_ANTHROPIC_MAXIMUM_MAX_TOKENS = 8191


class AmazonBedrockAnthropicAdapter(AnthropicAdapter):
    @classmethod
    def chat_to_model(cls, payload, config):
        payload = super().chat_to_model(payload, config)
        # "model" keys are not supported in Bedrock"
        payload.pop("model", None)
        payload["anthropic_version"] = "bedrock-2023-05-31"
        return payload

    @classmethod
    def completions_to_model(cls, payload, config):
        payload = super().completions_to_model(payload, config)

        if "\n\nHuman:" not in payload.get("stop_sequences", []):
            payload.setdefault("stop_sequences", []).append("\n\nHuman:")

        payload["max_tokens_to_sample"] = min(
            payload.get("max_tokens_to_sample", MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS),
            AWS_BEDROCK_ANTHROPIC_MAXIMUM_MAX_TOKENS,
        )
        payload["anthropic_version"] = "bedrock-2023-05-31"

        # "model" keys are not supported in Bedrock"
        payload.pop("model", None)
        return payload

    @classmethod
    def model_to_completions(cls, payload, config):
        payload["model"] = config.model.name
        return super().model_to_completions(payload, config)


class AWSTitanAdapter(ProviderAdapter):
    # TODO handle top_p, top_k, etc.
    @classmethod
    def completions_to_model(cls, payload, config):
        n = payload.pop("n", 1)
        if n != 1:
            raise AIGatewayException(
                status_code=422,
                detail=f"'n' must be '1' for AWS Titan models. Received value: '{n}'.",
            )

        # The range of Titan's temperature is 0-1, but ours is 0-2, so we halve it
        if "temperature" in payload:
            payload["temperature"] = 0.5 * payload["temperature"]
        return {
            "inputText": payload.pop("prompt"),
            "textGenerationConfig": rename_payload_keys(
                payload, {"max_tokens": "maxTokenCount", "stop": "stopSequences"}
            ),
        }

    @classmethod
    def model_to_completions(cls, resp, config):
        return completions.ResponsePayload(
            created=int(time.time()),
            object="text_completion",
            model=config.model.name,
            choices=[
                completions.Choice(
                    index=idx,
                    text=candidate.get("outputText"),
                    finish_reason=None,
                )
                for idx, candidate in enumerate(resp.get("results", []))
            ],
            usage=completions.CompletionsUsage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            ),
        )

    @classmethod
    def embeddings_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    def model_to_embeddings(cls, resp, config):
        raise NotImplementedError


class AI21Adapter(ProviderAdapter):
    # TODO handle top_p, top_k, etc.
    @classmethod
    def completions_to_model(cls, payload, config):
        return rename_payload_keys(
            payload,
            {
                "stop": "stopSequences",
                "n": "numResults",
                "max_tokens": "maxTokens",
            },
        )

    @classmethod
    def model_to_completions(cls, resp, config):
        return completions.ResponsePayload(
            created=int(time.time()),
            object="text_completion",
            model=config.model.name,
            choices=[
                completions.Choice(
                    index=idx,
                    text=candidate.get("data", {}).get("text"),
                    finish_reason=None,
                )
                for idx, candidate in enumerate(resp.get("completions", []))
            ],
            usage=completions.CompletionsUsage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            ),
        )

    @classmethod
    def embeddings_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    def model_to_embeddings(cls, resp, config):
        raise NotImplementedError


class ConverseAdapter(ProviderAdapter):
    # TODO Add tool calling support (toolUse/toolResult blocks, tool_choice mapping)
    # TODO handle images, documents, videos

    @classmethod
    def _transform_message_content(cls, content):
        if isinstance(content, str):
            return [{"text": content}]

        if isinstance(content, list):
            converse_content = []
            for part in content:
                part_type = part.get("type")

                if part_type == "text":
                    converse_content.append({"text": part["text"]})
                elif part_type == "image_url":
                    raise AIGatewayException(
                        status_code=422,
                        detail="Image content is not yet supported in "
                        "Bedrock Converse API integration",
                    )
                elif part_type == "document":
                    raise AIGatewayException(
                        status_code=422,
                        detail="Document content is not yet supported in "
                        "Bedrock Converse API integration",
                    )
                elif part_type == "video":
                    raise AIGatewayException(
                        status_code=422,
                        detail="Video content is not yet supported in "
                        "Bedrock Converse API integration",
                    )
                else:
                    raise AIGatewayException(
                        status_code=422,
                        detail=f"Unsupported content part type: '{part_type}'. "
                        "Converse API currently only supports 'text' content parts.",
                    )
            return converse_content

        return [{"text": str(content)}]

    @classmethod
    def chat_to_model(cls, payload, config):
        n = payload.pop("n", 1)
        if n != 1:
            raise AIGatewayException(
                status_code=422,
                detail=f"'n' must be '1' for Bedrock Converse API. Received value: '{n}'.",
            )

        # TODO: Add tool calling support in a future update
        if payload.get("tools"):
            raise AIGatewayException(
                status_code=422,
                detail="Tool calling is not yet supported in Bedrock Converse API integration. "
                "This feature will be added in a future update.",
            )
        if payload.get("tool_choice"):
            raise AIGatewayException(
                status_code=422,
                detail="tool_choice is not yet supported in Bedrock Converse API integration. "
                "This feature will be added in a future update.",
            )

        messages = payload.get("messages", [])

        # Separate system messages from regular messages
        # Converse API expects system messages in a separate "system" parameter
        system_messages = []
        converse_messages = []

        for msg in messages:
            role = msg["role"]

            # TODO: Add tool message support in a future update
            if role == "tool":
                raise AIGatewayException(
                    status_code=422,
                    detail="Tool result messages (role='tool') are not yet supported in "
                    "Bedrock Converse API integration. "
                    "This feature will be added in a future update.",
                )
            if msg.get("tool_calls"):
                raise AIGatewayException(
                    status_code=422,
                    detail="Assistant messages with tool_calls are not yet supported in "
                    "Bedrock Converse API integration. "
                    "This feature will be added in a future update.",
                )

            if role == "system":
                # System messages go in a separate list
                # Converse API only supports text in system blocks
                content = msg.get("content", "")
                if not isinstance(content, str):
                    raise AIGatewayException(
                        status_code=422,
                        detail="System message content must be a string. "
                        "Bedrock Converse API does not support content parts (e.g., images) "
                        "in system messages.",
                    )
                system_messages.append({"text": content})
            else:
                converse_msg = {
                    "role": role,
                    "content": cls._transform_message_content(msg.get("content", "")),
                }
                converse_messages.append(converse_msg)

        request = {"messages": converse_messages}
        if system_messages:
            request["system"] = system_messages

        inference_config = {}
        if "temperature" in payload:
            inference_config["temperature"] = payload["temperature"]
        if "max_tokens" in payload:
            inference_config["maxTokens"] = payload["max_tokens"]
        elif "max_completion_tokens" in payload:
            inference_config["maxTokens"] = payload["max_completion_tokens"]
        if "top_p" in payload:
            inference_config["topP"] = payload["top_p"]
        if "stop" in payload:
            inference_config["stopSequences"] = payload["stop"]

        if inference_config:
            request["inferenceConfig"] = inference_config

        return request

    @classmethod
    def model_to_chat(cls, resp, config):
        output_message = resp.get("output", {}).get("message", {})
        content_blocks = output_message.get("content", [])

        text_parts = [block["text"] for block in content_blocks if "text" in block]
        content = "".join(text_parts) if text_parts else None

        stop_reason = resp.get("stopReason", "end_turn")
        finish_reason_map = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "content_filtered": "content_filter",
        }
        finish_reason = finish_reason_map.get(stop_reason, "stop")

        message = chat.ResponseMessage(
            role=output_message.get("role", "assistant"),
            content=content,
            refusal=None,
        )

        usage_data = resp.get("usage", {})
        usage = chat.ChatUsage(
            prompt_tokens=usage_data.get("inputTokens"),
            completion_tokens=usage_data.get("outputTokens"),
            total_tokens=usage_data.get("totalTokens"),
        )

        return chat.ResponsePayload(
            id=None,  # Converse API doesn't provide request ID
            created=int(time.time()),
            object="chat.completion",
            model=config.model.name,
            choices=[
                chat.Choice(
                    index=0,
                    message=message,
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )

    @classmethod
    def chat_streaming_to_model(cls, payload, config):
        return cls.chat_to_model(payload, config)

    @classmethod
    def model_to_chat_streaming(cls, chunk, config):
        """
        The converse_stream API returns events like:
        - messageStart: Start of message
        - contentBlockStart: Start of content block
        - contentBlockDelta: Incremental content (text)
        - contentBlockStop: End of content block
        - messageStop: End of message with stop reason
        - metadata: Usage statistics
        """
        event_type = chunk.get("type")

        # Handle different event types
        if event_type == "contentBlockDelta":
            # Text delta events
            delta_data = chunk.get("delta", {})
            if "text" in delta_data:
                return chat.StreamResponsePayload(
                    id=None,
                    created=int(time.time()),
                    object="chat.completion.chunk",
                    model=config.model.name,
                    choices=[
                        chat.StreamChoice(
                            # Always use index=0 for choice index (assuming n=1)
                            # This is NOT the contentBlockIndex, which is used only internally
                            index=0,
                            finish_reason=None,
                            delta=chat.StreamDelta(
                                role="assistant",
                                content=delta_data["text"],
                            ),
                        )
                    ],
                )

        elif event_type == "messageStop":
            stop_reason = chunk.get("stopReason", "end_turn")
            finish_reason_map = {
                "end_turn": "stop",
                "max_tokens": "length",
                "stop_sequence": "stop",
                "content_filtered": "content_filter",
            }
            finish_reason = finish_reason_map.get(stop_reason, "stop")

            return chat.StreamResponsePayload(
                id=None,
                created=int(time.time()),
                object="chat.completion.chunk",
                model=config.model.name,
                choices=[
                    chat.StreamChoice(
                        index=0,
                        finish_reason=finish_reason,
                        delta=chat.StreamDelta(),
                    )
                ],
            )

        elif event_type == "metadata":
            # Usage metadata event - currently not converted to usage info in stream responses.
            # In the Converse API, metadata (including usage) arrives after messageStop,
            # which makes it difficult to include in the final content chunk.
            return None

        # For other events (messageStart, contentBlockStart, contentBlockStop), return None
        return None

    @classmethod
    def completions_to_model(cls, payload, config):
        raise NotImplementedError(
            "Converse API does not support completions. Use chat endpoint instead."
        )

    @classmethod
    def model_to_completions(cls, resp, config):
        raise NotImplementedError(
            "Converse API does not support completions. Use chat endpoint instead."
        )

    @classmethod
    def embeddings_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    def model_to_embeddings(cls, resp, config):
        raise NotImplementedError


class AmazonBedrockModelProvider(Enum):
    AMAZON = "amazon"
    COHERE = "cohere"
    AI21 = "ai21"
    ANTHROPIC = "anthropic"

    @property
    def adapter_class(self) -> type[ProviderAdapter]:
        return AWS_MODEL_PROVIDER_TO_ADAPTER.get(self)

    @classmethod
    def of_str(cls, name: str):
        name = name.lower()

        for opt in cls:
            if opt.name.lower() in name or opt.value.lower() in name:
                return opt


AWS_MODEL_PROVIDER_TO_ADAPTER = {
    AmazonBedrockModelProvider.COHERE: CohereAdapter,
    AmazonBedrockModelProvider.ANTHROPIC: AmazonBedrockAnthropicAdapter,
    AmazonBedrockModelProvider.AMAZON: AWSTitanAdapter,
    AmazonBedrockModelProvider.AI21: AI21Adapter,
}


class AmazonBedrockProvider(BaseProvider):
    NAME = "Amazon Bedrock"
    CONFIG_TYPE = AmazonBedrockConfig

    def __init__(self, config: EndpointConfig):
        super().__init__(config)

        if config.model.config is None or not isinstance(config.model.config, AmazonBedrockConfig):
            raise TypeError(f"Invalid config type {config.model.config}")
        self.bedrock_config: AmazonBedrockConfig = config.model.config
        self._client = None
        self._client_created = 0

    def _client_expired(self):
        if not isinstance(self.bedrock_config.aws_config, AWSRole):
            return False

        return (
            (time.monotonic_ns() - self._client_created)
            >= (self.bedrock_config.aws_config.session_length_seconds) * 1_000_000_000,
        )

    def get_bedrock_client(self):
        import boto3
        import botocore.exceptions

        if self._client is not None and not self._client_expired():
            return self._client

        session = boto3.Session(**self._construct_session_args())

        try:
            self._client = session.client(
                service_name="bedrock-runtime", **self._construct_client_args(session)
            )
            self._client_created = time.monotonic_ns()
            return self._client
        except botocore.exceptions.UnknownServiceError as e:
            raise AIGatewayConfigException(
                "Cannot create Amazon Bedrock client; ensure boto3/botocore "
                "linked from the Amazon Bedrock user guide are installed. "
                "Otherwise likely missing credentials or accessing account without to "
                "Amazon Bedrock Private Preview"
            ) from e

    def _construct_session_args(self):
        session_args = {
            "region_name": self.bedrock_config.aws_config.aws_region,
        }

        return {k: v for k, v in session_args.items() if v}

    def _construct_client_args(self, session):
        aws_config = self.bedrock_config.aws_config

        if isinstance(aws_config, AWSRole):
            role = session.client(service_name="sts").assume_role(
                RoleArn=aws_config.aws_role_arn,
                RoleSessionName="ai-gateway-bedrock",
                DurationSeconds=aws_config.session_length_seconds,
            )
            return {
                "aws_access_key_id": role["Credentials"]["AccessKeyId"],
                "aws_secret_access_key": role["Credentials"]["SecretAccessKey"],
                "aws_session_token": role["Credentials"]["SessionToken"],
            }
        elif isinstance(aws_config, AWSIdAndKey):
            return {
                "aws_access_key_id": aws_config.aws_access_key_id,
                "aws_secret_access_key": aws_config.aws_secret_access_key,
                "aws_session_token": aws_config.aws_session_token,
            }
        else:
            # TODO: handle session token authentication
            return {}

    @property
    def _underlying_provider(self):
        if (not self.config.model.name) or "." not in self.config.model.name:
            return None
        return AmazonBedrockModelProvider.of_str(self.config.model.name)

    @property
    def adapter_class(self) -> type[ProviderAdapter]:
        provider = self._underlying_provider
        if not provider:
            raise AIGatewayException(
                status_code=422,
                detail=f"Unknown Amazon Bedrock model type {self._underlying_provider}",
            )
        adapter = provider.adapter_class
        if not adapter:
            raise AIGatewayException(
                status_code=422,
                detail=f"Don't know how to handle {self._underlying_provider} for Amazon Bedrock",
            )
        return adapter

    def _request(self, body):
        import botocore.exceptions

        try:
            response = self.get_bedrock_client().invoke_model(
                body=json.dumps(body).encode(),
                modelId=self.config.model.name,
                # defaults
                # save=False,
                accept="application/json",
                contentType="application/json",
            )
            return json.loads(response.get("body").read())

        # TODO work though botocore.exceptions to make this catchable.
        # except botocore.exceptions.ValidationException as e:
        #     raise HTTPException(status_code=422, detail=str(e)) from e

        except botocore.exceptions.ReadTimeoutError as e:
            raise AIGatewayException(status_code=408) from e

    def _converse_request(self, **kwargs):
        """
        Make a request using the Bedrock Converse API.
        """
        import botocore.exceptions

        try:
            return self.get_bedrock_client().converse(modelId=self.config.model.name, **kwargs)
        except botocore.exceptions.ReadTimeoutError as e:
            raise AIGatewayException(status_code=408) from e

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        self.check_for_model_field(payload)
        payload = jsonable_encoder(payload, exclude_none=True, exclude_defaults=True)
        payload = self.adapter_class.completions_to_model(payload, self.config)
        response = self._request(payload)
        return self.adapter_class.model_to_completions(response, self.config)

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        """
        Chat completion using Bedrock Converse API.
        """
        from fastapi.encoders import jsonable_encoder

        self.check_for_model_field(payload)
        payload = jsonable_encoder(payload, exclude_none=True, exclude_defaults=True)

        # Transform to Converse API format
        converse_payload = ConverseAdapter.chat_to_model(payload, self.config)

        # Make the request
        response = self._converse_request(**converse_payload)

        # Transform response back to MLflow format
        return ConverseAdapter.model_to_chat(response, self.config)

    async def chat_stream(
        self, payload: chat.RequestPayload
    ) -> AsyncIterable[chat.StreamResponsePayload]:
        """
        Streaming chat completion using Bedrock Converse Stream API.

        The converse_stream API returns an event stream with the following event types:
        - messageStart: Beginning of the response
        - contentBlockStart: Start of a content block (text)
        - contentBlockDelta: Incremental content chunks
        - contentBlockStop: End of a content block
        - messageStop: End of message with stop reason
        - metadata: Usage statistics
        """
        from fastapi.encoders import jsonable_encoder

        self.check_for_model_field(payload)
        payload = jsonable_encoder(payload, exclude_none=True, exclude_defaults=True)

        if payload.get("stream_options", {}).get("include_usage"):
            raise AIGatewayException(
                status_code=422,
                detail="stream_options.include_usage is not supported for Bedrock streaming",
            )

        # Transform to Converse API format (same as non-streaming)
        converse_payload = ConverseAdapter.chat_streaming_to_model(payload, self.config)

        # Make streaming request
        try:
            response = self.get_bedrock_client().converse_stream(
                modelId=self.config.model.name, **converse_payload
            )
        except Exception as e:
            raise AIGatewayException(
                status_code=500,
                detail=f"Bedrock converse_stream request failed: {e}",
            ) from e

        # Parse event stream
        event_stream = response.get("stream")
        if not event_stream:
            raise AIGatewayException(
                status_code=500,
                detail="No event stream returned from converse_stream",
            )

        for event in event_stream:
            # Extract event type and data
            if "messageStart" in event:
                # Message start event - no data to yield
                continue

            elif "contentBlockStart" in event:
                # Content block start - no data to yield for text blocks
                continue

            elif "contentBlockDelta" in event:
                # Content delta event - yield chunk
                delta_data = event["contentBlockDelta"]

                chunk = {
                    "type": "contentBlockDelta",
                    "delta": delta_data.get("delta", {}),
                }
                if stream_chunk := ConverseAdapter.model_to_chat_streaming(chunk, self.config):
                    yield stream_chunk

            elif "contentBlockStop" in event:
                # Content block end - no data to yield
                continue

            elif "messageStop" in event:
                # Message stop event with stop reason
                stop_data = event["messageStop"]
                chunk = {
                    "type": "messageStop",
                    "stopReason": stop_data.get("stopReason", "end_turn"),
                }
                if stream_chunk := ConverseAdapter.model_to_chat_streaming(chunk, self.config):
                    yield stream_chunk

            elif "metadata" in event:
                # Metadata event with usage statistics
                continue
