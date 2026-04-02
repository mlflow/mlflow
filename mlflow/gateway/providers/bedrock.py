import asyncio
import json
import queue
import time
from enum import Enum
from typing import Any, AsyncIterable

from mlflow.gateway.config import (
    AmazonBedrockConfig,
    AWSBearerToken,
    AWSIdAndKey,
    AWSRole,
    EndpointConfig,
)
from mlflow.gateway.constants import (
    MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS,
)
from mlflow.gateway.exceptions import AIGatewayConfigException, AIGatewayException
from mlflow.gateway.providers.anthropic import AnthropicAdapter
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.cohere import CohereAdapter
from mlflow.gateway.providers.utils import rename_payload_keys, send_request
from mlflow.gateway.schemas import chat, completions, embeddings

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
    DISPLAY_NAME = "Amazon Bedrock"
    CONFIG_TYPE = AmazonBedrockConfig

    def get_provider_name(self) -> str:
        return "bedrock"

    def __init__(self, config: EndpointConfig, enable_tracing: bool = False):
        super().__init__(config, enable_tracing=enable_tracing)

        if config.model.config is None or not isinstance(config.model.config, AmazonBedrockConfig):
            raise TypeError(f"Invalid config type {config.model.config}")
        self.bedrock_config: AmazonBedrockConfig = config.model.config
        self._client = None
        self._client_created = 0

    @property
    def _is_token_auth(self) -> bool:
        return isinstance(self.bedrock_config.aws_config, AWSBearerToken)

    def _client_expired(self):
        if not isinstance(self.bedrock_config.aws_config, AWSRole):
            return False

        return (
            (time.monotonic_ns() - self._client_created)
            >= (self.bedrock_config.aws_config.session_length_seconds) * 1_000_000_000,
        )

    def get_bedrock_client(self):
        try:
            import boto3
            import botocore.exceptions
        except ImportError:
            raise ImportError("Bedrock provider requires boto3. Install it with: pip install boto3")

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

    # ---- Converse API helpers ----

    def _build_converse_kwargs(
        self, messages: list[dict[str, Any]], payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Build kwargs for the Converse API call."""
        system_prompts = []
        converse_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Normalize content: extract text from list of content parts
            if isinstance(content, list):
                content = "\n".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )

            if role == "system":
                system_prompts.append({"text": content})
            elif role == "assistant":
                converse_messages.append({
                    "role": "assistant",
                    "content": [{"text": content}],
                })
            elif role == "tool":
                converse_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": msg.get("tool_call_id", ""),
                                "content": [{"text": content}],
                            }
                        }
                    ],
                })
            else:
                converse_messages.append({
                    "role": "user",
                    "content": [{"text": content}],
                })

        kwargs = {
            "modelId": self.config.model.name,
            "messages": converse_messages,
        }

        if system_prompts:
            kwargs["system"] = system_prompts

        # Build inferenceConfig from OpenAI params
        inference_config = {}
        if "temperature" in payload:
            inference_config["temperature"] = payload["temperature"]
        if "top_p" in payload:
            inference_config["topP"] = payload["top_p"]
        if max_tokens := payload.get("max_tokens") or payload.get("max_completion_tokens"):
            inference_config["maxTokens"] = max_tokens
        if "stop" in payload:
            stop = payload["stop"]
            inference_config["stopSequences"] = stop if isinstance(stop, list) else [stop]

        if inference_config:
            kwargs["inferenceConfig"] = inference_config

        # Convert tools to Bedrock format
        if tools := payload.get("tools"):
            bedrock_tools = []
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool["function"]
                    bedrock_tools.append({
                        "toolSpec": {
                            "name": func["name"],
                            "description": func.get("description", ""),
                            "inputSchema": {"json": func.get("parameters", {})},
                        }
                    })
            if bedrock_tools:
                kwargs["toolConfig"] = {"tools": bedrock_tools}

        return kwargs

    def _converse_to_chat_response(self, response: dict[str, Any]) -> chat.ResponsePayload:
        """Convert Bedrock Converse response to OpenAI chat format."""
        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])

        text_parts = []
        tool_calls = []

        for block in content_blocks:
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block:
                tool_use = block["toolUse"]
                tool_calls.append(
                    chat.ToolCall(
                        id=tool_use.get("toolUseId", ""),
                        type="function",
                        function=chat.ToolCallFunction(
                            name=tool_use.get("name", ""),
                            arguments=json.dumps(tool_use.get("input", {})),
                        ),
                    )
                )

        usage = response.get("usage", {})
        finish_reason = self._map_stop_reason(response.get("stopReason", "end_turn"))

        return chat.ResponsePayload(
            created=int(time.time()),
            object="chat.completion",
            model=self.config.model.name,
            choices=[
                chat.Choice(
                    index=0,
                    message=chat.ResponseMessage(
                        role="assistant",
                        content="\n".join(text_parts) if text_parts else None,
                        tool_calls=tool_calls or None,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=chat.ChatUsage(
                prompt_tokens=usage.get("inputTokens"),
                completion_tokens=usage.get("outputTokens"),
                total_tokens=usage.get("totalTokens"),
            ),
        )

    # ---- API methods ----

    def _get_token_auth_base_url(self) -> str:
        region = self.bedrock_config.aws_config.aws_region
        if not region:
            raise AIGatewayException(
                status_code=400,
                detail="aws_region_name is required for Bedrock API key authentication. "
                "The API key can only be used in the AWS Region it was generated in.",
            )
        return f"https://bedrock-runtime.{region}.amazonaws.com"

    def _get_token_auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.bedrock_config.aws_config.aws_bearer_token}"}

    def _make_stream_chunk(
        self,
        delta: chat.StreamDelta,
        finish_reason: str | None = None,
        usage: chat.ChatUsage | None = None,
    ) -> chat.StreamResponsePayload:
        return chat.StreamResponsePayload(
            created=int(time.time()),
            model=self.config.model.name,
            choices=[chat.StreamChoice(index=0, finish_reason=finish_reason, delta=delta)],
            usage=usage,
        )

    @staticmethod
    def _map_stop_reason(stop_reason: str) -> str:
        match stop_reason:
            case "tool_use":
                return "tool_calls"
            case "max_tokens":
                return "length"
            case "content_filtered":
                return "content_filter"
            case _:
                return "stop"

    def _parse_stream_event(self, event: dict[str, Any]) -> chat.StreamResponsePayload | None:
        if "contentBlockStart" in event:
            start = event["contentBlockStart"].get("start", {})
            if "toolUse" in start:
                tool_use = start["toolUse"]
                return self._make_stream_chunk(
                    delta=chat.StreamDelta(
                        role=None,
                        content=None,
                        tool_calls=[
                            chat.ToolCallDelta(
                                index=event["contentBlockStart"].get("contentBlockIndex", 0),
                                id=tool_use.get("toolUseId"),
                                type="function",
                                function=chat.Function(
                                    name=tool_use.get("name"),
                                    arguments="",
                                ),
                            )
                        ],
                    ),
                )
        elif "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            if text := delta.get("text"):
                return self._make_stream_chunk(
                    delta=chat.StreamDelta(role=None, content=text),
                )
            elif "toolUse" in delta:
                tool_input = delta["toolUse"].get("input", "")
                if isinstance(tool_input, dict):
                    arguments = json.dumps(tool_input)
                else:
                    arguments = str(tool_input)
                return self._make_stream_chunk(
                    delta=chat.StreamDelta(
                        role=None,
                        content=None,
                        tool_calls=[
                            chat.ToolCallDelta(
                                index=event["contentBlockDelta"].get("contentBlockIndex", 0),
                                function=chat.Function(arguments=arguments),
                            )
                        ],
                    ),
                )
        elif "messageStop" in event:
            stop_reason = event["messageStop"].get("stopReason", "end_turn")
            return self._make_stream_chunk(
                delta=chat.StreamDelta(role=None, content=None),
                finish_reason=self._map_stop_reason(stop_reason),
            )
        elif "metadata" in event:
            if usage := event["metadata"].get("usage", {}):
                return self._make_stream_chunk(
                    delta=chat.StreamDelta(role=None, content=None),
                    usage=chat.ChatUsage(
                        prompt_tokens=usage.get("inputTokens"),
                        completion_tokens=usage.get("outputTokens"),
                        total_tokens=usage.get("totalTokens"),
                    ),
                )
        return None

    async def _chat_with_token(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        """Chat using bearer token auth via direct HTTP to Bedrock Converse API."""
        from fastapi.encoders import jsonable_encoder

        payload_dict = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload_dict)

        messages = payload_dict.pop("messages", [])
        kwargs = self._build_converse_kwargs(messages, payload_dict)
        model_id = kwargs.pop("modelId")

        resp = await send_request(
            headers=self._get_token_auth_headers(),
            base_url=self._get_token_auth_base_url(),
            path=f"model/{model_id}/converse",
            payload=kwargs,
        )
        return self._converse_to_chat_response(resp)

    async def _chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        if self._is_token_auth:
            return await self._chat_with_token(payload)

        from fastapi.encoders import jsonable_encoder

        payload_dict = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload_dict)

        messages = payload_dict.pop("messages", [])
        kwargs = self._build_converse_kwargs(messages, payload_dict)

        response = await asyncio.get_running_loop().run_in_executor(
            None, lambda: self.get_bedrock_client().converse(**kwargs)
        )

        return self._converse_to_chat_response(response)

    async def _chat_stream(
        self, payload: chat.RequestPayload
    ) -> AsyncIterable[chat.StreamResponsePayload]:
        if self._is_token_auth:
            raise AIGatewayException(
                status_code=501,
                detail="Streaming is not yet supported for Bedrock API key (bearer token) auth. "
                "Use a non-streaming chat request instead.",
            )

        from fastapi.encoders import jsonable_encoder

        payload_dict = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload_dict)

        messages = payload_dict.pop("messages", [])
        kwargs = self._build_converse_kwargs(messages, payload_dict)

        # Consume the synchronous boto3 EventStream in a thread to avoid
        # blocking the async event loop.
        event_queue: queue.Queue = queue.Queue()
        _SENTINEL = object()
        _stream_error: list[BaseException] = []

        def _consume_stream():
            try:
                response = self.get_bedrock_client().converse_stream(**kwargs)
                for event in response.get("stream", []):
                    event_queue.put(event)
            except BaseException as e:
                _stream_error.append(e)
            finally:
                event_queue.put(_SENTINEL)

        loop = asyncio.get_running_loop()
        # Fire-and-forget: stream is consumed in background thread,
        # events are read from the queue below. Not awaited intentionally.
        loop.run_in_executor(None, _consume_stream)

        while True:
            event = await loop.run_in_executor(None, event_queue.get)
            if event is _SENTINEL:
                if _stream_error:
                    raise _stream_error[0]
                break
            if chunk := self._parse_stream_event(event):
                yield chunk

    async def _embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        if self._is_token_auth:
            raise AIGatewayException(
                status_code=501,
                detail="Embeddings are not supported for Bedrock API key (bearer token) auth. "
                "Use access_keys or default_chain auth mode for embeddings.",
            )

        from fastapi.encoders import jsonable_encoder

        payload_dict = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload_dict)

        input_text = payload_dict.get("input", "")
        texts = input_text if isinstance(input_text, list) else [input_text]

        embedding_data = []
        total_tokens = 0

        for idx, text in enumerate(texts):
            body = {"inputText": text}
            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda b=body: json.loads(
                    self
                    .get_bedrock_client()
                    .invoke_model(
                        body=json.dumps(b).encode(),
                        modelId=self.config.model.name,
                        accept="application/json",
                        contentType="application/json",
                    )
                    .get("body")
                    .read()
                ),
            )

            embedding_data.append(
                embeddings.EmbeddingObject(
                    embedding=response.get("embedding", []),
                    index=idx,
                )
            )
            total_tokens += response.get("inputTextTokenCount", 0)

        return embeddings.ResponsePayload(
            data=embedding_data,
            model=self.config.model.name,
            usage=embeddings.EmbeddingsUsage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens,
            ),
        )

    async def _completions(
        self, payload: completions.RequestPayload
    ) -> completions.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        self.check_for_model_field(payload)
        payload = jsonable_encoder(payload, exclude_none=True, exclude_defaults=True)
        payload = self.adapter_class.completions_to_model(payload, self.config)
        response = self._request(payload)
        return self.adapter_class.model_to_completions(response, self.config)
