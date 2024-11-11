import json
from typing import Any, AsyncGenerator, AsyncIterable

from mlflow.exceptions import MlflowException
from mlflow.gateway.config import RouteConfig, TogetherAIConfig
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.utils import rename_payload_keys, send_request, send_stream_request
from mlflow.gateway.schemas import chat as chat_schema
from mlflow.gateway.schemas import completions as completions_schema
from mlflow.gateway.schemas import embeddings as embeddings_schema
from mlflow.gateway.utils import strip_sse_prefix


class TogetherAIAdapter(ProviderAdapter):
    @classmethod
    def model_to_embeddings(cls, resp, config):
        # Response example: (https://docs.together.ai/docs/embeddings-rest)
        # ```
        # {
        #  "object": "list",
        #  "data": [
        #    {
        #      "object": "embedding",
        #      "embedding": [
        #        0.44990748,
        #        -0.2521129,
        #        ...
        #        -0.43091708,
        #        0.214978
        #      ],
        #      "index": 0
        #    }
        #  ],
        #  "model": "togethercomputer/m2-bert-80M-8k-retrieval",
        #  "request_id": "840fc1b5bb2830cb-SEA"
        # }
        # ```
        return embeddings_schema.ResponsePayload(
            data=[
                embeddings_schema.EmbeddingObject(
                    embedding=item["embedding"],
                    index=item["index"],
                )
                for item in resp["data"]
            ],
            model=config.model.name,
            usage=embeddings_schema.EmbeddingsUsage(prompt_tokens=None, total_tokens=None),
        )

    @classmethod
    def model_to_completions(cls, resp, config):
        # Example response (https://docs.together.ai/reference/completions):
        # {
        #  "id": "8447f286bbdb67b3-SJC",
        #  "choices": [
        #    {
        #      "text": "Example text."
        #    }
        #  ],
        #  "usage": {
        #    "prompt_tokens": 16,
        #    "completion_tokens": 78,
        #    "total_tokens": 94
        #  },
        #  "created": 1705089226,
        #  "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        #  "object": "text_completion"
        # }

        return completions_schema.ResponsePayload(
            id=resp["id"],
            created=resp["created"],
            model=config.model.name,
            choices=[
                completions_schema.Choice(
                    index=idx,
                    text=c["text"],
                    finish_reason=None,
                )
                for idx, c in enumerate(resp["choices"])
            ],
            usage=completions_schema.CompletionsUsage(
                prompt_tokens=resp["usage"]["prompt_tokens"],
                completion_tokens=resp["usage"]["completion_tokens"],
                total_tokens=resp["usage"]["total_tokens"],
            ),
        )

    @classmethod
    def model_to_completions_streaming(cls, resp, config):
        # Response example (after manually calling API):
        #
        # {'id': '86d8d6e06df86f61-ATH', 'object': 'completion.chunk',
        # 'created': 1711977238, 'choices': [{'index': 0, 'text': '   ',
        # 'logprobs': None, 'finish_reason': None, 'delta': {'token_id': 2287, 'content': '   '}}],
        # 'model': 'mistralai/Mixtral-8x7B-v0.1', 'usage': None}
        #
        # {'id': '86d8d6e06df86f61-ATH', 'object': 'completion.chunk',
        # 'created': 1711977238, 'choices': [{'index': 0, 'text': ' "', 'logprobs': None,
        # 'finish_reason': None, 'delta': {'token_id': 345, 'content': ' "'}}],
        # 'model': 'mistralai/Mixtral-8x7B-v0.1', 'usage': None}
        #
        # "{'id': '86d8d6e06df86f61-ATH', 'object': 'completion.chunk',
        # 'created': 1711977238, 'choices': [{'index': 0, 'text': 'name', 'logprobs': None,
        # 'finish_reason': None, 'delta': {'token_id': 861, 'content': 'name'}}],
        # 'model': 'mistralai/Mixtral-8x7B-v0.1', 'usage': None}
        #
        # LAST CHUNK
        # {'id': '86d8d6e06df86f61-ATH', 'object': 'completion.chunk',
        # 'created': 1711977238, 'choices': [{'index': 0, 'text': '":', 'logprobs': None,
        # 'finish_reason': 'length', 'delta': {'token_id': 1264, 'content': '":'}}],
        # 'model': 'mistralai/Mixtral-8x7B-v0.1',
        # 'usage': {'prompt_tokens': 17, 'completion_tokens': 200, 'total_tokens': 217}}
        # ":[DONE]

        return completions_schema.StreamResponsePayload(
            id=resp.get("id"),
            created=resp.get("created"),
            model=config.model.name,
            choices=[
                completions_schema.StreamChoice(
                    index=idx,
                    # TODO this is questionable since the finish reason comes from togetherai api
                    finish_reason=choice.get("finish_reason"),
                    delta=completions_schema.StreamDelta(role=None, content=choice.get("text")),
                )
                for idx, choice in enumerate(resp.get("choices", []))
            ],
            # usage is not included in OpenAI StreamResponsePayload
        )

    @classmethod
    def completions_to_model(cls, payload, config):
        key_mapping = {
            # TogetherAI uses logprobs
            # OpenAI uses top_logprobs
            "top_logprobs": "logprobs"
        }

        # in openAI API the logprobs parameter
        # is a boolean flag.
        # Insert this here to prevent the user from mixing up the APIs
        logprobs_in_payload_condition = "logprobs" in payload and not isinstance(
            payload["logprobs"], int
        )

        if logprobs_in_payload_condition:
            raise AIGatewayException(
                status_code=422,
                detail="Wrong type for logprobs. It should be an 32bit integer.",
            )

        openai_top_logprobs_in_payload_condition = "top_logprobs" in payload and not isinstance(
            payload["top_logprobs"], int
        )

        if openai_top_logprobs_in_payload_condition:
            raise AIGatewayException(
                status_code=422,
                detail="Wrong type for top_logprobs. It should a 32bit integer.",
            )

        payload = rename_payload_keys(payload, key_mapping)
        return {"model": config.model.name, **payload}

    @classmethod
    def completions_streaming_to_model(cls, payload, config):
        # parameters for streaming completions are the same as the standard completions
        return TogetherAIAdapter.completions_to_model(payload, config)

    @classmethod
    def model_to_chat(cls, resp, config):
        # Example response (https://docs.together.ai/reference/chat-completions):
        # {
        #   "id": "8448080b880415ea-SJC",
        #   "choices": [
        #    {
        #        "message": {
        #           "role": "assistant",
        #           "content": "example"
        #         }
        #     }
        #   ],
        #   "usage": {
        #     "prompt_tokens": 31,
        #     "completion_tokens": 455,
        #     "total_tokens": 486
        #   },
        #   "created": 1705090115,
        #   "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        #   "object": "chat.completion"
        # }
        return chat_schema.ResponsePayload(
            id=resp["id"],
            object="chat.completion",
            created=resp["created"],
            model=config.model.name,
            choices=[
                chat_schema.Choice(
                    index=idx,
                    message=chat_schema.ResponseMessage(
                        role="assistant",
                        content=c["message"]["content"],
                    ),
                    finish_reason=None,
                )
                for idx, c in enumerate(resp["choices"])
            ],
            usage=chat_schema.ChatUsage(
                prompt_tokens=resp["usage"]["prompt_tokens"],
                completion_tokens=resp["usage"]["completion_tokens"],
                total_tokens=resp["usage"]["total_tokens"],
            ),
        )

    @classmethod
    def model_to_chat_streaming(cls, resp, config):
        # Response example (after running API manually):
        #
        # {'id': '86f2cfd18f6b38ca-ATH', 'object': 'chat.completion.chunk',
        # 'created': 1712249578, 'choices': [{'index': 0, 'text': ' The', 'logprobs': None,
        # 'finish_reason': None, 'delta': {'token_id': 415, 'content': ' The'}}],
        # 'model': 'mistralai/Mixtral-8x7B-v0.1', 'usage': None}
        #
        # {'id': '86f2cfd18f6b38ca-ATH', 'object': 'chat.completion.chunk',
        # 'created': 1712249578, 'choices': [{'index': 0, 'text': ' City', 'logprobs': None,
        # 'finish_reason': None, 'delta': {'token_id': 3805, 'content': ' City'}}],
        # 'model': 'mistralai/Mixtral-8x7B-v0.1', 'usage': None}
        #
        # {'id': '86f2cfd18f6b38ca-ATH', 'object': 'chat.completion.chunk',
        # 'created': 1712249578, 'choices': [{'index': 0, 'text': ' of', 'logprobs': None,
        # 'finish_reason': None, 'delta': {'token_id': 302, 'content': ' of'}}],
        # 'model': 'mistralai/Mixtral-8x7B-v0.1', 'usage': None}
        #
        # LAST CHUNK
        # {'id': '86f2cfd18f6b38ca-ATH', 'object': 'chat.completion.chunk',
        # 'created': 1712249578, 'choices': [{'index': 0, 'text': ' Paris', 'logprobs': None,
        # 'finish_reason': 'length', 'delta': {'token_id': 5465, 'content': ' Paris'}}],
        # 'model': 'mistralai/Mixtral-8x7B-v0.1',
        # 'usage': {'prompt_tokens': 93, 'completion_tokens': 100, 'total_tokens': 193}}

        return chat_schema.StreamResponsePayload(
            id=resp["id"],
            model=config.model.name,
            object="chat.completion.chunk",
            created=resp["created"],
            choices=[
                chat_schema.StreamChoice(
                    index=idx,
                    finish_reason=choice.get("finish_reason"),
                    delta=chat_schema.StreamDelta(
                        role=None,
                        content=choice.get("text"),
                    ),
                )
                # Added enumerate and a default empty list
                for idx, choice in enumerate(resp.get("choices", []))
            ],
            usage=resp.get("usage"),
        )

    @classmethod
    def chat_to_model(cls, payload, config):
        # completions and chat endpoint contain the same parameters
        return TogetherAIAdapter.completions_to_model(payload, config)

    @classmethod
    def chat_streaming_to_model(cls, payload, config):
        # streaming and standard chat contain the same parameters
        return TogetherAIAdapter.chat_to_model(payload, config)

    @classmethod
    def embeddings_to_model(cls, payload, config):
        # Example request (https://docs.together.ai/reference/embeddings):
        # curl --request POST \
        #   --url https://api.together.xyz/v1/embeddings \
        #   --header 'accept: application/json' \
        #   --header 'content-type: application/json' \
        #   --data '
        #   {
        #     "model": "togethercomputer/m2-bert-80M-8k-retrieval",
        #     "input": "Our solar system orbits the Milky Way galaxy at about 515,000 mph"
        #   }

        # This is just to keep the interface consistent the adapter
        # class is not needed here as the togetherai request similar
        # to the openAI one.

        return payload


class TogetherAIProvider(BaseProvider):
    NAME = "TogetherAI"
    CONFIG_TYPE = TogetherAIConfig

    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, TogetherAIConfig):
            # Should be unreachable
            raise MlflowException.invalid_parameter_value(
                f"Invalid config type {config.model.config}"
            )
        self.togetherai_config: TogetherAIConfig = config.model.config

    @property
    def base_url(self):
        # togetherai seems to support only this url
        return "https://api.together.xyz/v1"

    @property
    def headers(self):
        return {"Authorization": f"Bearer {self.togetherai_config.togetherai_api_key}"}

    @property
    def adapter_class(self) -> type[ProviderAdapter]:
        return TogetherAIAdapter

    def get_endpoint_url(self, route_type: str) -> str:
        if route_type == "llm/v1/chat":
            return f"{self.base_url}/chat/completions"
        elif route_type == "llm/v1/completions":
            return f"{self.base_url}/completions"
        elif route_type == "llm/v1/embeddings":
            return f"{self.base_url}/embeddings"
        else:
            raise ValueError(f"Invalid route type {route_type}")

    async def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await send_request(
            headers=self.headers,
            base_url=self.base_url,
            path=path,
            payload=payload,
        )

    async def _stream_request(
        self, path: str, payload: dict[str, Any]
    ) -> AsyncGenerator[bytes, None]:
        return send_stream_request(
            headers=self.headers,
            base_url=self.base_url,
            path=path,
            payload=payload,
        )

    async def embeddings(
        self, payload: embeddings_schema.RequestPayload
    ) -> embeddings_schema.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)

        resp = await self._request(
            path="embeddings",
            payload=TogetherAIAdapter.embeddings_to_model(payload, self.config),
        )

        return TogetherAIAdapter.model_to_embeddings(resp, self.config)

    async def completions_stream(
        self, payload: completions_schema.RequestPayload
    ) -> AsyncIterable[completions_schema.StreamResponsePayload]:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)

        if not payload.get("max_tokens"):
            raise AIGatewayException(
                status_code=422,
                detail=(
                    "max_tokens is not present in payload."
                    "It is a required parameter for TogetherAI completions."
                ),
            )

        stream = await self._stream_request(
            path="completions",
            payload=TogetherAIAdapter.completions_streaming_to_model(payload, self.config),
        )

        async for chunk in stream:
            chunk = chunk.strip()
            if not chunk:
                continue

            chunk = strip_sse_prefix(chunk.decode("utf-8"))
            if chunk == "[DONE]":
                return

            resp = json.loads(chunk)
            yield TogetherAIAdapter.model_to_completions_streaming(resp, self.config)

    async def completions(
        self, payload: completions_schema.RequestPayload
    ) -> completions_schema.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)

        if not payload.get("max_tokens"):
            raise AIGatewayException(
                status_code=422,
                detail=(
                    "max_tokens is not present in payload."
                    "It is a required parameter for TogetherAI completions."
                ),
            )

        resp = await self._request(
            path="completions", payload=TogetherAIAdapter.completions_to_model(payload, self.config)
        )

        return TogetherAIAdapter.model_to_completions(resp, self.config)

    async def chat_stream(self, payload: chat_schema.RequestPayload) -> chat_schema.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)

        stream = await self._stream_request(
            path="chat/completions",
            payload=TogetherAIAdapter.chat_streaming_to_model(payload, self.config),
        )

        async for chunk in stream:
            chunk = chunk.strip()
            if not chunk:
                continue

            chunk = strip_sse_prefix(chunk.decode("utf-8"))
            if chunk == "[DONE]":
                return

            resp = json.loads(chunk)
            yield TogetherAIAdapter.model_to_chat_streaming(resp, self.config)

    async def chat(self, payload: chat_schema.RequestPayload) -> chat_schema.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)

        resp = await self._request(
            path="chat/completions",
            payload=TogetherAIAdapter.chat_to_model(payload, self.config),
        )

        return TogetherAIAdapter.model_to_chat(resp, self.config)
