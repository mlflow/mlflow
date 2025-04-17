import json
import time
from typing import Any, AsyncGenerator, AsyncIterable

from mlflow.gateway.config import CohereConfig, RouteConfig
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.utils import rename_payload_keys, send_request, send_stream_request
from mlflow.gateway.schemas import chat, completions, embeddings


class CohereAdapter(ProviderAdapter):
    @staticmethod
    def _scale_temperature(payload):
        # The range of Cohere's temperature is 0-5, but ours is 0-2, so we scale it.
        if temperature := payload.get("temperature"):
            payload["temperature"] = 2.5 * temperature
        return payload

    @classmethod
    def model_to_completions(cls, resp, config):
        # Response example (https://docs.cohere.com/reference/generate)
        # ```
        # {
        #   "id": "string",
        #   "generations": [
        #     {
        #       "id": "string",
        #       "text": "string"
        #     }
        #   ],
        #   "prompt": "string"
        # }
        # ```
        return completions.ResponsePayload(
            created=int(time.time()),
            object="text_completion",
            model=config.model.name,
            choices=[
                completions.Choice(
                    index=idx,
                    text=c["text"],
                    finish_reason=None,
                )
                for idx, c in enumerate(resp["generations"])
            ],
            usage=completions.CompletionsUsage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            ),
        )

    @classmethod
    def model_to_completions_streaming(cls, resp, config):
        # Response example (https://docs.cohere.com/reference/generate)
        #
        # Streaming chunks:
        # ```
        # {"index":0,"text":" Hi","is_finished":false,"event_type":"text-generation"}
        # ```
        # ```
        # {"index":1,"text":" Hi","is_finished":false,"event_type":"text-generation"}
        # ```
        # notes: "index" is only present if "num_generations" > 1
        #
        # Final chunk:
        # ```
        # {"is_finished":true,"event_type":"stream-end","finish_reason":"COMPLETE",
        #   "response":{"id":"b32a70c5-8c91-4f96-958f-d942801ed22f",
        #       "generations":[
        #           {
        #               "id":"5d5d0851-35ac-4c25-a9a9-2fbb391bd415",
        #               "index":0,
        #               "text":" Hi there! How can I assist you today? ",
        #               "finish_reason":"COMPLETE"
        #           },
        #           {
        #               "id":"0a24787f-504e-470e-a088-0bf801a2c72d",
        #               "index":1,
        #               "text":" Hi there, how can I assist you today? ",
        #               "finish_reason":"COMPLETE"
        #           }
        #       ],
        #       "prompt":"Hello"
        #   }}
        # ```
        response = resp.get("response")
        return completions.StreamResponsePayload(
            id=response["id"] if response else None,
            created=int(time.time()),
            model=config.model.name,
            choices=[
                completions.StreamChoice(
                    index=resp.get("index", 0),
                    finish_reason=resp.get("finish_reason"),
                    text=resp.get("text"),
                )
            ],
            usage=completions.CompletionsUsage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            ),
        )

    @classmethod
    def model_to_embeddings(cls, resp, config):
        # Response example (https://docs.cohere.com/reference/embed):
        # ```
        # {
        #   "id": "bc57846a-3e56-4327-8acc-588ca1a37b8a",
        #   "texts": [
        #     "hello world"
        #   ],
        #   "embeddings": [
        #     [
        #       3.25,
        #       0.7685547,
        #       2.65625,
        #       ...
        #       -0.30126953,
        #       -2.3554688,
        #       1.2597656
        #     ]
        #   ],
        #   "meta": [
        #     {
        #       "api_version": [
        #         {
        #           "version": "1"
        #         }
        #       ]
        #     }
        #   ]
        # }
        # ```
        return embeddings.ResponsePayload(
            data=[
                embeddings.EmbeddingObject(
                    embedding=output,
                    index=idx,
                )
                for idx, output in enumerate(resp["embeddings"])
            ],
            model=config.model.name,
            usage=embeddings.EmbeddingsUsage(
                prompt_tokens=None,
                total_tokens=None,
            ),
        )

    @classmethod
    def completions_to_model(cls, payload, config):
        key_mapping = {
            "stop": "stop_sequences",
            "n": "num_generations",
        }
        cls.check_keys_against_mapping(key_mapping, payload)
        payload = cls._scale_temperature(payload)
        return rename_payload_keys(payload, key_mapping)

    @classmethod
    def completions_streaming_to_model(cls, payload, config):
        return cls.completions_to_model(payload, config)

    @classmethod
    def embeddings_to_model(cls, payload, config):
        key_mapping = {"input": "texts"}
        for k1, k2 in key_mapping.items():
            if k2 in payload:
                raise AIGatewayException(
                    status_code=422, detail=f"Invalid parameter {k2}. Use {k1} instead."
                )
        return rename_payload_keys(payload, key_mapping)

    @classmethod
    def chat_to_model(cls, payload, config):
        if payload["n"] != 1:
            raise AIGatewayException(
                status_code=422,
                detail=f"Parameter n must be 1 for Cohere chat, got {payload['n']}.",
            )
        del payload["n"]

        if "stop" in payload:
            raise AIGatewayException(
                status_code=422,
                detail="Parameter stop is not supported for Cohere chat.",
            )
        payload = cls._scale_temperature(payload)

        messages = payload.pop("messages")
        last_message = messages.pop()  # pydantic enforces min_items=1
        if last_message["role"] != "user":
            raise AIGatewayException(
                status_code=422,
                detail=f"Last message must be from user, got {last_message['role']}.",
            )
        payload["message"] = last_message["content"]

        # Cohere uses `preamble_override` to set the system message
        # we concatenate all system messages from the user with a newline
        system_messages = [m for m in messages if m["role"] == "system"]
        if len(system_messages) > 0:
            payload["preamble_override"] = "\n".join(m["content"] for m in system_messages)

        # remaining messages are chat history
        # we want to include only user and assistant messages
        messages = [m for m in messages if m["role"] in ("user", "assistant")]
        if messages:
            payload["chat_history"] = [
                {
                    "role": "USER" if m["role"] == "user" else "CHATBOT",
                    "message": m["content"],
                }
                for m in messages
            ]
        return payload

    @classmethod
    def chat_streaming_to_model(cls, payload, config):
        return cls.chat_to_model(payload, config)

    @classmethod
    def model_to_chat(cls, resp, config):
        # Response example (https://docs.cohere.com/reference/chat)
        # ```
        # {
        #   "response_id": "string",
        #   "text": "string",
        #   "generation_id": "string",
        #   "token_count": {
        #     "prompt_tokens": 0,
        #     "response_tokens": 0,
        #     "total_tokens": 0,
        #     "billed_tokens": 0
        #   },
        #   "meta": {
        #     "api_version": {
        #       "version": "1"
        #     },
        #     "billed_units": {
        #       "input_tokens": 0,
        #       "output_tokens": 0
        #     }
        #   },
        #   "tool_inputs": null
        # }
        # ```
        return chat.ResponsePayload(
            id=resp["response_id"],
            object="chat.completion",
            created=int(time.time()),
            model=config.model.name,
            choices=[
                chat.Choice(
                    index=0,
                    message=chat.ResponseMessage(
                        role="assistant",
                        content=resp["text"],
                    ),
                    finish_reason=None,
                ),
            ],
            usage=chat.ChatUsage(
                prompt_tokens=resp["token_count"]["prompt_tokens"],
                completion_tokens=resp["token_count"]["response_tokens"],
                total_tokens=resp["token_count"]["total_tokens"],
            ),
        )

    @classmethod
    def model_to_chat_streaming(cls, resp, config):
        # Response example (https://docs.cohere.com/reference/chat)
        # Streaming chunks:
        # ```
        # {
        #   "is_finished":false,
        #   "event_type":"stream-start",
        #   "generation_id":"string"
        # }
        # {"is_finished":false,"event_type":"text-generation","text":"How"}
        # {"is_finished":false,"event_type":"text-generation","text":" are"}
        # {"is_finished":false,"event_type":"text-generation","text":" you"}
        # {
        #   "is_finished":true,
        #   "event_type":"stream-end",
        #   "response":{
        #     "response_id":"string",
        #     "text":"How are you",
        #     "generation_id":"string",
        #     "token_count":{
        #       "prompt_tokens":83,"response_tokens":63,"total_tokens":146,"billed_tokens":128
        #     },
        #     "tool_inputs":null
        #   },
        #   "finish_reason":"COMPLETE"
        # }
        # ```
        response = resp.get("response")
        return chat.StreamResponsePayload(
            # first chunk has "generation_id" but not "response_id"
            id=response["response_id"] if response else None,
            created=int(time.time()),
            model=config.model.name,
            choices=[
                chat.StreamChoice(
                    index=0,
                    finish_reason=resp.get("finish_reason"),
                    delta=chat.StreamDelta(
                        role=None,
                        content=resp.get("text"),
                    ),
                )
            ],
            usage=chat.ChatUsage(
                prompt_tokens=response["token_count"]["prompt_tokens"] if response else None,
                completion_tokens=response["token_count"]["response_tokens"] if response else None,
                total_tokens=response["token_count"]["total_tokens"] if response else None,
            ),
        )


class CohereProvider(BaseProvider):
    NAME = "Cohere"
    CONFIG_TYPE = CohereConfig

    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, CohereConfig):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.cohere_config: CohereConfig = config.model.config

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.cohere_config.cohere_api_key}"}

    @property
    def base_url(self) -> str:
        return "https://api.cohere.ai/v1"

    @property
    def adapter_class(self) -> type[ProviderAdapter]:
        return CohereAdapter

    def get_endpoint_url(self, route_type: str) -> str:
        if route_type == "llm/v1/chat":
            return f"{self.base_url}/chat"
        elif route_type == "llm/v1/completions":
            return f"{self.base_url}/generate"
        elif route_type == "llm/v1/embeddings":
            return f"{self.base_url}/embed"
        else:
            raise ValueError(f"Invalid route type {route_type}")

    async def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await send_request(
            headers=self.headers,
            base_url=self.base_url,
            path=path,
            payload=payload,
        )

    def _stream_request(self, path: str, payload: dict[str, Any]) -> AsyncGenerator[bytes, None]:
        return send_stream_request(
            headers=self.headers,
            base_url=self.base_url,
            path=path,
            payload=payload,
        )

    async def chat_stream(
        self, payload: chat.RequestPayload
    ) -> AsyncIterable[chat.StreamResponsePayload]:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        stream = self._stream_request(
            "chat",
            {
                "model": self.config.model.name,
                **CohereAdapter.chat_streaming_to_model(payload, self.config),
            },
        )
        async for chunk in stream:
            if not chunk:
                continue

            resp = json.loads(chunk)
            if resp["event_type"] == "stream-start":
                continue
            yield CohereAdapter.model_to_chat_streaming(resp, self.config)

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        resp = await self._request(
            "chat",
            {
                "model": self.config.model.name,
                **CohereAdapter.chat_to_model(payload, self.config),
            },
        )
        return CohereAdapter.model_to_chat(resp, self.config)

    async def completions_stream(
        self, payload: completions.RequestPayload
    ) -> AsyncIterable[completions.StreamResponsePayload]:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        stream = self._stream_request(
            "generate",
            {
                "model": self.config.model.name,
                **CohereAdapter.completions_streaming_to_model(payload, self.config),
            },
        )
        async for chunk in stream:
            if not chunk:
                continue

            resp = json.loads(chunk)
            yield CohereAdapter.model_to_completions_streaming(resp, self.config)

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        resp = await self._request(
            "generate",
            {
                "model": self.config.model.name,
                **CohereAdapter.completions_to_model(payload, self.config),
            },
        )
        return CohereAdapter.model_to_completions(resp, self.config)

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        resp = await self._request(
            "embed",
            {
                "model": self.config.model.name,
                **CohereAdapter.embeddings_to_model(payload, self.config),
            },
        )
        return CohereAdapter.model_to_embeddings(resp, self.config)
