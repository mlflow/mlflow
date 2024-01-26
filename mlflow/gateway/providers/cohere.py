import json
import time
from typing import Any, AsyncGenerator, AsyncIterable, Dict

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import CohereConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.utils import rename_payload_keys, send_request, send_stream_request
from mlflow.gateway.schemas import completions, embeddings


class CohereAdapter(ProviderAdapter):
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
                    delta=completions.StreamDelta(
                        role=None,
                        content=resp.get("text"),
                    ),
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
        # The range of Cohere's temperature is 0-5, but ours is 0-2, so we scale it.
        if "temperature" in payload:
            payload["temperature"] = 2.5 * payload["temperature"]
        return rename_payload_keys(payload, key_mapping)

    @classmethod
    def completions_streaming_to_model(cls, payload, config):
        return cls.completions_to_model(payload, config)

    @classmethod
    def embeddings_to_model(cls, payload, config):
        key_mapping = {"input": "texts"}
        for k1, k2 in key_mapping.items():
            if k2 in payload:
                raise HTTPException(
                    status_code=422, detail=f"Invalid parameter {k2}. Use {k1} instead."
                )
        return rename_payload_keys(payload, key_mapping)


class CohereProvider(BaseProvider):
    NAME = "Cohere"

    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, CohereConfig):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.cohere_config: CohereConfig = config.model.config

    @property
    def auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.cohere_config.cohere_api_key}"}

    @property
    def base_url(self) -> str:
        return "https://api.cohere.ai/v1"

    async def _request(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await send_request(
            headers=self.auth_headers,
            base_url=self.base_url,
            path=path,
            payload=payload,
        )

    def _stream_request(self, path: str, payload: Dict[str, Any]) -> AsyncGenerator[bytes, None]:
        return send_stream_request(
            headers=self.auth_headers,
            base_url=self.base_url,
            path=path,
            payload=payload,
        )

    async def completions_stream(
        self, payload: completions.RequestPayload
    ) -> AsyncIterable[completions.StreamResponsePayload]:
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
