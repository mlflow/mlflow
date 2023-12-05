import time
from typing import Any, Dict

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import CohereConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.utils import rename_payload_keys, send_request
from mlflow.gateway.schemas import chat, completions, embeddings


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
    def embeddings_to_model(cls, payload, config):
        key_mapping = {"input": "texts"}
        for k1, k2 in key_mapping.items():
            if k2 in payload:
                raise HTTPException(
                    status_code=422, detail=f"Invalid parameter {k2}. Use {k1} instead."
                )
        return rename_payload_keys(payload, key_mapping)


class CohereProvider(BaseProvider):
    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, CohereConfig):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.cohere_config: CohereConfig = config.model.config

    async def _request(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.cohere_config.cohere_api_key}"}
        return await send_request(
            headers=headers,
            base_url="https://api.cohere.ai/v1",
            path=path,
            payload=payload,
        )

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        raise HTTPException(status_code=422, detail="The chat route is not available for Cohere.")

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
