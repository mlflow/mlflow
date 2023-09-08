from typing import Any, Dict

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import CohereConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import rename_payload_keys, send_request
from mlflow.gateway.schemas import chat, completions, embeddings


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
        raise HTTPException(status_code=404, detail="The chat route is not available for Cohere.")

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        key_mapping = {
            "stop": "stop_sequences",
            "candidate_count": "num_generations",
        }
        for k1, k2 in key_mapping.items():
            if k2 in payload:
                raise HTTPException(
                    status_code=400, detail=f"Invalid parameter {k2}. Use {k1} instead."
                )
        # The range of Cohere's temperature is 0-5, but ours is 0-1, so we scale it.
        payload["temperature"] = 5 * payload["temperature"]
        payload = rename_payload_keys(payload, key_mapping)
        resp = await self._request(
            "generate",
            {"model": self.config.model.name, **payload},
        )
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
            **{
                "candidates": [
                    {
                        "text": c["text"],
                        "metadata": {},
                    }
                    for c in resp["generations"]
                ],
                "metadata": {
                    "model": self.config.model.name,
                    "route_type": self.config.route_type,
                },
            }
        )

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        key_mapping = {"text": "texts"}
        for k1, k2 in key_mapping.items():
            if k2 in payload:
                raise HTTPException(
                    status_code=400, detail=f"Invalid parameter {k2}. Use {k1} instead."
                )
        payload = rename_payload_keys(payload, key_mapping)
        resp = await self._request(
            "embed",
            {"model": self.config.model.name, **payload},
        )
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
            **{
                "embeddings": resp["embeddings"],
                "metadata": {
                    "model": self.config.model.name,
                    "route_type": self.config.route_type,
                },
            }
        )
