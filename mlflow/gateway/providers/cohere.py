from typing import Dict, Any

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from .base import BaseProvider
from ..schemas import completions, embeddings
from ..config import CohereConfig, RouteConfig


class CohereProvider(BaseProvider):
    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, CohereConfig):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.openai_config: CohereConfig = config.model.config

    async def _request(self, path: str, payload: Dict[str, Any]):
        import aiohttp

        headers = {"Authorization": f"Bearer {self.openai_config.openai_api_key}"}
        if org := self.openai_config.openai_organization:
            headers["OpenAI-Organization"] = org
        async with aiohttp.ClientSession(headers=headers) as session:
            url = "/".join([self.openai_config.openai_api_base.rstrip("/"), path.lstrip("/")])
            async with session.post(url, json=payload) as response:
                js = await response.json()
                try:
                    response.raise_for_status()
                except aiohttp.ClientResponseError as e:
                    raise HTTPException(status_code=e.status, detail=js)
                return js

    @staticmethod
    def _make_payload(payload: Dict[str, Any], mapping: Dict[str, str]):
        res = {}
        for k, v in payload.items():
            if new_k := mapping.get(k):
                res[new_k] = v
            else:
                res[k] = v
        return res

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        payload = jsonable_encoder(payload)
        key_mapping = {
            "stop": "stop_sequences",
            "candidate_count": "num_generations",
        }
        for _k1, k2 in key_mapping.items():
            if k2 in payload:
                raise HTTPException(
                    status_code=400, detail=f"Invalid parameter {k2}. Use {k2} instead."
                )
        payload = CohereProvider._make_payload(
            payload,
            key_mapping,
        )
        # The range of Cohere's temperature is 0-5, but ours is 0-1, so we double it.
        payload["temperature"] = 5 * payload["temperature"]
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
        payload = jsonable_encoder(payload)
        key_mapping = {"text": "texts"}
        for _k1, k2 in key_mapping.items():
            if k2 in payload:
                raise HTTPException(
                    status_code=400, detail=f"Invalid parameter {k2}. Use {k2} instead."
                )
        payload = CohereProvider._make_payload(
            payload,
            key_mapping,
        )
        resp = await self._request(
            "embeddings",
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
