from typing import Dict, Any

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import send_request, rename_payload_keys
from mlflow.gateway.schemas import chat, completions, embeddings
from mlflow.gateway.config import MosaicMLConfig, RouteConfig


class MosaicMLProvider(BaseProvider):
    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, MosaicMLConfig):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.mosaicml_config: MosaicMLConfig = config.model.config

    async def _request(self, model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Authorization": f"{self.mosaicml_config.mosaicml_api_key}"}
        return await send_request(
            headers=headers,
            base_url="https://models.hosted-on.mosaicml.hosting",
            path=model + "/v1/predict",
            payload=payload,
        )

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        raise HTTPException(status_code=404, detail="The chat route is not available for MosaicML.")

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        key_mapping = {
            "max_tokens": "max_new_tokens",
        }
        for k1, k2 in key_mapping.items():
            if k2 in payload:
                raise HTTPException(
                    status_code=400, detail=f"Invalid parameter {k2}. Use {k1} instead."
                )
        payload = rename_payload_keys(payload, key_mapping)
        resp = await self._request(
            self.config.model.name,
            payload,
        )
        # Response example (https://docs.mosaicml.com/en/latest/inference.html#text-completion-models)
        # ```
        # {
        #   "outputs": [
        #     "string",
        #   ],
        # }
        # ```
        return completions.ResponsePayload(
            **{
                "candidates": [
                    {
                        "text": c,
                        "metadata": {},
                    }
                    for c in resp["outputs"]
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
        key_mapping = {"text": "inputs"}
        for k1, k2 in key_mapping.items():
            if k2 in payload:
                raise HTTPException(
                    status_code=400, detail=f"Invalid parameter {k2}. Use {k1} instead."
                )
        payload = rename_payload_keys(payload, key_mapping)
        resp = await self._request(
            self.config.model.name,
            payload,
        )
        # Response example (https://docs.mosaicml.com/en/latest/inference.html#text-embedding-models):
        # ```
        # {
        #   "outputs": [
        #     [
        #       3.25,
        #       0.7685547,
        #       2.65625,
        #       ...
        #       -0.30126953,
        #       -2.3554688,
        #       1.2597656
        #     ]
        #   ]
        # }
        # ```
        return embeddings.ResponsePayload(
            **{
                "embeddings": resp["outputs"],
                "metadata": {
                    "model": self.config.model.name,
                    "route_type": self.config.route_type,
                },
            }
        )
