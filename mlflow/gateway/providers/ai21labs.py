from typing import Any, Dict

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import AI21LabsConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import rename_payload_keys, send_request
from mlflow.gateway.schemas import chat, completions, embeddings


class AI21LabsProvider(BaseProvider):
    allowed_models = set(["j2-ultra", "j2-mid", "j2-light"])
    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, AI21LabsConfig):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.ai21labs_config: AI21LabsConfig = config.model.config
        self.headers = {"Authorization": f"Bearer {self.ai21labs_config.ai21labs_api_key}"}
        self.base_url = f"https://api.ai21.com/studio/v1/{self.config.model.name}/"
        if (self.config.model.name not in self.allowed_models):
            raise TypeError(
                f"Invalid modelName: {self.config.model.name}.” \
                “ Supported models for AI21Labs are {self.allowed_models}.")

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        key_mapping = {
            "stop": "stopSequences",
            "candidate_count": "numResults",
            "max_tokens": "maxTokens",
        }
        for k1, k2 in key_mapping.items():
            if k2 in payload:
                raise HTTPException(
                    status_code=422, detail=f"Invalid parameter {k2}. Use {k1} instead."
                )
        if payload.get("stream", None) == "true":
            raise HTTPException(
                status_code=422,
                detail="Setting the 'stream' parameter to 'true' is not supported with the MLflow "
                "Gateway.",
            )
        payload = rename_payload_keys(payload, key_mapping)
        resp = await send_request(
            headers=self.headers,
            base_url=self.base_url,
            path='complete',
            payload=payload,
        )
        # Response example (https://docs.ai21.com/reference/j2-complete-ref)
        # ```
        # {
        #   "id": "7921a78e-d905-c9df-27e3-88e4831e3c3b",
        #   "prompt": {
        #     "text": "I will"
        #   },
        #   "completions": [
        #     {
        #       "data": {
        #         "text": " complete this"
        #       },
        #       "finishReason": {
        #         "reason": "length",
        #         "length": 2
        #       }
        #     }
        #   ]
        # }
        # ```
        return completions.ResponsePayload(
            **{
                "candidates": [
                    {
                        "text": c["data"]["text"],
                        "metadata": {"finish_reason": c["finishReason"]["reason"]},
                    }
                    for c in resp["completions"]
                ],
                "metadata": {
                    "model": self.config.model.name,
                    "route_type": self.config.route_type,
                },
            }
        )

    async def chat(self, payload: chat.RequestPayload) -> None:
        # AI21Labs does not have a chat endpoint
        raise HTTPException(
            status_code=404, detail="The chat route is not available for AI21Labs models."
        )

    async def embeddings(self, payload: embeddings.RequestPayload) -> None:
        # AI21Labs does not have an embeddings endpoint
        raise HTTPException(
            status_code=404, detail="The embeddings route is not available for AI21Labs models."
        )