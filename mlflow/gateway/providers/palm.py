from typing import Any, Dict

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import PaLMConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import rename_payload_keys, send_request
from mlflow.gateway.schemas import chat, completions, embeddings


class PaLMProvider(BaseProvider):
    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, PaLMConfig):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.palm_config: PaLMConfig = config.model.config

    async def _request(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await send_request(
            headers={},
            base_url="https://generativelanguage.googleapis.com/v1beta3/models/",
            path=f"{path}?key={self.palm_config.palm_api_key}",
            payload=payload,
        )

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        key_mapping = {
            "stop": "stopSequences",
            "candidate_count": "candidateCount",
            "max_tokens": "maxOutputTokens",
        }
        for k1, k2 in key_mapping.items():
            if k2 in payload:
                raise HTTPException(
                    status_code=422, detail=f"Invalid parameter {k2}. Use {k1} instead."
                )
        payload = rename_payload_keys(payload, key_mapping)

        # Replace 'role' with 'author' in payload
        for m in payload["messages"]:
            if "role" in m:
                m["author"] = m.pop("role")

        # Map 'messages', 'examples, and 'context' to 'prompt'
        prompt = {"messages": payload.pop("messages")}
        if "examples" in payload:
            prompt["examples"] = payload.pop("examples")
        if "context" in payload:
            prompt["context"] = payload.pop("context")
        payload["prompt"] = prompt

        resp = await self._request(
            f"{self.config.model.name}:generateMessage",
            payload,
        )

        # Response example
        # (https://developers.generativeai.google/api/rest/generativelanguage/models/generateMessage)
        # ```
        # {
        #   "candidates": [
        #     {
        #       "author": "1",
        #       "content": "Hi there! How can I help you today?"
        #     }
        #   ],
        #   "messages": [
        #     {
        #       "author": "0",
        #       "content": "hi"
        #     }
        #   ]
        # }
        # ```
        return chat.ResponsePayload(
            **{
                "candidates": [
                    {
                        "message": {
                            "role": c["author"],
                            "content": c["content"],
                        },
                        "metadata": {},
                    }
                    for c in resp["candidates"]
                ],
                "metadata": {
                    "model": self.config.model.name,
                    "route_type": self.config.route_type,
                },
            }
        )

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        key_mapping = {
            "stop": "stopSequences",
            "candidate_count": "candidateCount",
            "max_tokens": "maxOutputTokens",
        }
        for k1, k2 in key_mapping.items():
            if k2 in payload:
                raise HTTPException(
                    status_code=422, detail=f"Invalid parameter {k2}. Use {k1} instead."
                )
        payload = rename_payload_keys(payload, key_mapping)
        payload["prompt"] = {"text": payload["prompt"]}
        resp = await self._request(
            f"{self.config.model.name}:generateText",
            payload,
        )
        # Response example (https://developers.generativeai.google/api/rest/generativelanguage/models/generateText)
        # ```
        # {
        #   "candidates": [
        #     {
        #       "output": "Once upon a time, there was a young girl named Lily...",
        #       "safetyRatings": [
        #         {
        #           "category": "HARM_CATEGORY_DEROGATORY",
        #           "probability": "NEGLIGIBLE"
        #         }, ...
        #       ]
        #     {
        #       "output": "Once upon a time, there was a young boy named Billy...",
        #       "safetyRatings": [
        #           ...
        #       ]
        #     }
        #   ]
        # }
        # ```
        return completions.ResponsePayload(
            **{
                "candidates": [
                    {
                        "text": c["output"],
                        "metadata": {},
                    }
                    for c in resp["candidates"]
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
        postfix = "embedText"

        # Remap text to texts to setup for batch embeddings
        if isinstance(payload["text"], list):
            postfix = "batchEmbedText"
            key_mapping = {
                "text": "texts",
            }
            for k1, k2 in key_mapping.items():
                if k2 in payload:
                    raise HTTPException(
                        status_code=422, detail=f"Invalid parameter {k2}. Use {k1} instead."
                    )
            payload = rename_payload_keys(payload, key_mapping)

        resp = await self._request(
            f"{self.config.model.name}:{postfix}",
            payload,
        )
        # Single-text response example (https://developers.generativeai.google/api/rest/generativelanguage/models/embedText):
        # ```
        # {
        #   "embedding": {
        #     "value": [
        #       3.25,
        #       0.7685547,
        #       2.65625,
        #       ...
        #       -0.30126953,
        #       -2.3554688,
        #       1.2597656
        #     ]
        #   }
        # }
        # ```
        # Batch-text response example (https://developers.generativeai.google/api/rest/generativelanguage/models/batchEmbedText):
        # ```
        # {
        #   "embeddings": [
        #     {
        #       "value": [
        #         3.25,
        #         0.7685547,
        #         2.65625,
        #         ...
        #         -0.30126953,
        #         -2.3554688,
        #         1.2597656
        #       ]
        #     }
        #   ]
        # }
        # ```

        if postfix == "batchEmbedText":
            embedResponse = [e["value"] for e in resp["embeddings"]]
        else:
            embedResponse = [resp["embedding"]["value"]]
        return embeddings.ResponsePayload(
            **{
                "embeddings": embedResponse,
                "metadata": {
                    "model": self.config.model.name,
                    "route_type": self.config.route_type,
                },
            }
        )
