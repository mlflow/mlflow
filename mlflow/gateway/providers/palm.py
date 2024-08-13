import time
from typing import Any, Dict

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import PaLMConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import rename_payload_keys, send_request
from mlflow.gateway.schemas import chat, completions, embeddings


class PaLMProvider(BaseProvider):
    NAME = "PaLM"
    CONFIG_TYPE = PaLMConfig

    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, PaLMConfig):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.palm_config: PaLMConfig = config.model.config

    async def _request(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"x-goog-api-key": self.palm_config.palm_api_key}
        return await send_request(
            headers=headers,
            base_url="https://generativelanguage.googleapis.com/v1beta3/models/",
            path=path,
            payload=payload,
        )

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        if "max_tokens" in payload or "maxOutputTokens" in payload:
            raise HTTPException(
                status_code=422, detail="Max tokens is not supported for PaLM chat."
            )
        key_mapping = {
            "stop": "stopSequences",
            "n": "candidateCount",
        }
        for k1, k2 in key_mapping.items():
            if k2 in payload:
                raise HTTPException(
                    status_code=422, detail=f"Invalid parameter {k2}. Use {k1} instead."
                )
        payload = rename_payload_keys(payload, key_mapping)
        # The range of PaLM's temperature is 0-1, but ours is 0-2, so we halve it
        payload["temperature"] = 0.5 * payload["temperature"]

        # Replace 'role' with 'author' in payload
        for m in payload["messages"]:
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
            created=int(time.time()),
            model=self.config.model.name,
            choices=[
                chat.Choice(
                    index=idx,
                    message=chat.ResponseMessage(role=c["author"], content=c["content"]),
                    finish_reason=None,
                )
                for idx, c in enumerate(resp["candidates"])
            ],
            usage=chat.ChatUsage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            ),
        )

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        key_mapping = {
            "stop": "stopSequences",
            "n": "candidateCount",
            "max_tokens": "maxOutputTokens",
        }
        for k1, k2 in key_mapping.items():
            if k2 in payload:
                raise HTTPException(
                    status_code=422, detail=f"Invalid parameter {k2}. Use {k1} instead."
                )
        payload = rename_payload_keys(payload, key_mapping)
        # The range of PaLM's temperature is 0-1, but ours is 0-2, so we halve it
        payload["temperature"] = 0.5 * payload["temperature"]
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
            created=int(time.time()),
            object="text_completion",
            model=self.config.model.name,
            choices=[
                completions.Choice(
                    index=idx,
                    text=c["output"],
                    finish_reason=None,
                )
                for idx, c in enumerate(resp["candidates"])
            ],
            usage=completions.CompletionsUsage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            ),
        )

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        key_mapping = {
            "input": "texts",
        }
        for k1, k2 in key_mapping.items():
            if k2 in payload:
                raise HTTPException(
                    status_code=422, detail=f"Invalid parameter {k2}. Use {k1} instead."
                )
        payload = rename_payload_keys(payload, key_mapping)

        resp = await self._request(
            f"{self.config.model.name}:batchEmbedText",
            payload,
        )

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
        return embeddings.ResponsePayload(
            data=[
                embeddings.EmbeddingObject(
                    embedding=embedding["value"],
                    index=idx,
                )
                for idx, embedding in enumerate(resp["embeddings"])
            ],
            model=self.config.model.name,
            usage=embeddings.EmbeddingsUsage(
                prompt_tokens=None,
                total_tokens=None,
            ),
        )
