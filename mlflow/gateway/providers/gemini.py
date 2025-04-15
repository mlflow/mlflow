from typing import Any

from mlflow.gateway.config import GeminiConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.utils import send_request
from mlflow.gateway.schemas import embeddings


class GeminiAdapter(ProviderAdapter):
    @classmethod
    def embeddings_to_model(cls, payload, config):
        # Example payload for the embedding API.
        # Documentation: https://ai.google.dev/api/embeddings#v1beta.ContentEmbedding
        #
        # {
        #     "requests": [
        #         {
        #             "model": "models/text-embedding-004",
        #             "content": {
        #                 "parts": [
        #                     {
        #                         "text": "What is the meaning of life?"
        #                     }
        #                 ]
        #             }
        #         },
        #         {
        #             "model": "models/text-embedding-004",
        #             "content": {
        #                 "parts": [
        #                     {
        #                         "text": "How much wood would a woodchuck chuck?"
        #                     }
        #                 ]
        #             }
        #         },
        #         {
        #             "model": "models/text-embedding-004",
        #             "content": {
        #                 "parts": [
        #                     {
        #                         "text": "How does the brain work?"
        #                     }
        #                 ]
        #             }
        #         }
        #     ]
        # }

        texts = payload["input"]
        if isinstance(texts, str):
            texts = [texts]
        return (
            {"content": {"parts": [{"text": texts[0]}]}}
            if len(texts) == 1
            else {
                "requests": [
                    {"model": f"models/{config.model.name}", "content": {"parts": [{"text": text}]}}
                    for text in texts
                ]
            }
        )

    @classmethod
    def model_to_embeddings(cls, resp, config):
        # Documentation: https://ai.google.dev/api/embeddings#v1beta.ContentEmbedding
        #
        # Example Response:
        # {
        #   "embeddings": [
        #     {
        #       "values": [
        #         3.25,
        #         0.7685547,
        #         2.65625,
        #         ...,
        #         -0.30126953,
        #         -2.3554688,
        #         1.2597656
        #       ]
        #     }
        #   ]
        # }

        data = [
            embeddings.EmbeddingObject(embedding=item.get("values", []), index=i)
            for i, item in enumerate(resp.get("embeddings") or [resp.get("embedding", {})])
        ]

        # Create and return response payload directly
        return embeddings.ResponsePayload(
            data=data,
            model=config.model.name,
            usage=embeddings.EmbeddingsUsage(
                prompt_tokens=None,
                total_tokens=None,
            ),
        )


class GeminiProvider(BaseProvider):
    NAME = "Gemini"
    CONFIG_TYPE = GeminiConfig

    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, GeminiConfig):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.gemini_config: GeminiConfig = config.model.config

    @property
    def headers(self):
        return {"x-goog-api-key": self.gemini_config.gemini_api_key}

    @property
    def base_url(self):
        return "https://generativelanguage.googleapis.com/v1beta/models"

    @property
    def adapter_class(self):
        return GeminiAdapter

    async def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await send_request(
            headers=self.headers,
            base_url=self.base_url,
            path=path,
            payload=payload,
        )

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)

        embedding_payload = self.adapter_class.embeddings_to_model(payload, self.config)
        # Documentation: https://ai.google.dev/api/embeddings

        # Use the batch endpoint if payload contains "requests"
        if "requests" in embedding_payload:
            endpoint_suffix = ":batchEmbedContents"
        else:
            endpoint_suffix = ":embedContent"

        resp = await self._request(
            f"{self.config.model.name}{endpoint_suffix}",
            embedding_payload,
        )
        return self.adapter_class.model_to_embeddings(resp, self.config)
