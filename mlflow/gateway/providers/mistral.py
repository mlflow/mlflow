import time
from typing import Any, Dict

from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import MistralConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.utils import send_request
from mlflow.gateway.schemas import completions, embeddings


class MistralAdapter(ProviderAdapter):
    @classmethod
    def model_to_completions(cls, resp, config):
        # Response example (https://docs.mistral.ai/api/#operation/createChatCompletion)
        # ```
        # {
        #   "id": "string",
        #   "object": "string",
        #   "created": "integer",
        #   "model": "string",
        #   "choices": [
        #     {
        #       "index": "integer",
        #       "message": {
        #           "role": "string",
        #           "content": "string"
        #       },
        #       "finish_reason": "string",
        #     }
        #   ],
        #   "usage":
        #   {
        #       "prompt_tokens": "integer",
        #       "completion_tokens": "integer",
        #       "total_tokens": "integer",
        #   }
        # }
        # ```
        return completions.ResponsePayload(
            created=int(time.time()),
            object="text_completion",
            model=config.model.name,
            choices=[
                completions.Choice(
                    index=idx,
                    text=c["message"]["content"],
                    finish_reason=c["finish_reason"],
                )
                for idx, c in enumerate(resp["choices"])
            ],
            usage=completions.CompletionsUsage(
                prompt_tokens=resp["usage"]["prompt_tokens"],
                completion_tokens=resp["usage"]["completion_tokens"],
                total_tokens=resp["usage"]["total_tokens"],
            ),
        )

    @classmethod
    def model_to_embeddings(cls, resp, config):
        # Response example (https://docs.mistral.ai/api/#operation/createEmbedding):
        # ```
        # {
        #   "id": "string",
        #   "object": "string",
        #   "data": [
        #     {
        #       "object": "string",
        #       "embedding":
        #       [
        #           float,
        #           float
        #       ]
        #       "index": "integer",
        #     }
        #   ],
        #   "model": "string",
        #   "usage":
        #   {
        #       "prompt_tokens": "integer",
        #       "total_tokens": "integer",
        #   }
        # }
        # ```
        return embeddings.ResponsePayload(
            data=[
                embeddings.EmbeddingObject(
                    embedding=data["embedding"],
                    index=data["index"],
                )
                for data in resp["data"]
            ],
            model=config.model.name,
            usage=embeddings.EmbeddingsUsage(
                prompt_tokens=resp["usage"]["prompt_tokens"],
                total_tokens=resp["usage"]["total_tokens"],
            ),
        )

    @classmethod
    def completions_to_model(cls, payload, config):
        payload.pop("stop", None)
        payload.pop("n", None)
        payload["messages"] = [{"role": "user", "content": payload.pop("prompt")}]

        # The range of Mistral's temperature is 0-1, but ours is 0-2, so we scale it.
        if "temperature" in payload:
            payload["temperature"] = 0.5 * payload["temperature"]

        return payload

    @classmethod
    def embeddings_to_model(cls, payload, config):
        return payload


class MistralProvider(BaseProvider):
    NAME = "Mistral"
    CONFIG_TYPE = MistralConfig

    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, MistralConfig):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.mistral_config: MistralConfig = config.model.config

    @property
    def auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.mistral_config.mistral_api_key}"}

    @property
    def base_url(self) -> str:
        return "https://api.mistral.ai/v1/"

    async def _request(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await send_request(
            headers=self.auth_headers,
            base_url=self.base_url,
            path=path,
            payload=payload,
        )

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        resp = await self._request(
            "chat/completions",
            {
                "model": self.config.model.name,
                **MistralAdapter.completions_to_model(payload, self.config),
            },
        )
        return MistralAdapter.model_to_completions(resp, self.config)

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        resp = await self._request(
            "embeddings",
            {
                "model": self.config.model.name,
                **MistralAdapter.embeddings_to_model(payload, self.config),
            },
        )
        return MistralAdapter.model_to_embeddings(resp, self.config)
