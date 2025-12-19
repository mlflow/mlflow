import time
from typing import Any

from mlflow.gateway.config import EndpointConfig, MistralConfig
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.utils import send_request
from mlflow.gateway.schemas import chat, completions, embeddings


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
    def model_to_chat(cls, resp, config):
        # Response example (https://docs.mistral.ai/api/#operation/createChatCompletion)
        return chat.ResponsePayload(
            id=resp["id"],
            object=resp["object"],
            created=resp["created"],
            model=resp["model"],
            choices=[
                chat.Choice(
                    index=idx,
                    message=chat.ResponseMessage(
                        role=c["message"]["role"],
                        content=c["message"].get("content"),
                        tool_calls=(
                            (calls := c["message"].get("tool_calls"))
                            and [chat.ToolCall(**c) for c in calls]
                        ),
                    ),
                    finish_reason=c.get("finish_reason"),
                )
                for idx, c in enumerate(resp["choices"])
            ],
            usage=chat.ChatUsage(
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
        payload["model"] = config.model.name
        payload.pop("stop", None)
        payload.pop("n", None)
        payload["messages"] = [{"role": "user", "content": payload.pop("prompt")}]

        # The range of Mistral's temperature is 0-1, but ours is 0-2, so we scale it.
        if "temperature" in payload:
            payload["temperature"] = 0.5 * payload["temperature"]

        return payload

    @classmethod
    def chat_to_model(cls, payload, config):
        return {"model": config.model.name, **payload}

    @classmethod
    def embeddings_to_model(cls, payload, config):
        return {"model": config.model.name, **payload}


class MistralProvider(BaseProvider):
    NAME = "Mistral"
    CONFIG_TYPE = MistralConfig

    def __init__(self, config: EndpointConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, MistralConfig):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.mistral_config: MistralConfig = config.model.config

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.mistral_config.mistral_api_key}"}

    @property
    def base_url(self) -> str:
        return "https://api.mistral.ai/v1"

    @property
    def adapter_class(self) -> type[ProviderAdapter]:
        return MistralAdapter

    def get_endpoint_url(self, route_type: str) -> str:
        if route_type == "llm/v1/chat":
            return f"{self.base_url}/chat/completions"
        else:
            raise ValueError(f"Invalid route type {route_type}")

    async def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await send_request(
            headers=self.headers,
            base_url=self.base_url,
            path=path,
            payload=payload,
        )

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        resp = await self._request(
            "chat/completions",
            MistralAdapter.completions_to_model(payload, self.config),
        )
        return MistralAdapter.model_to_completions(resp, self.config)

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        resp = await self._request(
            "embeddings",
            MistralAdapter.embeddings_to_model(payload, self.config),
        )
        return MistralAdapter.model_to_embeddings(resp, self.config)
