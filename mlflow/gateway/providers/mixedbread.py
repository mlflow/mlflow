
from typing import Any, Dict

from fastapi import HTTPException

from mlflow.exceptions import MlflowException
from mlflow.gateway.config import MixedBreadConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.utils import send_request
from mlflow.gateway.schemas import embeddings


class MixedBreadAdapter(ProviderAdapter):

    @classmethod
    def embeddings_to_model(cls, resp, config):
        # Example from API reference(https://www.mixedbread.ai/api-reference/endpoints/embeddings#create-embeddings):
        #{
        #   "model": "UAE-Large-V1",
        #   "data": [
        #      {
        #          "embedding": [
        #             0.1,
        #             0.06069,
        #             ...
        #          ],
        #          "index": 0
        #      },
        #      {
        #          "embedding": [
        #              -0.1,
        #              0.3,
        #              ...
        #          ],
        #          "index": 1,
        #          "truncated": true
        #      }
        #   ],
        #   "usage":{
        #      "prompt_tokens": 420,
        #      "total_tokens": 420
        #   },
        #   "normalized": true
        #}

        usage = resp.get("usage")
        prompt_tokens = None
        total_tokens = None
        # if usage is not None
        if usage:
            prompt_tokens = usage.get("prompt_tokens")
            total_tokens = usage.get("total_tokens")

        return embeddings.ResponsePayload(
            model=config.model.name,
            data=[
                embeddings.EmbeddingObject(
                    index=idx,
                    embedding=embedding_item['embedding'],
                )
                for idx, embedding_item in enumerate(resp.get("data", []))
                if 'embedding' in embedding_item
            ],
            usage=embeddings.EmbeddingsUsage(
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens
            ),
        )

    @classmethod
    def model_to_embeddings(cls, payload, config):

        #TODO maybe let the end API handle these?

        if "encoding_format" in payload:
            raise HTTPException(
                status_code=422,
                detail=("Parameter encoding_format in payload."
                "Mixedbread does not support encoding_format.")
            )

        if "dimensions" in payload:
            raise HTTPException(
                "Invalid parameter dimensions in payload."
                "The parameter is not supported in mixedbread."
            )

        if "user" in payload:
            raise HTTPException(
                "Invalid parameter user in payload."
                "The parameter is not supported in mixedbread."
            )

        return payload

class MixedBreadProvider(BaseProvider):

    NAME = "MixedBread"

    @property
    def base_url(self):
        return "https://api.mixedbread.ai/v1/"

    @property
    def auth_headers(self):

        return {"Authorization": f"Bearer {self.mixedbread_config.mixedbread_api_key}"}

    async def _request(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:

        return await send_request(
            headers=self.auth_headers,
            base_url=self.base_url,
            path=path,
            payload=payload
        )

    def __init__(self, config: RouteConfig) -> None:

        super().__init__(config)

        if config.model.config is None or not isinstance(config.model.config,MixedBreadConfig):
            raise MlflowException.invalid_parameter_value(
                f"Invalid config type {config.model.config}"
            )

        self.mixedbread_config: MixedBreadConfig = config.model.config



    async def embeddings(
        self,
        payload: embeddings.RequestPayload
    ) -> embeddings.ResponsePayload:

        from fastapi.encoders import jsonable_encoder


        payload = jsonable_encoder(payload, exclude_none=True)

        resp = await self._request(
            path="embeddings",
            payload={
                "model": self.config.model.name,
                **MixedBreadAdapter.model_to_embeddings(payload, self.config)
            }
        )

        return MixedBreadAdapter.embeddings_to_model(resp, self.config)
