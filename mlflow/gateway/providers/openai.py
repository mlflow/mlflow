from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from mlflow.exceptions import MlflowException
from mlflow.gateway.config import OpenAIAPIType, OpenAIConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import rename_payload_keys, send_request
from mlflow.gateway.schemas import chat, completions, embeddings
from mlflow.utils.uri import append_to_uri_path, append_to_uri_query_params


class OpenAIProvider(BaseProvider):
    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, OpenAIConfig):
            # Should be unreachable
            raise MlflowException.invalid_parameter_value(
                "Invalid config type {config.model.config}"
            )
        self.openai_config: OpenAIConfig = config.model.config

    @property
    def _request_base_url(self):
        api_type = self.openai_config.openai_api_type
        if api_type == OpenAIAPIType.OPENAI:
            base_url = self.openai_config.openai_api_base or "https://api.openai.com/v1"
            if api_version := self.openai_config.openai_api_version is not None:
                return append_to_uri_query_params(base_url, ("api-version", api_version))
            else:
                return base_url
        elif api_type in (OpenAIAPIType.AZURE, OpenAIAPIType.AZUREAD):
            openai_url = append_to_uri_path(
                self.openai_config.openai_api_base,
                "openai",
                "deployments",
                self.openai_config.openai_deployment_name,
            )
            return append_to_uri_query_params(
                openai_url,
                ("api-version", self.openai_config.openai_api_version),
            )
        else:
            raise MlflowException.invalid_parameter_value(
                f"Invalid OpenAI API type '{self.openai_api_type}'"
            )

    @property
    def _request_headers(self):
        api_type = self.openai_config.openai_api_type
        if api_type == OpenAIAPIType.OPENAI:
            headers = {
                "Authorization": f"Bearer {self.openai_config.openai_api_key}",
            }
            if org := self.openai_config.openai_organization:
                headers["OpenAI-Organization"] = org
            return headers
        elif api_type == OpenAIAPIType.AZUREAD:
            return {
                "Authorization": f"Bearer {self.openai_config.openai_api_key}",
            }
        elif api_type == OpenAIAPIType.AZURE:
            return {
                "api-key": self.openai_config.openai_api_key,
            }
        else:
            raise MlflowException.invalid_parameter_value(
                f"Invalid OpenAI API type '{self.openai_api_type}'"
            )

    def _add_model_to_payload_if_necessary(self, payload):
        # NB: For Azure OpenAI, the deployment name (which is included in the URL) specifies
        # the model; it is not specified in the payload. For OpenAI outside of Azure, the
        # model is always specified in the payload
        if self.openai_config.openai_api_type not in (OpenAIAPIType.AZURE, OpenAIAPIType.AZUREAD):
            return {"model": self.config.model.name, **payload}
        else:
            return payload

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        if "n" in payload:
            raise HTTPException(
                status_code=400, detail="Invalid parameter `n`. Use `candidate_count` instead."
            )

        payload = rename_payload_keys(
            payload,
            {"candidate_count": "n"},
        )
        # The range of OpenAI's temperature is 0-2, but ours is 0-1, so we double it.
        payload["temperature"] = 2 * payload["temperature"]

        resp = await send_request(
            headers=self._request_headers,
            base_url=self._request_base_url,
            path="chat/completions",
            payload=self._add_model_to_payload_if_necessary(payload),
        )
        # Response example (https://platform.openai.com/docs/api-reference/chat/create)
        # ```
        # {
        #    "id":"chatcmpl-abc123",
        #    "object":"chat.completion",
        #    "created":1677858242,
        #    "model":"gpt-3.5-turbo-0301",
        #    "usage":{
        #       "prompt_tokens":13,
        #       "completion_tokens":7,
        #       "total_tokens":20
        #    },
        #    "choices":[
        #       {
        #          "message":{
        #             "role":"assistant",
        #             "content":"\n\nThis is a test!"
        #          },
        #          "finish_reason":"stop",
        #          "index":0
        #       }
        #    ]
        # }
        # ```
        return chat.ResponsePayload(
            **{
                "candidates": [
                    {
                        "message": {
                            "role": c["message"]["role"],
                            "content": c["message"]["content"],
                        },
                        "metadata": {
                            "finish_reason": c["finish_reason"],
                        },
                    }
                    for c in resp["choices"]
                ],
                "metadata": {
                    "input_tokens": resp["usage"]["prompt_tokens"],
                    "output_tokens": resp["usage"]["completion_tokens"],
                    "total_tokens": resp["usage"]["total_tokens"],
                    "model": resp["model"],
                    "route_type": self.config.route_type,
                },
            }
        )

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        if "n" in payload:
            raise HTTPException(
                status_code=400, detail="Invalid parameter `n`. Use `candidate_count` instead."
            )
        payload = rename_payload_keys(
            payload,
            {"candidate_count": "n"},
        )
        # The range of OpenAI's temperature is 0-2, but ours is 0-1, so we double it.
        payload["temperature"] = 2 * payload["temperature"]
        payload["messages"] = [{"role": "user", "content": payload.pop("prompt")}]
        resp = await send_request(
            headers=self._request_headers,
            base_url=self._request_base_url,
            path="chat/completions",
            payload=self._add_model_to_payload_if_necessary(payload),
        )
        # Response example (https://platform.openai.com/docs/api-reference/completions/create)
        # ```
        # {
        #   "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
        #   "object": "text_completion",
        #   "created": 1589478378,
        #   "model": "text-davinci-003",
        #   "choices": [
        #     {
        #       "text": "\n\nThis is indeed a test",
        #       "index": 0,
        #       "logprobs": null,
        #       "finish_reason": "length"
        #     }
        #   ],
        #   "usage": {
        #     "prompt_tokens": 5,
        #     "completion_tokens": 7,
        #     "total_tokens": 12
        #   }
        # }
        # ```
        return completions.ResponsePayload(
            **{
                "candidates": [
                    {
                        "text": c["message"]["content"],
                        "metadata": {"finish_reason": c["finish_reason"]},
                    }
                    for c in resp["choices"]
                ],
                "metadata": {
                    "input_tokens": resp["usage"]["prompt_tokens"],
                    "output_tokens": resp["usage"]["completion_tokens"],
                    "total_tokens": resp["usage"]["total_tokens"],
                    "model": resp["model"],
                    "route_type": self.config.route_type,
                },
            }
        )

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        payload = rename_payload_keys(
            jsonable_encoder(payload, exclude_none=True),
            {"text": "input"},
        )
        self.check_for_model_field(payload)
        resp = await send_request(
            headers=self._request_headers,
            base_url=self._request_base_url,
            path="embeddings",
            payload=self._add_model_to_payload_if_necessary(payload),
        )
        # Response example (https://platform.openai.com/docs/api-reference/embeddings/create):
        # ```
        # {
        #   "object": "list",
        #   "data": [
        #     {
        #       "object": "embedding",
        #       "embedding": [
        #         0.0023064255,
        #         -0.009327292,
        #         .... (1536 floats total for ada-002)
        #         -0.0028842222,
        #       ],
        #       "index": 0
        #     }
        #   ],
        #   "model": "text-embedding-ada-002",
        #   "usage": {
        #     "prompt_tokens": 8,
        #     "total_tokens": 8
        #   }
        # }
        # ```
        return embeddings.ResponsePayload(
            **{
                "embeddings": [d["embedding"] for d in resp["data"]],
                "metadata": {
                    "input_tokens": resp["usage"]["prompt_tokens"],
                    "output_tokens": 0,  # output_tokens is not available for embeddings
                    "total_tokens": resp["usage"]["total_tokens"],
                    "model": resp["model"],
                    "route_type": self.config.route_type,
                },
            }
        )
