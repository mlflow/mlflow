from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from mlflow.exceptions import MlflowException
from mlflow.utils.uri import append_to_uri_path, append_to_uri_query_params

from .base import BaseProvider
from .utils import send_request, rename_payload_keys
from ..schemas import chat, completions, embeddings
from ..config import OpenAIConfig, RouteConfig

_API_TYPE_OPENAI = "openai"
_API_TYPE_AZURE = "azure"
_API_TYPE_AZUREAD = "azuread"


class OpenAIProvider(BaseProvider):
    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, OpenAIConfig):
            # Should be unreachable
            raise MlflowException.invalid_parameter_value("Invalid config type {config.model.config}")
        self.openai_config: OpenAIConfig = config.model.config
        self.openai_api_type = (self.openai_config.openai_api_type or _API_TYPE_OPENAI).lower()
        self._validate_openai_config()

    def _validate_openai_config(self):
        if not self.openai_api_key:
            raise MlflowException.invalid_parameter_value("'openai_api_key' must be specified")

        if self.openai_api_type == _API_TYPE_OPENAI:
            if self.openai_config.openai_deployment_name is not None:
                raise MlflowException.invalid_parameter_value(
                    f"OpenAI route configuration can only specify a value for "
                    f"'openai_deployment_name' if 'openai_api_type' is '{_API_TYPE_AZURE}' or "
                    f"'{_API_TYPE_AZUREAD}'. Found type: '{self.openai_api_type}'"
                )
        elif self.openai_api_type in (_API_TYPE_AZURE, _API_TYPE_AZUREAD):
            base_url = self.openai_config.openai_api_base
            deployment_name = self.openai_config.openai_deployment_name
            api_version = self.openai_config.openai_api_version
            if (base_url, deployment_name, api_version).count(None) >= 0:
                raise MlflowException.invalid_parameter_value(
                    f"OpenAI route configuration must specify 'openai_api_base', "
                    f"'openai_deployment_name', and 'openai_api_version' if 'openai_api_type' is "
                    f"'{_API_TYPE_AZURE}' or '{_API_TYPE_AZUREAD}'."
                )
        else:
            raise MlflowException.invalid_parameter_value(f"Invalid OpenAI API type '{self.openai_api_type}'")

    @property
    def _request_base_url(self):
        if self.openai_api_type == _API_TYPE_OPENAI:
            return self.openai_config.openai_api_base or "https://api.openai.com/v1"
        elif self.openai_api_type in (_API_TYPE_AZURE, _API_TYPE_AZUREAD):
            if self.openai_config.openai_api_base is None:
                raise MlflowException.invalid_parameter_value(
                    f"'openai_api_base' must be specified when 'openai_api_type' is "
                    f"'{_API_TYPE_AZURE}' or '{_API_TYPE_AZUREAD}'."
                )
            openai_url = append_to_uri_path(
                self.openai_config.openai_api_base,
                "openai",
                "deployment",
                self.openai_config.openai_deployment_name
            )
            return append_to_uri_query_params(
                openai_url,
                ("api-version", self.openai_config.openai_api_version),
            )
        else:
            raise MlflowException.invalid_parameter_value(f"Invalid OpenAI API type '{self.openai_api_type}'")

    @property
    def _request_headers(self):
        if self.openai_api_type == _API_TYPE_OPENAI:
            headers = {
                "Authorization": f"Bearer {self.openai_config.openai_api_key}",
            }
            if org := self.openai_config.openai_organization:
                headers["OpenAI-Organization"] = org
            return headers
        elif self.openai_api_type == _API_TYPE_AZUREAD:
            return {
                "Authorization": f"Bearer {self.openai_config.openai_api_key}",
            }
        elif self.openai_api_type == _API_TYPE_AZURE:
            return {
                "api-key": self.openai_config.openai_api_key,
            }
        else:
            raise MlflowException.invalid_parameter_value(f"Invalid OpenAI API type '{self.openai_api_type}'")

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
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
            payload={"model": self.config.model.name, **payload},
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
            payload={"model": self.config.model.name, **payload},
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
        resp = await send_request(
            headers=self._request_headers,
            base_url=self._request_base_url,
            path="embeddings",
            payload={"model": self.config.model.name, **payload},
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
