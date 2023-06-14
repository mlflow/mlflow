from typing import Dict, Any

from fastapi.encoders import jsonable_encoder

from .base import BaseProvider
from ..schemas import chat, completions, embeddings


class OpenAIProvider(BaseProvider):
    NAME = "openai"
    SUPPORTED_ROUTES = ("chat", "completions", "embeddings")

    async def _request(self, path: str, payload: Dict[str, Any]):
        import aiohttp

        config = self.config.model.config
        token = config["openai_api_key"]
        headers = {"Authorization": f"Bearer {token}"}
        if org := config.get("openai_organization"):
            headers["OpenAI-Organization"] = org
        async with aiohttp.ClientSession(headers=headers) as session:
            url = "/".join([config["openai_api_base"].rstrip("/"), path.lstrip("/")])
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                return await response.json()

    @staticmethod
    def _convert_payload(payload: Dict[str, Any], key_mapping: Dict[str, str]):
        payload = payload.copy()
        for k1, k2 in key_mapping.items():
            if v := payload.pop(k1, None):
                payload[k2] = v
        return payload

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        payload = OpenAIProvider._convert_payload(
            jsonable_encoder(payload), {"candidate_count": "n"}
        )
        resp = await self._request("chat", payload)
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
                    "route_type": self.config.type,
                },
            }
        )

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        payload = OpenAIProvider._convert_payload(
            jsonable_encoder(payload),
            {"candidate_count": "n"},
        )
        resp = await self._request("completions", payload)
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
                        "text": c["text"],
                        "metadata": {"finish_reason": c["finish_reason"]},
                    }
                    for c in resp["choices"]
                ],
                "metadata": {
                    "input_tokens": resp["usage"]["prompt_tokens"],
                    "output_tokens": resp["usage"]["completion_tokens"],
                    "total_tokens": resp["usage"]["total_tokens"],
                    "model": resp["model"],
                    "route_type": self.config.type,
                },
            }
        )

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        payload = OpenAIProvider._convert_payload(
            jsonable_encoder(payload),
            {"text": "input"},
        )
        resp = await self._request("embeddings", payload)
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
                "embeddings": resp["data"][0]["embedding"],
                "metadata": {
                    "input_tokens": resp["usage"]["prompt_tokens"],
                    "output_tokens": 0,  # output_tokens is not available for embeddings
                    "total_tokens": resp["usage"]["total_tokens"],
                    "model": resp["model"],
                    "route_type": self.config.type,
                },
            }
        )
