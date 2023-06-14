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
        headers = {"Authorization": f"Bearer {config.api_key}"}
        if config.openai_organization:
            headers["OpenAI-Organization"] = config.open_ai_organization
        async with aiohttp.ClientSession(self.config.openai_base, headers=headers) as session:
            async with session.post(path, json=payload) as response:
                response.raise_for_status()
                return await response.json()

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        resp = await self._make_request(
            "chat",
            {
                "model": self.config.model.name,
                "messages": jsonable_encoder(payload.messages),
            },
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
                    "route_type": self.config.type,
                },
            }
        )

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        resp = await self._make_request(
            "completions",
            {
                "model": self.config.model.name,
                "prompt": payload.prompt,
            },
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
        resp = await self._request(
            "embeddings",
            {
                "model": self.config.model.name,
                "input": jsonable_encoder(payload.documents),
            },
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
