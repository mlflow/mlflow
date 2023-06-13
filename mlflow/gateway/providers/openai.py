import openai
from fastapi.encoders import jsonable_encoder

from .base import BaseProvider
from ..schemas import chat, completions, embeddings


class OpenAIProvider(BaseProvider):
    NAME = "openai"
    SUPPORTED_ROUTES = ("chat", "completions", "embeddings")

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        resp = await openai.ChatCompletion.acreate(
            model=self.config.model.name,
            messages=jsonable_encoder(payload.messages),
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
        resp = await openai.Completion.acreate(
            model=self.config.model.name,
            prompt=payload.prompt,
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
        resp = await openai.Embedding.acreate(
            model=self.config.model.name,
            input=payload.text,
        )
        # Response example (https://platform.openai.com/docs/api-reference/embeddings/create):
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
        return embeddings.ResponsePayload(
            **{
                "embeddings": resp["data"][0]["embedding"],
                "metadata": {
                    "input_tokens": resp["usage"]["prompt_tokens"],
                    # The output tokens are not defined for embeddings
                    "output_tokens": 0,
                    "total_tokens": resp["usage"]["total_tokens"],
                    "model": resp["model"],
                    "route_type": self.config.type,
                },
            }
        )
