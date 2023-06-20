from typing import Dict, Any

import aiohttp
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from .base import BaseProvider
from ..schemas import completions, chat, embeddings


class AnthropicProvider(BaseProvider):
    NAME = "anthropic"
    SUPPORTED_ROUTES = "completions"

    async def _request(self, path: str, payload: Dict[str, Any]):
        config = self.config.model.config
        token = config["anthropic_api_key"]
        headers = {"x-api-key": token}
        async with aiohttp.ClientSession(headers=headers) as session:
            url = "/".join([config["anthropic_api_base"].rstrip("/"), path.lstrip("/")])
            async with session.post(url, json=payload) as response:
                js = await response.json()
                try:
                    response.raise_for_status()
                except aiohttp.ClientResponseError as e:
                    detail = js.get("error", {}).get("message", e.message)
                    raise HTTPException(status_code=e.status, detail=detail)
                return js

    @staticmethod
    def _make_payload(payload: Dict[str, Any], mapping: Dict[str, str]):
        payload = payload.copy()
        for k1, k2 in mapping.items():
            if v := payload.pop(k1, None):
                payload[k2] = v
        return {k: v for k, v in payload.items() if v is not None and v != []}

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        payload = jsonable_encoder(payload)
        if "top_p" in payload:
            raise HTTPException(
                status_code=400,
                detail="Cannot set both 'temperature' and 'top_p' parameters. "
                "Please use only the temperature parameter for your query.",
            )
        if payload["max_tokens"] is None:
            raise HTTPException(
                status_code=400,
                detail="You must set an integer value for 'max_tokens' for the Anthropic provider "
                "that provides the upper bound on the returned token count.",
            )
        if payload.get("stream", None) == "true":
            raise HTTPException(
                status_code=400,
                detail="Setting the 'stream' parameter to 'true' is not supported with the MLflow "
                "Gateway.",
            )

        payload = AnthropicProvider._make_payload(
            payload, {"max_tokens_to_sample": "max_tokens", "stop_sequences": "stop"}
        )

        payload["prompt"] = f"\n\nHuman: {payload['prompt']}\n\nAssistant:"

        resp = await self._request("complete", {"model": self.config.model.name, **payload})

        # Example response:
        # Documentation: https://docs.anthropic.com/claude/reference/complete_post
        # ```
        # {
        #     "completion": " Hello! My name is Claude."
        #     "stop_reason": "stop_sequence",
        #     "model": "claude-instant-1.1",
        #     "truncated": False,
        #     "stop": None,
        #     "log_id": "dee173f87ddf1357da639dee3c38d833",
        #     "exception": None,
        # }
        # ```

        stop_reason = "stop" if resp["stop_reason"] == "stop_sequence" else "length"

        return completions.ResponsePayload(
            **{
                "candidates": [
                    {"text": resp["completion"], "metadata": {"finish_reason": stop_reason}}
                ],
                "metadata": {
                    "model": resp["model"],
                    "route_type": self.config.type,
                },
            }
        )

    async def chat(self, payload: chat.RequestPayload) -> None:
        # Anthropic does not have a chat endpoint
        raise HTTPException(
            status_code=404, detail="The chat route is not available for Anthropic models."
        )

    async def embeddings(self, payload: embeddings.RequestPayload) -> None:
        # Anthropic does not have an embeddings endpoint
        raise HTTPException(
            status_code=404, detail="The embeddings route is not available for Anthropic models."
        )
