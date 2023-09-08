from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import AnthropicConfig, RouteConfig
from mlflow.gateway.constants import (
    MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS,
    MLFLOW_AI_GATEWAY_ANTHROPIC_MAXIMUM_MAX_TOKENS,
)
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import rename_payload_keys, send_request
from mlflow.gateway.schemas import chat, completions, embeddings


class AnthropicProvider(BaseProvider):
    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, AnthropicConfig):
            raise TypeError(f"Invalid config type {config.model.config}")
        self.anthropic_config: AnthropicConfig = config.model.config
        self.headers = {"x-api-key": self.anthropic_config.anthropic_api_key}
        self.base_url = "https://api.anthropic.com/v1/"

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        if "top_p" in payload:
            raise HTTPException(
                status_code=422,
                detail="Cannot set both 'temperature' and 'top_p' parameters. "
                "Please use only the temperature parameter for your query.",
            )
        max_tokens = payload.get("max_tokens", MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS)

        if max_tokens > MLFLOW_AI_GATEWAY_ANTHROPIC_MAXIMUM_MAX_TOKENS:
            raise HTTPException(
                status_code=422,
                detail="Invalid value for max_tokens: cannot exceed "
                f"{MLFLOW_AI_GATEWAY_ANTHROPIC_MAXIMUM_MAX_TOKENS}.",
            )

        payload["max_tokens"] = max_tokens

        if payload.get("stream", None) == "true":
            raise HTTPException(
                status_code=422,
                detail="Setting the 'stream' parameter to 'true' is not supported with the MLflow "
                "Gateway.",
            )
        candidate_count = payload.get("candidate_count", 1)
        if candidate_count != 1:
            raise HTTPException(
                status_code=422,
                detail="'candidate_count' must be '1' for the Anthropic provider. "
                f"Received value: '{candidate_count}'.",
            )

        payload = rename_payload_keys(
            payload, {"max_tokens": "max_tokens_to_sample", "stop": "stop_sequences"}
        )

        payload["prompt"] = f"\n\nHuman: {payload['prompt']}\n\nAssistant:"

        resp = await send_request(
            headers=self.headers,
            base_url=self.base_url,
            path="complete",
            payload={"model": self.config.model.name, **payload},
        )

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
                    "route_type": self.config.route_type,
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
