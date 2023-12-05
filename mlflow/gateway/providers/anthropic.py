import time

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import AnthropicConfig, RouteConfig
from mlflow.gateway.constants import (
    MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS,
    MLFLOW_AI_GATEWAY_ANTHROPIC_MAXIMUM_MAX_TOKENS,
)
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.utils import rename_payload_keys, send_request
from mlflow.gateway.schemas import chat, completions, embeddings


class AnthropicAdapter(ProviderAdapter):
    @classmethod
    def model_to_completions(cls, resp, config):
        stop_reason = "stop" if resp["stop_reason"] == "stop_sequence" else "length"

        return completions.ResponsePayload(
            created=int(time.time()),
            object="text_completion",
            model=resp["model"],
            choices=[
                completions.Choice(
                    index=0,
                    text=resp["completion"],
                    finish_reason=stop_reason,
                )
            ],
            usage=completions.CompletionsUsage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            ),
        )

    @classmethod
    def completions_to_model(cls, payload, config):
        key_mapping = {"max_tokens": "max_tokens_to_sample", "stop": "stop_sequences"}

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
        n = payload.pop("n", 1)
        if n != 1:
            raise HTTPException(
                status_code=422,
                detail="'n' must be '1' for the Anthropic provider. Received value: '{n}'.",
            )

        payload = rename_payload_keys(payload, key_mapping)

        if payload["prompt"].startswith("Human: "):
            payload["prompt"] = "\n\n" + payload["prompt"]

        if not payload["prompt"].startswith("\n\nHuman: "):
            payload["prompt"] = "\n\nHuman: " + payload["prompt"]

        if not payload["prompt"].endswith("\n\nAssistant:"):
            payload["prompt"] = payload["prompt"] + "\n\nAssistant:"

        # The range of Anthropic's temperature is 0-1, but ours is 0-2, so we halve it
        if "temperature" in payload:
            payload["temperature"] = 0.5 * payload["temperature"]

        return payload

    @classmethod
    def embeddings_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    def model_to_embeddings(cls, resp, config):
        raise NotImplementedError


class AnthropicProvider(BaseProvider, AnthropicAdapter):
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

        resp = await send_request(
            headers=self.headers,
            base_url=self.base_url,
            path="complete",
            payload={
                "model": self.config.model.name,
                **AnthropicAdapter.completions_to_model(payload, self.config),
            },
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

        return AnthropicAdapter.model_to_completions(resp, self.config)

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
