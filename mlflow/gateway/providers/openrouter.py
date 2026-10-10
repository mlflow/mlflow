from typing import Any, AsyncIterable

from mlflow.gateway.config import _OpenAICompatibleConfig
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.providers.base import PassthroughAction
from mlflow.gateway.providers.openai_compatible import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    DISPLAY_NAME = "OpenRouter"
    CONFIG_TYPE = _OpenAICompatibleConfig
    DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
    PASSTHROUGH_PROVIDER_PATHS = {
        **OpenAICompatibleProvider.PASSTHROUGH_PROVIDER_PATHS,
        PassthroughAction.TYPESAFE_SYSTEM_ONE: "systemone",
    }

    async def _passthrough(
        self,
        action: PassthroughAction,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | AsyncIterable[Any]:
        if action != PassthroughAction.TYPESAFE_SYSTEM_ONE:
            return await super()._passthrough(action, payload, headers)

        if payload.get("stream"):
            raise AIGatewayException(
                status_code=400,
                detail="TypeSafe System One does not support streaming.",
            )

        # System One is a typed route, not a raw proxy. Do not forward caller headers.
        return await super()._passthrough(action, payload, headers=None)

    def _extract_passthrough_token_usage(
        self, action: PassthroughAction, result: dict[str, Any]
    ) -> dict[str, int] | None:
        if action == PassthroughAction.TYPESAFE_SYSTEM_ONE:
            return self._extract_token_usage_from_dict(
                result.get("usage"), "input_tokens", "output_tokens"
            )
        return super()._extract_passthrough_token_usage(action, result)
