from typing import Any, AsyncIterable

from mlflow.gateway.config import EndpointConfig, _OpenAICompatibleConfig
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.providers.base import PassthroughAction
from mlflow.gateway.providers.openai_compatible import OpenAICompatibleProvider


class LiteLLMProxyProvider(OpenAICompatibleProvider):
    """Routes requests to a self-hosted LiteLLM Proxy server.

    A LiteLLM Proxy exposes an OpenAI-compatible chat API plus a TypeSafe Jev decisions
    route at ``/typesafe/v1/systemone``. Unlike the SDK-based ``litellm`` provider, this
    one HTTP-forwards to the proxy, so it can reach the System One route that the SDK
    cannot. ``api_base`` must point at the proxy (e.g. ``http://my-proxy:4000/v1``); there
    is no public default.
    """

    DISPLAY_NAME = "LiteLLM Proxy"
    CONFIG_TYPE = _OpenAICompatibleConfig
    PASSTHROUGH_PROVIDER_PATHS = {
        **OpenAICompatibleProvider.PASSTHROUGH_PROVIDER_PATHS,
        PassthroughAction.TYPESAFE_SYSTEM_ONE: "typesafe/v1/systemone",
    }

    def __init__(self, config: EndpointConfig, enable_tracing: bool = False) -> None:
        super().__init__(config, enable_tracing=enable_tracing)
        if not getattr(self._provider_config, "api_base", None):
            raise AIGatewayException(
                status_code=400,
                detail="LiteLLM Proxy endpoints require 'api_base' pointing at the proxy server.",
            )

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
