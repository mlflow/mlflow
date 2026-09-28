from typing import Any

from mlflow.gateway.config import EndpointConfig, TypeSafeConfig
from mlflow.gateway.constants import TYPESAFE_API_BASE_URL, TYPESAFE_SYSTEM_ONE_PATH
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.providers.base import BaseProvider, PassthroughAction
from mlflow.gateway.providers.utils import send_request


class TypeSafeProvider(BaseProvider):
    DISPLAY_NAME = "TypeSafe"
    CONFIG_TYPE = TypeSafeConfig
    PASSTHROUGH_PROVIDER_PATHS = {PassthroughAction.TYPESAFE_SYSTEM_ONE: TYPESAFE_SYSTEM_ONE_PATH}

    def __init__(self, config: EndpointConfig, enable_tracing: bool = False):
        super().__init__(config, enable_tracing=enable_tracing)
        if not isinstance(config.model.config, TypeSafeConfig):
            raise TypeError(f"Invalid config type {config.model.config}")
        self.typesafe_config = config.model.config

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.typesafe_config.typesafe_api_key}"}

    @property
    def base_url(self) -> str:
        return TYPESAFE_API_BASE_URL

    async def _passthrough(
        self,
        action: PassthroughAction,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        provider_path = self._validate_passthrough_action(action)
        if payload.get("stream"):
            raise AIGatewayException(
                status_code=400, detail="TypeSafe System One does not support streaming."
            )
        request_payload = {k: v for k, v in payload.items() if k != "stream"}
        request_payload["model"] = self.config.model.name
        return await send_request(
            headers=self.headers,
            base_url=self.base_url,
            path=provider_path,
            payload=request_payload,
        )

    def _extract_passthrough_token_usage(
        self, action: PassthroughAction, result: dict[str, Any]
    ) -> dict[str, int] | None:
        return self._extract_token_usage_from_dict(
            result.get("usage"), "input_tokens", "output_tokens"
        )
