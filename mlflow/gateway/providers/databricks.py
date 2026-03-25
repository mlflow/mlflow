import logging
import time
from typing import Any

import requests
from pydantic import field_validator, model_validator

from mlflow.gateway.base_models import ConfigModel
from mlflow.gateway.config import (
    EndpointConfig,
    _OpenAICompatibleConfig,
    _resolve_api_key_from_input,
)
from mlflow.gateway.providers.base import PassthroughAction
from mlflow.gateway.providers.openai_compatible import (
    OpenAICompatibleAdapter,
    OpenAICompatibleProvider,
)
from mlflow.gateway.schemas import chat
from mlflow.gateway.utils import normalize_databricks_base_url

_logger = logging.getLogger(__name__)


class DatabricksConfig(_OpenAICompatibleConfig):
    """Config for Databricks PAT token authentication."""

    api_base: str

    @field_validator("api_key", mode="before")
    def validate_api_key(cls, value):
        return _resolve_api_key_from_input(value)

    @model_validator(mode="after")
    def normalize_api_base(self):
        self.api_base = normalize_databricks_base_url(self.api_base)
        return self


class DatabricksOAuthConfig(ConfigModel):
    """Config for Databricks OAuth M2M (Service Principal) authentication."""

    api_base: str
    client_id: str
    client_secret: str

    @field_validator("client_secret", mode="before")
    def validate_client_secret(cls, value):
        return _resolve_api_key_from_input(value)

    @model_validator(mode="after")
    def normalize_api_base(self):
        self.api_base = normalize_databricks_base_url(self.api_base)
        return self


_SUPPORTED_CONTENT_PART_TYPES = {"text", "image_url", "input_audio"}


class DatabricksAdapter(OpenAICompatibleAdapter):
    """Adapter that handles Databricks-specific response quirks."""

    @classmethod
    def _normalize_content(cls, content: Any) -> str | None:
        if not isinstance(content, list):
            return content
        supported = [p for p in content if p.get("type") in _SUPPORTED_CONTENT_PART_TYPES]
        if all(p.get("type") == "text" for p in supported):
            return "\n".join(p.get("text", "") for p in supported) or None
        return supported or None

    @classmethod
    def model_to_chat(cls, resp: dict[str, Any], config: EndpointConfig) -> chat.ResponsePayload:
        # Normalize content before delegating to the base adapter
        for choice in resp.get("choices", []):
            if msg := choice.get("message"):
                msg["content"] = cls._normalize_content(msg.get("content"))
        return super().model_to_chat(resp, config)


class DatabricksProvider(OpenAICompatibleProvider):
    """Databricks provider supporting PAT token and OAuth M2M authentication.

    Auth mode is determined by the config type:
    - ``DatabricksConfig``: Uses PAT token (api_key) directly as Bearer token.
    - ``DatabricksOAuthConfig``: Exchanges client_id + client_secret for a
      short-lived access token via the workspace OIDC endpoint, with
      automatic refresh before expiry.
    """

    NAME = "Databricks"
    CONFIG_TYPE = DatabricksConfig
    DEFAULT_API_BASE = ""

    PASSTHROUGH_PROVIDER_PATHS = {
        PassthroughAction.OPENAI_CHAT: "chat/completions",
        PassthroughAction.OPENAI_EMBEDDINGS: "embeddings",
        PassthroughAction.OPENAI_RESPONSES: "responses",
    }

    @property
    def adapter_class(self):
        return DatabricksAdapter

    def __init__(self, config, enable_tracing=False):
        # Initialize via BaseProvider directly because this provider accepts
        # two config types (DatabricksConfig for PAT, DatabricksOAuthConfig
        # for OAuth M2M), while OpenAICompatibleProvider.__init__ only allows
        # CONFIG_TYPE.
        from mlflow.gateway.providers.base import BaseProvider

        BaseProvider.__init__(self, config, enable_tracing=enable_tracing)
        self._provider_config = config.model.config
        self._access_token: str | None = None
        self._token_expiry: float = 0

    @property
    def _is_oauth(self) -> bool:
        return isinstance(self._provider_config, DatabricksOAuthConfig)

    def _get_workspace_host(self) -> str:
        return self._provider_config.api_base.removesuffix("/serving-endpoints")

    def _refresh_token(self) -> None:
        host = self._get_workspace_host()
        token_url = f"{host}/oidc/v1/token"

        response = requests.post(
            token_url,
            data={"grant_type": "client_credentials", "scope": "all-apis"},
            auth=(self._provider_config.client_id, self._provider_config.client_secret),
            timeout=30,
        )
        response.raise_for_status()

        token_data = response.json()
        self._access_token = token_data["access_token"]
        expires_in = token_data.get("expires_in", 3600)
        self._token_expiry = time.monotonic() + expires_in - 60

    def _ensure_token(self) -> str:
        if self._access_token is None or time.monotonic() >= self._token_expiry:
            self._refresh_token()
        return self._access_token

    @property
    def headers(self) -> dict[str, str]:
        if self._is_oauth:
            token = self._ensure_token()
            return {"Authorization": f"Bearer {token}"}
        return {"Authorization": f"Bearer {self._provider_config.api_key}"}
