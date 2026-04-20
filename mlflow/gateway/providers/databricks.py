from collections.abc import AsyncIterable
from typing import Any

from mlflow.gateway.base_models import ConfigModel
from mlflow.gateway.config import EndpointConfig, EndpointType
from mlflow.gateway.providers.base import PassthroughAction
from mlflow.gateway.providers.openai_compatible import (
    OpenAICompatibleAdapter,
    OpenAICompatibleProvider,
)
from mlflow.gateway.providers.utils import send_request, send_stream_request
from mlflow.gateway.schemas import chat
from mlflow.gateway.utils import normalize_databricks_base_url

_SUPPORTED_CONTENT_PART_TYPES = {"text", "image_url", "input_audio"}
_GEMINI_ACTIONS = {
    PassthroughAction.GEMINI_GENERATE_CONTENT,
    PassthroughAction.GEMINI_STREAM_GENERATE_CONTENT,
}


class DatabricksConfig(ConfigModel):
    """Config for Databricks provider.

    All fields are optional. When omitted, the Databricks SDK's default
    credential chain resolves host and authentication automatically
    (PAT, OAuth M2M, Azure CLI, etc.).

    Args:
        host: Databricks workspace URL (e.g., "https://my-workspace.databricks.com").
        token: Databricks Personal Access Token.
        client_id: OAuth M2M client ID (Service Principal).
        client_secret: OAuth M2M client secret (Service Principal).
    """

    host: str | None = None
    token: str | None = None
    client_id: str | None = None
    client_secret: str | None = None


class DatabricksAdapter(OpenAICompatibleAdapter):
    """Adapter that handles Databricks-specific response quirks.

    Databricks models may return unsupported content part types (e.g.,
    "reasoning") that the gateway schema doesn't support. This adapter
    filters them out.
    """

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
        for choice in resp.get("choices", []):
            if msg := choice.get("message"):
                msg["content"] = cls._normalize_content(msg.get("content"))
        return super().model_to_chat(resp, config)


class DatabricksProvider(OpenAICompatibleProvider):
    """Databricks provider using the Databricks SDK for authentication.

    Supports all auth methods provided by the Databricks SDK's default
    credential chain: PAT tokens, OAuth M2M (Service Principal), Azure CLI,
    Azure MSI, Google Cloud credentials, Databricks CLI profiles, etc.

    When explicit credentials (host/token/client_id/client_secret) are
    provided in the config, they are passed to the SDK. Otherwise the SDK
    resolves credentials from environment variables, ~/.databrickscfg, etc.
    """

    DISPLAY_NAME = "Databricks"
    CONFIG_TYPE = DatabricksConfig

    PASSTHROUGH_PROVIDER_PATHS = {
        PassthroughAction.OPENAI_CHAT: "chat/completions",
        PassthroughAction.OPENAI_EMBEDDINGS: "embeddings",
        PassthroughAction.OPENAI_RESPONSES: "responses",
        PassthroughAction.ANTHROPIC_MESSAGES: "anthropic/v1/messages",
        PassthroughAction.GEMINI_GENERATE_CONTENT: "gemini/v1beta/models/{model}:generateContent",
        PassthroughAction.GEMINI_STREAM_GENERATE_CONTENT: (
            "gemini/v1beta/models/{model}:streamGenerateContent"
        ),
    }

    @property
    def adapter_class(self):
        return DatabricksAdapter

    def __init__(self, config: EndpointConfig, enable_tracing: bool = False):
        self.config = config
        self._enable_tracing = enable_tracing
        self._provider_config = config.model.config
        self._workspace_client = None

    def _get_workspace_client(self):
        if self._workspace_client is None:
            from databricks.sdk import WorkspaceClient

            # Pass only non-None fields to WorkspaceClient; omitted fields
            # are resolved by the SDK's default credential chain.
            kwargs = {k: v for k, v in self._provider_config.model_dump().items() if v is not None}
            self._workspace_client = WorkspaceClient(**kwargs)
        return self._workspace_client

    @property
    def _api_base(self) -> str:
        client = self._get_workspace_client()
        host = client.config.host.rstrip("/")
        return normalize_databricks_base_url(host)

    def get_endpoint_url(self, route_type: str) -> str:
        _ROUTE_SUFFIXES = {
            EndpointType.LLM_V1_CHAT: "chat/completions",
            EndpointType.LLM_V1_COMPLETIONS: "completions",
            EndpointType.LLM_V1_EMBEDDINGS: "embeddings",
        }
        if route_type not in _ROUTE_SUFFIXES:
            raise ValueError(
                f"Unsupported route_type '{route_type}' for Databricks provider. "
                f"Supported: {sorted(_ROUTE_SUFFIXES)}"
            )
        return f"{self._api_base}/{_ROUTE_SUFFIXES[route_type]}"

    async def _passthrough(
        self,
        action: PassthroughAction,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | AsyncIterable[Any]:
        if action not in _GEMINI_ACTIONS:
            return await super()._passthrough(action, payload, headers)

        # Gemini actions need {model} formatted in the path and use
        # action-based streaming (not payload-based).
        provider_path = self._validate_passthrough_action(action)
        provider_path = provider_path.format(model=self.config.model.name)
        request_headers = self._get_headers(headers)

        is_streaming = action == PassthroughAction.GEMINI_STREAM_GENERATE_CONTENT
        if is_streaming:
            stream = send_stream_request(
                headers=request_headers,
                base_url=self._api_base,
                path=provider_path,
                payload=payload,
            )
            return self._stream_passthrough_with_usage(stream)
        else:
            return await send_request(
                headers=request_headers,
                base_url=self._api_base,
                path=provider_path,
                payload=payload,
            )

    @property
    def headers(self) -> dict[str, str]:
        client = self._get_workspace_client()
        return client.config.authenticate()
