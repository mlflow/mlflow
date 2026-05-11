"""
Vertex AI provider for MLflow AI Gateway.

Extends the Gemini provider to use Vertex AI endpoints with Google Cloud
authentication (Application Default Credentials or service account JSON).

For Google-published models (e.g. gemini-2.0-flash), this uses the Gemini API
format with :generateContent/:streamGenerateContent endpoints.

For Anthropic Claude models, this uses AnthropicProvider's logic with Vertex AI
auth (Bearer token) and :rawPredict/:streamRawPredict endpoints.
"""

import json
from enum import Enum
from pathlib import Path

from mlflow.gateway.config import EndpointConfig, VertexAIConfig
from mlflow.gateway.providers.anthropic import AnthropicProvider
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.gemini import GeminiProvider

_DEFAULT_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# Version string required by Vertex AI for Anthropic models
_VERTEX_ANTHROPIC_VERSION = "vertex-2023-10-16"


class _VertexAIClaudeProvider(AnthropicProvider):
    """AnthropicProvider adapted for Claude models hosted on Vertex AI.

    Uses Vertex AI Bearer-token auth and :rawPredict/:streamRawPredict endpoints
    instead of the standard Anthropic API key and /messages endpoint.
    """

    DISPLAY_NAME = "Vertex AI"
    CONFIG_TYPE = VertexAIConfig

    def __init__(self, config: EndpointConfig, vertex_config: VertexAIConfig, get_credentials_fn):
        # Call BaseProvider.__init__ directly — AnthropicProvider.__init__ would reject
        # VertexAIConfig since it expects AnthropicConfig.
        BaseProvider.__init__(self, config)
        self.vertex_config = vertex_config
        self._get_creds = get_credentials_fn

    @property
    def headers(self) -> dict[str, str]:
        creds = self._get_creds()
        return {"Authorization": f"Bearer {creds.token}"}

    @property
    def base_url(self) -> str:
        project = self.vertex_config.vertex_project
        location = self.vertex_config.vertex_location or "global"
        prefix = "" if location == "global" else f"{location}-"
        host = f"https://{prefix}aiplatform.googleapis.com"
        path = f"/v1/projects/{project}/locations/{location}/publishers/anthropic/models"
        return f"{host}{path}"

    def _get_chat_path(self) -> str:
        return f"{self.config.model.name}:rawPredict"

    def _get_chat_stream_path(self) -> str:
        return f"{self.config.model.name}:streamRawPredict"

    def _prepare_payload(self, payload: dict) -> dict:
        payload.pop("model", None)
        payload["anthropic_version"] = _VERTEX_ANTHROPIC_VERSION
        return payload


class VertexAIProvider(GeminiProvider):
    """Vertex AI provider supporting both Google (Gemini) and Anthropic (Claude) models.

    Google-published models use the Gemini API format (:generateContent).
    Anthropic Claude models are handled by _VertexAIClaudeProvider, which reuses
    AnthropicProvider's request/response logic with Vertex AI auth and endpoints.
    """

    DISPLAY_NAME = "Vertex AI"
    CONFIG_TYPE = VertexAIConfig

    def get_provider_name(self) -> str:
        return "vertex_ai"

    def __init__(self, config: EndpointConfig, enable_tracing: bool = False) -> None:
        if not isinstance(config.model.config, VertexAIConfig):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.config = config
        self._enable_tracing = enable_tracing
        provider = config.model.provider
        self._provider_name = provider.value if isinstance(provider, Enum) else str(provider)
        self.vertex_config: VertexAIConfig = config.model.config
        self._cached_credentials = None
        self._claude_provider: _VertexAIClaudeProvider | None = (
            _VertexAIClaudeProvider(config, self.vertex_config, self._get_credentials)
            if self._is_claude_model()
            else None
        )

    def _is_claude_model(self) -> bool:
        return self.config.model.name.lower().startswith("claude")

    def _get_credentials(self):
        """Get Google Cloud credentials, caching them for reuse."""
        try:
            import google.auth
            import google.auth.transport.requests
            from google.oauth2 import service_account
        except ImportError:
            raise ImportError(
                "Vertex AI provider requires the google-auth package. "
                "Install it with: pip install google-auth"
            )

        if self._cached_credentials is not None and self._cached_credentials.valid:
            return self._cached_credentials

        if creds_data := self.vertex_config.vertex_credentials:
            # Try parsing as JSON string
            try:
                info = json.loads(creds_data)
            except (json.JSONDecodeError, TypeError):
                # Try as file path
                try:
                    info = json.loads(Path(creds_data).read_text())
                except Exception as e:
                    raise ValueError(
                        "vertex_credentials must be a JSON string or path to a JSON file"
                    ) from e
            credentials = service_account.Credentials.from_service_account_info(
                info, scopes=_DEFAULT_SCOPES
            )
        else:
            # Use Application Default Credentials
            credentials, _ = google.auth.default(scopes=_DEFAULT_SCOPES)

        # Refresh to get a valid access token
        credentials.refresh(google.auth.transport.requests.Request())
        self._cached_credentials = credentials
        return credentials

    @property
    def headers(self) -> dict[str, str]:
        credentials = self._get_credentials()
        return {"Authorization": f"Bearer {credentials.token}"}

    def _get_publisher(self) -> str:
        return "anthropic" if self._is_claude_model() else "google"

    @property
    def base_url(self) -> str:
        project = self.vertex_config.vertex_project
        location = self.vertex_config.vertex_location or "global"
        # Regional endpoints use a "{location}-" prefix; the global endpoint has no prefix.
        # https://docs.cloud.google.com/vertex-ai/docs/general/googleapi-access-methods#regional-global-endpoints
        prefix = "" if location == "global" else f"{location}-"
        host = f"https://{prefix}aiplatform.googleapis.com"
        publisher = self._get_publisher()
        path = f"/v1/projects/{project}/locations/{location}/publishers/{publisher}/models"
        return f"{host}{path}"

    async def _chat(self, payload):
        if self._claude_provider:
            return await self._claude_provider._chat(payload)
        return await super()._chat(payload)

    async def _chat_stream(self, payload):
        if self._claude_provider:
            async for chunk in self._claude_provider._chat_stream(payload):
                yield chunk
            return
        async for chunk in super()._chat_stream(payload):
            yield chunk

    async def _completions(self, payload):
        if self._claude_provider:
            raise NotImplementedError(
                "The completions endpoint is not supported for Anthropic Claude models on "
                "Vertex AI. Use the chat endpoint instead."
            )
        return await super()._completions(payload)

    async def _completions_stream(self, payload):
        if self._claude_provider:
            raise NotImplementedError(
                "The completions endpoint is not supported for Anthropic Claude models on "
                "Vertex AI. Use the chat endpoint instead."
            )
        async for chunk in super()._completions_stream(payload):
            yield chunk
