"""
Vertex AI provider for MLflow AI Gateway.

Extends the Gemini provider to use Vertex AI endpoints with Google Cloud
authentication (Application Default Credentials or service account JSON).

Three model types are supported:

- **Google models** (gemini-*, medlm-*, text-embedding-*, etc.):
  Gemini API format with :generateContent/:streamGenerateContent endpoints.

- **Anthropic Claude models** (claude-*):
  Anthropic Messages API format with :rawPredict/:streamRawPredict endpoints,
  routed through publishers/anthropic.

- **OpenAI-compatible MaaS models** (Meta/Llama, Mistral, DeepSeek,
  AI21 Jamba, xAI Grok, Qwen, etc.):
  OpenAI Chat Completions format via the shared /endpoints/openapi endpoint.
"""

import json
from enum import Enum
from pathlib import Path
from typing import Any

from mlflow.gateway.config import EndpointConfig, VertexAIConfig
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.providers.anthropic import AnthropicProvider
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.gemini import GeminiProvider
from mlflow.gateway.providers.openai_compatible import OpenAICompatibleProvider

_DEFAULT_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# Version string required by Vertex AI for Anthropic models
_VERTEX_ANTHROPIC_VERSION = "vertex-2023-10-16"

# No-slash model name prefixes that belong to the MaaS (OpenAI-compatible) type
# rather than the Google (Gemini API) type.
_MAAS_PREFIXES = ("mistral", "codestral", "jamba")


def _classify_model(model_name: str) -> str:
    """Return the model type for a Vertex AI model name: 'gemini', 'claude', or 'maas'."""
    name = model_name.lower()
    if name.startswith("claude"):
        return "claude"
    if "/" in name or name.startswith(_MAAS_PREFIXES) or name.endswith("-maas"):
        return "maas"
    return "gemini"


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

    def get_endpoint_url(self, route_type: str) -> str:
        if route_type == "llm/v1/chat":
            return f"{self.base_url}/{self.config.model.name}:rawPredict"
        raise ValueError(f"Unsupported route type for Vertex AI Claude: {route_type}")

    def _get_chat_path(self) -> str:
        return f"{self.config.model.name}:rawPredict"

    def _get_chat_stream_path(self) -> str:
        return f"{self.config.model.name}:streamRawPredict"

    def _prepare_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload.pop("model", None)
        payload["anthropic_version"] = _VERTEX_ANTHROPIC_VERSION
        return payload


class _VertexAIMaaSProvider(OpenAICompatibleProvider):
    """OpenAICompatibleProvider adapted for MaaS models hosted on Vertex AI.

    All partner MaaS models (Meta/Llama, Mistral, DeepSeek, AI21, xAI, Qwen, etc.)
    share a single OpenAI-compatible endpoint at /endpoints/openapi, authenticated
    with a GCP Bearer token.
    """

    DISPLAY_NAME = "Vertex AI"
    CONFIG_TYPE = VertexAIConfig

    def __init__(self, config: EndpointConfig, vertex_config: VertexAIConfig, get_credentials_fn):
        # Call BaseProvider.__init__ directly — OpenAICompatibleProvider.__init__ would
        # reject VertexAIConfig since it expects an _OpenAICompatibleConfig.
        BaseProvider.__init__(self, config)
        self.vertex_config = vertex_config
        self._get_creds = get_credentials_fn

    @property
    def headers(self) -> dict[str, str]:
        creds = self._get_creds()
        return {"Authorization": f"Bearer {creds.token}"}

    @property
    def _api_base(self) -> str:
        project = self.vertex_config.vertex_project
        location = self.vertex_config.vertex_location or "us-central1"
        prefix = "" if location == "global" else f"{location}-"
        host = f"https://{prefix}aiplatform.googleapis.com"
        return f"{host}/v1/projects/{project}/locations/{location}/endpoints/openapi"


class VertexAIProvider(GeminiProvider):
    """Vertex AI provider supporting Google, Anthropic, and OpenAI-compatible MaaS models.

    - Google models  → Gemini API (:generateContent)
    - Claude models  → Anthropic API (:rawPredict) via _VertexAIClaudeProvider
    - MaaS models    → OpenAI Chat Completions (/endpoints/openapi) via _VertexAIMaaSProvider
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

        self._model_type = _classify_model(config.model.name)
        if self._model_type == "claude":
            self._delegate = _VertexAIClaudeProvider(
                config, self.vertex_config, self._get_credentials
            )
        elif self._model_type == "maas":
            self._delegate = _VertexAIMaaSProvider(
                config, self.vertex_config, self._get_credentials
            )
        else:
            self._delegate = None

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
            try:
                info = json.loads(creds_data)
            except (json.JSONDecodeError, TypeError):
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
            credentials, _ = google.auth.default(scopes=_DEFAULT_SCOPES)

        credentials.refresh(google.auth.transport.requests.Request())
        self._cached_credentials = credentials
        return credentials

    @property
    def headers(self) -> dict[str, str]:
        credentials = self._get_credentials()
        return {"Authorization": f"Bearer {credentials.token}"}

    @property
    def base_url(self) -> str:
        if self._model_type == "maas":
            return self._delegate._api_base
        project = self.vertex_config.vertex_project
        location = self.vertex_config.vertex_location or "global"
        # Regional endpoints use a "{location}-" prefix; the global endpoint has no prefix.
        # https://docs.cloud.google.com/vertex-ai/docs/general/googleapi-access-methods#regional-global-endpoints
        prefix = "" if location == "global" else f"{location}-"
        host = f"https://{prefix}aiplatform.googleapis.com"
        publisher = "anthropic" if self._model_type == "claude" else "google"
        path = f"/v1/projects/{project}/locations/{location}/publishers/{publisher}/models"
        return f"{host}{path}"

    def get_endpoint_url(self, route_type: str) -> str:
        if self._delegate:
            return self._delegate.get_endpoint_url(route_type)
        return super().get_endpoint_url(route_type)

    async def _chat(self, payload):
        if self._delegate:
            return await self._delegate._chat(payload)
        return await super()._chat(payload)

    async def _chat_stream(self, payload):
        if self._delegate:
            async for chunk in self._delegate._chat_stream(payload):
                yield chunk
            return
        async for chunk in super()._chat_stream(payload):
            yield chunk

    async def _completions(self, payload):
        if self._model_type in ("claude", "maas"):
            raise AIGatewayException(
                status_code=501,
                detail=(
                    f"The completions endpoint is not supported for {self.config.model.name} on "
                    "Vertex AI. Use the chat endpoint instead."
                ),
            )
        return await super()._completions(payload)

    async def _completions_stream(self, payload):
        if self._model_type in ("claude", "maas"):
            raise AIGatewayException(
                status_code=501,
                detail=(
                    f"The completions endpoint is not supported for {self.config.model.name} on "
                    "Vertex AI. Use the chat endpoint instead."
                ),
            )
        async for chunk in super()._completions_stream(payload):
            yield chunk
