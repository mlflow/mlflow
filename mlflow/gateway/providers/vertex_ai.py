"""
Vertex AI provider for MLflow AI Gateway.

Extends the Gemini provider to use Vertex AI endpoints with Google Cloud
authentication (Application Default Credentials or service account JSON).

Vertex AI uses the same Gemini API format but with different URL patterns
and auth mechanism.
"""

import json
from pathlib import Path

from mlflow.gateway.config import EndpointConfig, VertexAIConfig
from mlflow.gateway.providers.gemini import GeminiProvider

_DEFAULT_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class VertexAIProvider(GeminiProvider):
    """Vertex AI provider for Google's Gemini models.

    Currently supports Google-published models only (e.g., gemini-2.0-flash).
    Partner models hosted on Vertex AI (Anthropic, Mistral, Llama, etc.) use
    different publisher paths and API formats (e.g., :rawPredict with
    Anthropic Messages API) and are not yet supported by this provider.
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
        self.vertex_config: VertexAIConfig = config.model.config
        self._cached_credentials = None

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

    @property
    def base_url(self) -> str:
        project = self.vertex_config.vertex_project
        location = self.vertex_config.vertex_location or "global"
        # Regional endpoints use a "{location}-" prefix; the global endpoint has no prefix.
        # https://docs.cloud.google.com/vertex-ai/docs/general/googleapi-access-methods#regional-global-endpoints
        prefix = "" if location == "global" else f"{location}-"
        host = f"https://{prefix}aiplatform.googleapis.com"
        path = f"/v1/projects/{project}/locations/{location}/publishers/google/models"
        return f"{host}{path}"
