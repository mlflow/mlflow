"""Ollama preset of the OpenAI-compatible assistant provider.

Talks to a locally running Ollama server via its native ``/api/tags``
listing endpoint and the OpenAI-compatible ``/v1/chat/completions`` shim.
"""

from typing import ClassVar

import requests

from mlflow.assistant.providers.openai_compatible import OpenAICompatibleProvider


class OllamaProvider(OpenAICompatibleProvider):
    """OpenAI-compatible provider for a locally running Ollama server."""

    # Exposed as a ClassVar so callers can reference the provider identifier
    # without instantiating the provider
    OLLAMA_PROVIDER_NAME: ClassVar[str] = "ollama"

    @staticmethod
    def _list_models(base_url: str, api_key: str | None = None) -> list[str]:
        """List models from a local Ollama server via `GET /api/tags`.

        Vanilla Ollama is auth-free, but the api_key is forwarded as a
        Bearer token when set so users who reverse-proxy Ollama behind an
        auth layer can still list models.
        """
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", headers=headers, timeout=10)
        response.raise_for_status()
        return [m["model"] for m in response.json().get("models", []) if m.get("model")]

    def __init__(self) -> None:
        super().__init__(
            name=self.OLLAMA_PROVIDER_NAME,
            display_name="Ollama",
            description="AI-powered assistant using a locally running Ollama server.",
            connection_hint="Make sure Ollama is running: ollama serve",
            list_models_fn=self._list_models,
            default_base_url="http://localhost:11434",
        )
