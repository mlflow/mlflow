"""Model-listing strategies for the OpenAI-compatible provider presets."""

import requests

_TIMEOUT_SECONDS = 10
_OLLAMA_TAGS_PATH = "/api/tags"


def list_ollama_tags(base_url: str, api_key: str | None = None) -> list[str]:
    """List models from a local Ollama server via `GET /api/tags`."""
    url = f"{base_url.rstrip('/')}{_OLLAMA_TAGS_PATH}"
    response = requests.get(url, timeout=_TIMEOUT_SECONDS)
    response.raise_for_status()
    return [m["model"] for m in response.json().get("models", []) if m.get("model")]
