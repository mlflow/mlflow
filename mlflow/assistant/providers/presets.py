"""Model-listing strategies for the OpenAI-compatible provider presets."""

import requests

_TIMEOUT_SECONDS = 10
_OLLAMA_TAGS_PATH = "/api/tags"


def list_ollama_tags(base_url: str, api_key: str | None = None) -> list[str]:
    """List models from a local Ollama server via `GET /api/tags`.

    Vanilla Ollama is auth-free, but the api_key is forwarded as a Bearer
    token when set so users who reverse-proxy Ollama behind an auth layer
    can still list models.
    """
    url = f"{base_url.rstrip('/')}{_OLLAMA_TAGS_PATH}"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = requests.get(url, headers=headers, timeout=_TIMEOUT_SECONDS)
    response.raise_for_status()
    return [m["model"] for m in response.json().get("models", []) if m.get("model")]
