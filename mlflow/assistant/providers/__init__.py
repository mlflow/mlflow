import requests

from mlflow.assistant.providers.base import AssistantProvider
from mlflow.assistant.providers.claude_code import ClaudeCodeProvider
from mlflow.assistant.providers.codex import CodexProvider
from mlflow.assistant.providers.openai_compatible import OpenAICompatibleProvider

# Provider name for the in-server MLflow AI Gateway backend. The frontend
# mirrors this literal in `server/js/src/assistant/constants.ts`
# (GATEWAY_PROVIDER_ID); keep the two in sync.
GATEWAY_PROVIDER_NAME = "mlflow_gateway"


__all__ = [
    "AssistantProvider",
    "ClaudeCodeProvider",
    "CodexProvider",
    "OpenAICompatibleProvider",
    "GATEWAY_PROVIDER_NAME",
    "list_providers",
]


def _gateway_chat_url(_base_url: str | None, tracking_uri: str) -> str | None:
    """The in-server MLflow Gateway is reachable through the same MLflow server,
    so the chat URL is derived from the tracking URI instead of a separate
    base_url stored in config.
    """
    if not tracking_uri:
        return None
    return f"{tracking_uri.rstrip('/')}/gateway/mlflow/v1/chat/completions"


def _list_ollama_tags(base_url: str, api_key: str | None = None) -> list[str]:
    """List models from a local Ollama server via `GET /api/tags`.

    Vanilla Ollama is auth-free, but the api_key is forwarded as a Bearer
    token when set so users who reverse-proxy Ollama behind an auth layer
    can still list models.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = requests.get(f"{base_url.rstrip('/')}/api/tags", headers=headers, timeout=10)
    response.raise_for_status()
    return [m["model"] for m in response.json().get("models", []) if m.get("model")]


def _build_providers() -> list[AssistantProvider]:
    return [
        ClaudeCodeProvider(),
        CodexProvider(),
        OpenAICompatibleProvider(
            name=GATEWAY_PROVIDER_NAME,
            display_name="MLflow AI Gateway",
            description=(
                "AI-powered assistant backed by an MLflow AI Gateway endpoint "
                "configured on this server."
            ),
            connection_hint=(
                "Configure an LLM chat endpoint on the MLflow AI Gateway and select it."
            ),
            chat_url_builder=_gateway_chat_url,
        ),
        OpenAICompatibleProvider(
            name="ollama",
            display_name="Ollama",
            description="AI-powered assistant using a locally running Ollama server.",
            connection_hint="Make sure Ollama is running: ollama serve",
            list_models_fn=_list_ollama_tags,
            default_base_url="http://localhost:11434",
        ),
    ]


def list_providers() -> list[AssistantProvider]:
    return _build_providers()
