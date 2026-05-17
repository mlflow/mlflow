from mlflow.assistant.providers.base import AssistantProvider
from mlflow.assistant.providers.claude_code import ClaudeCodeProvider
from mlflow.assistant.providers.codex import CodexProvider
from mlflow.assistant.providers.openai_compatible import OpenAICompatibleProvider
from mlflow.assistant.providers.presets import list_ollama_tags

__all__ = [
    "AssistantProvider",
    "ClaudeCodeProvider",
    "CodexProvider",
    "OpenAICompatibleProvider",
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


def _build_providers() -> list[AssistantProvider]:
    return [
        ClaudeCodeProvider(),
        CodexProvider(),
        OpenAICompatibleProvider(
            name="mlflow_gateway",
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
            list_models_fn=list_ollama_tags,
            default_base_url="http://localhost:11434",
        ),
    ]


def list_providers() -> list[AssistantProvider]:
    return _build_providers()
