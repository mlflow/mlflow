from mlflow.assistant.providers.base import AssistantProvider
from mlflow.assistant.providers.claude_code import ClaudeCodeProvider
from mlflow.assistant.providers.ollama import OllamaProvider

__all__ = [
    "AssistantProvider",
    "ClaudeCodeProvider",
    "OllamaProvider",
    "list_providers",
]

_PROVIDERS: list[type[AssistantProvider]] = [
    ClaudeCodeProvider,
    OllamaProvider,
]


def list_providers() -> list[AssistantProvider]:
    return [provider() for provider in _PROVIDERS]
