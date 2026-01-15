from mlflow.assistant.providers.base import AssistantProvider
from mlflow.assistant.providers.claude_code import ClaudeCodeProvider

__all__ = ["AssistantProvider", "ClaudeCodeProvider", "list_providers"]

# Registry of all available providers
_PROVIDERS: list[type[AssistantProvider]] = [
    ClaudeCodeProvider,
]


def list_providers() -> list[AssistantProvider]:
    """Return instances of all available providers."""
    return [provider() for provider in _PROVIDERS]
