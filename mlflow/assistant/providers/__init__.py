import logging

from mlflow.assistant.providers.base import AssistantProvider
from mlflow.assistant.providers.claude_code import ClaudeCodeProvider
from mlflow.assistant.providers.codex import CodexProvider
from mlflow.assistant.providers.mlflow_gateway import MlflowGatewayProvider
from mlflow.assistant.providers.ollama import OllamaProvider
from mlflow.assistant.providers.openai_compatible import OpenAICompatibleProvider

_logger = logging.getLogger(__name__)

__all__ = [
    "AssistantProvider",
    "ClaudeCodeProvider",
    "CodexProvider",
    "MlflowGatewayProvider",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "list_providers",
    "resolve_default_provider",
]


def _build_providers() -> list[AssistantProvider]:
    return [
        ClaudeCodeProvider(),
        CodexProvider(),
        MlflowGatewayProvider(),
        OllamaProvider(),
    ]


def list_providers() -> list[AssistantProvider]:
    return _build_providers()


def _default_provider_precedence() -> list[AssistantProvider]:
    return [
        ClaudeCodeProvider(),
        CodexProvider(),
        MlflowGatewayProvider(),
    ]


def resolve_default_provider(remote: bool = False) -> AssistantProvider | None:
    """Pick a provider when the user has not selected one explicitly."""
    for provider in _default_provider_precedence():
        if remote and not provider.allows_remote_access:
            continue
        try:
            if provider.is_available():
                return provider
        except Exception:
            _logger.debug(
                "Availability probe failed for assistant provider %r", provider.name, exc_info=True
            )
    return None
