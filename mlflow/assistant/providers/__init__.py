from mlflow.assistant.providers.base import AssistantProvider
from mlflow.assistant.providers.claude_code import ClaudeCodeProvider
from mlflow.assistant.providers.codex import CodexProvider
from mlflow.assistant.providers.mlflow_gateway import MlflowGatewayProvider
from mlflow.assistant.providers.ollama import OllamaProvider
from mlflow.assistant.providers.openai_compatible import OpenAICompatibleProvider

__all__ = [
    "AssistantProvider",
    "ClaudeCodeProvider",
    "CodexProvider",
    "MlflowGatewayProvider",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "list_providers",
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
