import shutil
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Literal

from mlflow.assistant.config import AssistantConfig, ProviderConfig
from mlflow.assistant.types import Event
from mlflow.environment_variables import (
    MLFLOW_ENABLE_ASSISTANT_SANDBOX,
    MLFLOW_ENABLE_REMOTE_ASSISTANT,
)

ClientToolDelivery = Literal["tool", "structured", "unsupported"]


def assistant_sandbox_enabled() -> bool:
    """Whether the assistant should run untrusted work in a Docker sandbox instead of on the host.

    ``MLFLOW_ENABLE_ASSISTANT_SANDBOX`` is a tri-state override: ``true`` forces the sandbox on and
    ``false`` forces it off (letting an operator opt out), regardless of the deployment. When it is
    unset (the default), sandboxing is derived: on for a remote/multi-user server
    (``MLFLOW_ENABLE_REMOTE_ASSISTANT``) that has a ``docker`` executable on PATH, off otherwise (a
    local server runs the work in a host subprocess, as before).

    Only the presence of the ``docker`` CLI is probed here — cheap and non-blocking, so it is safe
    to call from provider availability checks; an unreachable daemon surfaces later, when the
    container is actually started.
    """
    if (override := MLFLOW_ENABLE_ASSISTANT_SANDBOX.get()) is not None:
        return override
    return MLFLOW_ENABLE_REMOTE_ASSISTANT.get() and shutil.which("docker") is not None


@lru_cache(maxsize=10)
def load_config(name: str) -> ProviderConfig:
    cfg = AssistantConfig.load()
    if not cfg or name not in cfg.providers:
        raise RuntimeError(f"Provider configuration not found for {name}")
    return cfg.providers[name]


def clear_config_cache() -> None:
    """Clear the config cache to pick up config changes."""
    load_config.cache_clear()


def load_config_or_default(name: str) -> ProviderConfig:
    try:
        return load_config(name)
    except RuntimeError:
        return ProviderConfig()


class ProviderNotConfiguredError(Exception):
    """Raised when a provider is not properly configured."""


class CLINotInstalledError(ProviderNotConfiguredError):
    """Raised when the provider CLI is not installed."""


class NotAuthenticatedError(ProviderNotConfiguredError):
    """Raised when the user is not authenticated with the provider."""


class AssistantProvider(ABC):
    """Abstract base class for assistant providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider identifier (e.g., 'claude_code')."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Return the human-readable provider name (e.g., 'Claude Code')."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a short description of the provider."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and ready to use."""

    @property
    def allows_remote_access(self) -> bool:
        """Whether this provider can serve requests from remote clients."""
        return False

    @property
    def client_tool_delivery(self) -> ClientToolDelivery:
        """How this provider delivers actions executed by the client.

        ``tool`` pauses and resumes around a native client-tool call, ``structured`` encodes the
        action in a schema-constrained final response, and ``unsupported`` cannot request
        client-executed tools.
        """
        return "unsupported"

    @abstractmethod
    def check_connection(self, echo: Callable[[str], None] | None = None) -> None:
        """
        Check if the provider is properly configured and can connect.

        Args:
            echo: Optional function to print status messages.

        Raises:
            ProviderNotConfiguredError: If the provider is not properly configured.
        """

    @abstractmethod
    def resolve_skills_path(self, base_directory: Path) -> Path:
        """Resolve the skills installation path.

        Args:
            base_directory: Base directory to resolve skills path from.

        Returns:
            Resolved absolute path for skills installation.
        """

    def list_models(self, base_url: str | None = None, api_key: str | None = None) -> list[str]:
        raise NotImplementedError(f"Model listing is not supported for provider '{self.name}'")

    @abstractmethod
    def astream(
        self,
        prompt: str,
        tracking_uri: str,
        session_id: str | None = None,
        mlflow_session_id: str | None = None,
        cwd: Path | None = None,
        context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[Event, None]:
        """
        Stream responses from the assistant asynchronously.

        Args:
            prompt: The prompt to send to the assistant
            tracking_uri: MLflow tracking server URI for the assistant to use
            session_id: Session ID for conversation continuity
            mlflow_session_id: MLflow session ID for process tracking / cancellation
            cwd: Working directory for the assistant
            context: Additional context for the assistant, such as information from
                the current UI page the user is viewing (e.g., experimentId, traceId)

        Yields:
            Event objects with 'type' and 'data' payloads.
        """
