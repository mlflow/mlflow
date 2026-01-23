from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Literal

from mlflow.assistant.config import AssistantConfig, ProviderConfig


@lru_cache(maxsize=10)
def load_config(name: str) -> ProviderConfig:
    cfg = AssistantConfig.load()
    if not cfg or name not in cfg.providers:
        raise RuntimeError(f"Provider configuration not found for {name}")
    return cfg.providers[name]


def clear_config_cache() -> None:
    """Clear the config cache to pick up config changes."""
    load_config.cache_clear()


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
    def resolve_skills_path(
        self,
        skills_type: Literal["global", "project", "custom"],
        custom_path: str | None = None,
        project_path: Path | None = None,
    ) -> Path:
        """Resolve the skills installation path based on type.

        Args:
            skills_type: 'global', 'project', or 'custom'
            custom_path: User-provided path (required if type='custom')
            project_path: Current project path (used if type='project')

        Returns:
            Resolved absolute path for skills installation.
        """

    @abstractmethod
    def install_skills(self, skill_path: Path) -> list[str]:
        """Install provider-specific skills.

        Args:
            skill_path: Directory where skills should be installed.

        Returns:
            List of installed skill names.
        """

    @abstractmethod
    def astream(
        self,
        prompt: str,
        session_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream responses from the assistant asynchronously.

        Args:
            prompt: The prompt to send to the assistant
            session_id: Session ID for conversation continuity

        Yields:
            Event dictionaries with 'type' and 'data' keys.
            Event types: 'message', 'status', 'done', 'error'
        """
