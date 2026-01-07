"""Base class for assistant providers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncGenerator, Callable

from pydantic import BaseModel

# Constant for .mlflow assistant directory
MLFLOW_ASSISTANT_HOME = Path.home() / ".mlflow" / "assistant"


class ProviderConfig(BaseModel):
    """Base configuration for assistant providers.

    This is a concrete Pydantic model that providers can subclass
    to add provider-specific validation.
    """


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

    @property
    @abstractmethod
    def config_path(self) -> Path:
        """Return the path to the provider's config file."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and ready to use."""

    @abstractmethod
    def load_config(self) -> ProviderConfig:
        """Load provider configuration.

        Returns:
            ProviderConfig subclass with validated configuration.
        """

    def check_connection(self, echo: Callable[[str], None] = print) -> None:
        """
        Check if the provider is properly configured and can connect.

        Args:
            echo: Function to print status messages. Defaults to print().

        Raises:
            RuntimeError: If the provider is not accessible.
        """


    @abstractmethod
    def install_skills(self) -> list[str]:
        """Install provider-specific skills.

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
