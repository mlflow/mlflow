from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncGenerator

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
        """Return the provider name."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and ready to use."""

    @abstractmethod
    def load_config(self) -> ProviderConfig:
        """Load provider configuration.

        Returns:
            ProviderConfig subclass with validated configuration.
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
