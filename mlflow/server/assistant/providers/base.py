"""Base class for assistant providers."""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator


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
    def load_config(self) -> dict[str, Any]:
        """Load provider configuration."""

    @abstractmethod
    async def run(
        self,
        prompt: str,
        session_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Run the assistant and stream responses.

        Args:
            prompt: The prompt to send to the assistant
            session_id: Session ID for conversation continuity

        Yields:
            Event dictionaries with 'type' and 'data' keys.
            Types: 'message', 'done', 'error'
        """
        yield {}  # pragma: no cover
