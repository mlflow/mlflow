"""
Internal PromptInfo entity for Unity Catalog prompt operations.

This is an implementation detail for the Unity Catalog store and should not be
considered part of the public MLflow API.
"""

from typing import Optional


class PromptInfo:
    """
    Internal entity for prompt information from Unity Catalog. This represents
    prompt metadata without version-specific details like template.

    This maps to the Unity Catalog PromptInfo protobuf message.

    Note: This is an internal implementation detail and not part of the public API.
    """

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        creation_timestamp: Optional[int] = None,
        tags: Optional[dict[str, str]] = None,
    ):
        """
        Construct a PromptInfo entity.

        Args:
            name: Name of the prompt.
            description: Description of the prompt.
            creation_timestamp: Timestamp when the prompt was created.
            tags: Prompt-level metadata as key-value pairs.
        """
        self._name = name
        self._description = description
        self._creation_timestamp = creation_timestamp
        self._tags = tags or {}

    @property
    def name(self) -> str:
        """The name of the prompt."""
        return self._name

    @property
    def description(self) -> Optional[str]:
        """The description of the prompt."""
        return self._description

    @property
    def creation_timestamp(self) -> Optional[int]:
        """The creation timestamp of the prompt."""
        return self._creation_timestamp

    @property
    def tags(self) -> dict[str, str]:
        """Prompt-level metadata as key-value pairs."""
        return self._tags.copy()

    def __eq__(self, other) -> bool:
        if not isinstance(other, PromptInfo):
            return False
        return (
            self.name == other.name
            and self.description == other.description
            and self.creation_timestamp == other.creation_timestamp
            and self.tags == other.tags
        )

    def __repr__(self) -> str:
        return (
            f"<PromptInfo: name='{self.name}', description='{self.description}', tags={self.tags}>"
        )
