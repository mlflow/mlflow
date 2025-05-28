"""
The prompt info entity for Unity Catalog MLflow Model Registry.
"""

from typing import Dict, List, Optional


class PromptInfo:
    """
    MLflow entity for prompt information from Unity Catalog. This represents
    prompt metadata without version-specific details like template.
    
    This maps to the Unity Catalog PromptInfo protobuf message.
    """

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        creation_timestamp: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
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
    def tags(self) -> Dict[str, str]:
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
            f"<PromptInfo: name='{self.name}', description='{self.description}', "
            f"tags={self.tags}>"
        ) 