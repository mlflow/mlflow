"""
Prompt entity for MLflow Model Registry.

This represents a prompt in the registry with its metadata, without version-specific
content like template text. For version-specific content, use PromptVersion.
"""


class Prompt:
    """
    Entity representing a prompt in the MLflow Model Registry.

    This contains prompt-level information (name, description, tags) but not version-specific
    content. To access version-specific content like the template, use PromptVersion.
    """

    def __init__(
        self,
        name: str,
        description: str | None = None,
        creation_timestamp: int | None = None,
        tags: dict[str, str] | None = None,
    ):
        """
        Construct a Prompt entity.

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
    def description(self) -> str | None:
        """The description of the prompt."""
        return self._description

    @property
    def creation_timestamp(self) -> int | None:
        """The creation timestamp of the prompt."""
        return self._creation_timestamp

    @property
    def tags(self) -> dict[str, str]:
        """Prompt-level metadata as key-value pairs."""
        return self._tags.copy()

    def __eq__(self, other) -> bool:
        if not isinstance(other, Prompt):
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
