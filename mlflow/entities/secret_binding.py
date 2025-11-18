from dataclasses import dataclass
from enum import Enum

from mlflow.entities._mlflow_object import _MlflowObject


class SecretResourceType(str, Enum):
    """
    Enum defining valid resource types that can have secrets bound to them.

    This enum restricts what resources can use secrets, providing type safety
    and preventing arbitrary resource types from being used.
    """

    SCORER_JOB = "SCORER_JOB"
    """LLM judge scorer jobs that need API keys for LLM providers."""

    GLOBAL = "GLOBAL"
    """Global workspace-level secrets that aren't bound to specific resources."""

    @classmethod
    def from_string(cls, resource_type_str: str) -> "SecretResourceType":
        """
        Convert a string to a SecretResourceType enum.

        Args:
            resource_type_str: String representation of the resource type.

        Returns:
            SecretResourceType enum value.

        Raises:
            ValueError: If the resource type string is not valid.
        """
        try:
            return cls(resource_type_str)
        except ValueError:
            valid_types = ", ".join([t.value for t in cls])
            raise ValueError(
                f"Invalid resource type: '{resource_type_str}'. Valid types are: {valid_types}"
            )

    def __str__(self) -> str:
        """Return the string value of the enum."""
        return self.value


@dataclass
class SecretBinding(_MlflowObject):
    """
    MLflow entity representing a Secret Binding.

    Secret bindings map secrets to resources and define how the secret should be injected
    as environment variables. This allows secrets to be reused across multiple resources
    when marked as shared.

    Args:
        binding_id: String containing binding ID (UUID).
        secret_id: String containing the secret ID this binding references.
        resource_type: String containing the type of resource this secret is bound to.
        resource_id: String containing the ID of the resource using this secret.
        field_name: String containing the environment variable name (e.g., "OPENAI_API_KEY").
        created_at: Creation timestamp in milliseconds since the UNIX epoch.
        last_updated_at: Last update timestamp in milliseconds since the UNIX epoch.
        created_by: String containing the user ID who created the binding, or None.
        last_updated_by: String containing the user ID who last updated the binding, or None.
    """

    binding_id: str
    secret_id: str
    resource_type: str
    resource_id: str
    field_name: str
    created_at: int = 0
    last_updated_at: int = 0
    created_by: str | None = None
    last_updated_by: str | None = None
