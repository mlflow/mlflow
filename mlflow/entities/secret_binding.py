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

    Secret bindings map routes (model configurations) to resources and define how the secret
    should be injected as environment variables. A binding connects a specific route
    (which combines a secret and model) to a resource that needs to use it.

    Args:
        binding_id: String containing binding ID (UUID).
        route_id: String containing the route ID this binding references.
        secret_id: String containing the secret ID (denormalized from route for convenience).
        resource_type: String containing the type of resource this secret is bound to.
        resource_id: String containing the ID of the resource using this secret.
        field_name: String containing the environment variable name (e.g., "OPENAI_API_KEY").
        created_at: Creation timestamp in milliseconds since the UNIX epoch.
        last_updated_at: Last update timestamp in milliseconds since the UNIX epoch.
        created_by: String containing the user ID who created the binding, or None.
        last_updated_by: String containing the user ID who last updated the binding, or None.
    """

    binding_id: str
    route_id: str
    secret_id: str
    resource_type: str
    resource_id: str
    field_name: str
    created_at: int = 0
    last_updated_at: int = 0
    created_by: str | None = None
    last_updated_by: str | None = None

    def to_proto(self):
        from mlflow.protos.service_pb2 import SecretBinding as ProtoSecretBinding

        return ProtoSecretBinding(
            binding_id=self.binding_id,
            route_id=self.route_id,
            resource_type=self.resource_type,
            resource_id=self.resource_id,
            field_name=self.field_name,
            created_at=self.created_at,
            last_updated_at=self.last_updated_at,
            created_by=self.created_by or "",
            last_updated_by=self.last_updated_by or "",
        )

    @classmethod
    def from_proto(cls, proto):
        return cls(
            binding_id=proto.binding_id,
            route_id=proto.route_id,
            secret_id=proto.secret_id if proto.HasField("secret_id") else "",
            resource_type=proto.resource_type,
            resource_id=proto.resource_id,
            field_name=proto.field_name,
            created_at=proto.created_at,
            last_updated_at=proto.last_updated_at,
            created_by=proto.created_by if proto.HasField("created_by") else None,
            last_updated_by=proto.last_updated_by if proto.HasField("last_updated_by") else None,
        )


@dataclass
class SecretBindingListItem(SecretBinding):
    """
    SecretBinding with additional display information for list responses.

    Extends SecretBinding with human-readable fields populated via JOIN
    with secrets and routes tables. Used by list_secret_bindings() to provide
    UI-friendly data without additional API calls.

    Args:
        secret_name: User-friendly secret name (e.g., "company_openai_key").
        route_name: Route display name or model_name if no display name set.
        provider: LLM provider (e.g., "openai", "anthropic", "google").
    """

    secret_name: str = ""
    route_name: str = ""
    provider: str = ""
