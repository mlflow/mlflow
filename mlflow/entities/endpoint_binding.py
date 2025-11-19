from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.service_pb2 import EndpointBinding as ProtoEndpointBinding

if TYPE_CHECKING:
    from mlflow.entities.endpoint_model import EndpointModel


class SecretResourceType(str, Enum):
    """
    Enum defining valid resource types that can have endpoints bound to them.

    This enum restricts what resources can use endpoints, providing type safety
    and preventing arbitrary resource types from being used.
    """

    SCORER_JOB = "SCORER_JOB"
    """LLM judge scorer jobs that need API keys for LLM providers."""

    GLOBAL = "GLOBAL"
    """Global workspace-level endpoints that aren't bound to specific resources."""

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
class EndpointBinding(_MlflowObject):
    """
    MLflow entity representing an Endpoint Binding.

    Endpoint bindings connect endpoints (with their configured models and secrets)
    to resources that need to use them. A binding defines where and how the endpoint
    should be made available to the resource.

    Args:
        binding_id: String containing binding ID (UUID).
        endpoint_id: String containing the endpoint ID this binding references.
        resource_type: String containing the type of resource this endpoint is bound to.
        resource_id: String containing the ID of the resource using this endpoint.
        field_name: String containing the field/variable name for this binding
            (e.g., "llm_endpoint", "OPENAI_API_KEY").
        created_at: Creation timestamp in milliseconds since the UNIX epoch.
        last_updated_at: Last update timestamp in milliseconds since the UNIX epoch.
        created_by: String containing the user ID who created the binding, or None.
        last_updated_by: String containing the user ID who last updated the binding, or None.
    """

    binding_id: str
    endpoint_id: str
    resource_type: str
    resource_id: str
    field_name: str
    created_at: int = 0
    last_updated_at: int = 0
    created_by: str | None = None
    last_updated_by: str | None = None

    def to_proto(self):
        proto = ProtoEndpointBinding()
        proto.binding_id = self.binding_id
        proto.endpoint_id = self.endpoint_id
        proto.resource_type = self.resource_type
        proto.resource_id = self.resource_id
        proto.field_name = self.field_name
        proto.created_at = self.created_at
        proto.last_updated_at = self.last_updated_at
        proto.created_by = self.created_by or ""
        proto.last_updated_by = self.last_updated_by or ""
        return proto

    @classmethod
    def from_proto(cls, proto):
        return cls(
            binding_id=proto.binding_id,
            endpoint_id=proto.endpoint_id,
            resource_type=proto.resource_type,
            resource_id=proto.resource_id,
            field_name=proto.field_name,
            created_at=proto.created_at,
            last_updated_at=proto.last_updated_at,
            created_by=proto.created_by or None,
            last_updated_by=proto.last_updated_by or None,
        )


@dataclass
class EndpointBindingListItem(EndpointBinding):
    """
    Endpoint binding with additional display information for list responses.

    Extends EndpointBinding with human-readable fields populated via JOIN
    with endpoints and models tables. Used by list_endpoint_bindings() to provide
    UI-friendly data without additional API calls.

    This shows users what capabilities the endpoint provides when bound to a resource.

    Args:
        endpoint_name: User-friendly endpoint name (e.g., "Production LLM Endpoint").
        endpoint_description: Optional description of the endpoint.
        models: List of EndpointModel entities configured in this endpoint.
    """

    endpoint_name: str = ""
    endpoint_description: str = ""
    models: list[EndpointModel] = field(default_factory=list)

    def to_proto(self):
        """Override to include additional display fields in the proto."""
        proto = super().to_proto()
        proto.endpoint_name = self.endpoint_name
        proto.endpoint_description = self.endpoint_description
        proto.models.extend([m.to_proto() for m in self.models])
        return proto

    @classmethod
    def from_proto(cls, proto):
        """Override to include additional display fields from the proto."""
        from mlflow.entities.endpoint_model import EndpointModel

        return cls(
            binding_id=proto.binding_id,
            endpoint_id=proto.endpoint_id,
            resource_type=proto.resource_type,
            resource_id=proto.resource_id,
            field_name=proto.field_name,
            created_at=proto.created_at,
            last_updated_at=proto.last_updated_at,
            created_by=proto.created_by or None,
            last_updated_by=proto.last_updated_by or None,
            endpoint_name=proto.endpoint_name if proto.HasField("endpoint_name") else "",
            endpoint_description=(
                proto.endpoint_description if proto.HasField("endpoint_description") else ""
            ),
            models=[EndpointModel.from_proto(m) for m in proto.models],
        )
