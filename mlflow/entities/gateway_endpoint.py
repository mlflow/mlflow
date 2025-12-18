from dataclasses import dataclass, field
from enum import Enum

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.service_pb2 import (
    GatewayEndpoint as ProtoGatewayEndpoint,
)
from mlflow.protos.service_pb2 import (
    GatewayEndpointBinding as ProtoGatewayEndpointBinding,
)
from mlflow.protos.service_pb2 import (
    GatewayEndpointModelMapping as ProtoGatewayEndpointModelMapping,
)
from mlflow.protos.service_pb2 import (
    GatewayModelDefinition as ProtoGatewayModelDefinition,
)


class GatewayResourceType(str, Enum):
    """Valid MLflow resource types that can use gateway endpoints."""

    SCORER_JOB = "scorer_job"


@dataclass
class GatewayModelDefinition(_MlflowObject):
    """
    Represents a reusable LLM model configuration.

    Model definitions can be shared across multiple endpoints, enabling
    centralized management of model configurations and API credentials.

    Args:
        model_definition_id: Unique identifier for this model definition.
        name: User-friendly name for identification and reuse.
        secret_id: ID of the secret containing authentication credentials (None if orphaned).
        secret_name: Name of the secret for display/reference purposes (None if orphaned).
        provider: LLM provider (e.g., "openai", "anthropic", "cohere", "bedrock").
        model_name: Provider-specific model identifier (e.g., "gpt-4o", "claude-3-5-sonnet").
        created_at: Timestamp (milliseconds) when the model definition was created.
        last_updated_at: Timestamp (milliseconds) when the model definition was last updated.
        created_by: User ID who created the model definition.
        last_updated_by: User ID who last updated the model definition.
    """

    model_definition_id: str
    name: str
    secret_id: str | None
    secret_name: str | None
    provider: str
    model_name: str
    created_at: int
    last_updated_at: int
    created_by: str | None = None
    last_updated_by: str | None = None

    def to_proto(self):
        proto = ProtoGatewayModelDefinition()
        proto.model_definition_id = self.model_definition_id
        proto.name = self.name
        if self.secret_id is not None:
            proto.secret_id = self.secret_id
        if self.secret_name is not None:
            proto.secret_name = self.secret_name
        proto.provider = self.provider
        proto.model_name = self.model_name
        proto.created_at = self.created_at
        proto.last_updated_at = self.last_updated_at
        if self.created_by is not None:
            proto.created_by = self.created_by
        if self.last_updated_by is not None:
            proto.last_updated_by = self.last_updated_by
        return proto

    @classmethod
    def from_proto(cls, proto):
        return cls(
            model_definition_id=proto.model_definition_id,
            name=proto.name,
            secret_id=proto.secret_id or None,
            secret_name=proto.secret_name or None,
            provider=proto.provider,
            model_name=proto.model_name,
            created_at=proto.created_at,
            last_updated_at=proto.last_updated_at,
            created_by=proto.created_by or None,
            last_updated_by=proto.last_updated_by or None,
        )


@dataclass
class GatewayEndpointModelMapping(_MlflowObject):
    """
    Represents a mapping between an endpoint and a model definition.

    This is a junction entity that links endpoints to model definitions,
    enabling many-to-many relationships and traffic routing configuration.

    Args:
        mapping_id: Unique identifier for this mapping.
        endpoint_id: ID of the endpoint.
        model_definition_id: ID of the model definition.
        model_definition: The full model definition (populated via JOIN).
        weight: Routing weight for traffic distribution (default 1).
        created_at: Timestamp (milliseconds) when the mapping was created.
        created_by: User ID who created the mapping.
    """

    mapping_id: str
    endpoint_id: str
    model_definition_id: str
    model_definition: GatewayModelDefinition | None
    weight: float
    created_at: int
    created_by: str | None = None

    def to_proto(self):
        proto = ProtoGatewayEndpointModelMapping()
        proto.mapping_id = self.mapping_id
        proto.endpoint_id = self.endpoint_id
        proto.model_definition_id = self.model_definition_id
        if self.model_definition is not None:
            proto.model_definition.CopyFrom(self.model_definition.to_proto())
        proto.weight = self.weight
        proto.created_at = self.created_at
        if self.created_by is not None:
            proto.created_by = self.created_by
        return proto

    @classmethod
    def from_proto(cls, proto):
        model_def = None
        if proto.HasField("model_definition"):
            model_def = GatewayModelDefinition.from_proto(proto.model_definition)
        return cls(
            mapping_id=proto.mapping_id,
            endpoint_id=proto.endpoint_id,
            model_definition_id=proto.model_definition_id,
            model_definition=model_def,
            weight=proto.weight,
            created_at=proto.created_at,
            created_by=proto.created_by or None,
        )


@dataclass
class GatewayEndpointTag(_MlflowObject):
    """
    Represents a tag (key-value pair) associated with a gateway endpoint.

    Tags are used for categorization, filtering, and metadata storage for endpoints.

    Args:
        key: Tag key (max 250 characters).
        value: Tag value (max 5000 characters, can be None).
    """

    key: str
    value: str | None

    def to_proto(self):
        from mlflow.protos.service_pb2 import GatewayEndpointTag as ProtoGatewayEndpointTag

        proto = ProtoGatewayEndpointTag()
        proto.key = self.key
        if self.value is not None:
            proto.value = self.value
        return proto

    @classmethod
    def from_proto(cls, proto):
        return cls(
            key=proto.key,
            value=proto.value or None,
        )


@dataclass
class GatewayEndpoint(_MlflowObject):
    """
    Represents an LLM gateway endpoint with its associated model configurations.

    Args:
        endpoint_id: Unique identifier for this endpoint.
        name: User-friendly name for the endpoint (optional).
        created_at: Timestamp (milliseconds) when the endpoint was created.
        last_updated_at: Timestamp (milliseconds) when the endpoint was last updated.
        model_mappings: List of model mappings bound to this endpoint.
        tags: List of tags associated with this endpoint.
        created_by: User ID who created the endpoint.
        last_updated_by: User ID who last updated the endpoint.
    """

    endpoint_id: str
    name: str | None
    created_at: int
    last_updated_at: int
    model_mappings: list[GatewayEndpointModelMapping] = field(default_factory=list)
    tags: list["GatewayEndpointTag"] = field(default_factory=list)
    created_by: str | None = None
    last_updated_by: str | None = None

    def to_proto(self):
        proto = ProtoGatewayEndpoint()
        proto.endpoint_id = self.endpoint_id
        proto.name = self.name or ""
        proto.created_at = self.created_at
        proto.last_updated_at = self.last_updated_at
        proto.model_mappings.extend([m.to_proto() for m in self.model_mappings])
        proto.tags.extend([t.to_proto() for t in self.tags])
        proto.created_by = self.created_by or ""
        proto.last_updated_by = self.last_updated_by or ""
        return proto

    @classmethod
    def from_proto(cls, proto):
        return cls(
            endpoint_id=proto.endpoint_id,
            name=proto.name or None,
            created_at=proto.created_at,
            last_updated_at=proto.last_updated_at,
            model_mappings=[
                GatewayEndpointModelMapping.from_proto(m) for m in proto.model_mappings
            ],
            tags=[GatewayEndpointTag.from_proto(t) for t in proto.tags],
            created_by=proto.created_by or None,
            last_updated_by=proto.last_updated_by or None,
        )


@dataclass
class GatewayEndpointBinding(_MlflowObject):
    """
    Represents a binding between an endpoint and an MLflow resource.

    Bindings track which MLflow resources (e.g., scorer jobs) are configured to use
    which endpoints. The composite key (endpoint_id, resource_type, resource_id) uniquely
    identifies each binding.

    Args:
        endpoint_id: ID of the endpoint this binding references.
        resource_type: Type of MLflow resource (e.g., "scorer_job").
        resource_id: ID of the specific resource instance.
        created_at: Timestamp (milliseconds) when the binding was created.
        last_updated_at: Timestamp (milliseconds) when the binding was last updated.
        created_by: User ID who created the binding.
        last_updated_by: User ID who last updated the binding.
    """

    endpoint_id: str
    resource_type: GatewayResourceType
    resource_id: str
    created_at: int
    last_updated_at: int
    created_by: str | None = None
    last_updated_by: str | None = None

    def to_proto(self):
        proto = ProtoGatewayEndpointBinding()
        proto.endpoint_id = self.endpoint_id
        proto.resource_type = self.resource_type.value
        proto.resource_id = self.resource_id
        proto.created_at = self.created_at
        proto.last_updated_at = self.last_updated_at
        if self.created_by is not None:
            proto.created_by = self.created_by
        if self.last_updated_by is not None:
            proto.last_updated_by = self.last_updated_by
        return proto

    @classmethod
    def from_proto(cls, proto):
        return cls(
            endpoint_id=proto.endpoint_id,
            resource_type=GatewayResourceType(proto.resource_type),
            resource_id=proto.resource_id,
            created_at=proto.created_at,
            last_updated_at=proto.last_updated_at,
            created_by=proto.created_by or None,
            last_updated_by=proto.last_updated_by or None,
        )
