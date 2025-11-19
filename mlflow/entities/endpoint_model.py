from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.service_pb2 import EndpointModel as ProtoEndpointModel


@dataclass
class EndpointModelSpec:
    """
    Specification for creating a model within an endpoint.

    This is used as input to endpoint creation and model addition operations.
    It defines the contract for what's required to add a model to an endpoint.

    Args:
        model_name: Model identifier (e.g., "gpt-4-turbo", "claude-3-5-sonnet").
        secret_id: ID of the secret (API key) to use for this model.
        routing_config: Optional routing configuration dict (e.g., {"weight": 0.5}).
    """

    model_name: str
    secret_id: str
    routing_config: dict[str, Any] | None = None


@dataclass
class EndpointModel(_MlflowObject):
    """
    MLflow entity representing a Model within an Endpoint.

    An Endpoint can contain multiple models for routing/failover scenarios.
    Each model has its own secret (API key) for authentication.

    This entity contains only metadata about the model - cryptographic fields
    (encrypted_model_config, wrapped_model_config_dek) are never exposed outside the store layer.

    Args:
        model_id: String containing model ID (UUID).
        endpoint_id: String containing the parent endpoint ID (UUID).
        model_name: String containing the model identifier.
            E.g., "claude-3-5-sonnet-20241022", "gpt-4-turbo", "gemini-2.5-pro".
        secret_id: String containing the secret ID (UUID) used for this model's authentication.
        routing_config: Optional JSON string containing flexible routing configuration.
            Examples: {"weight": 0.5}, {"priority": 1}, {"strategy": "round_robin"}
        created_at: Creation timestamp in milliseconds since the UNIX epoch.
        last_updated_at: Last update timestamp in milliseconds since the UNIX epoch.
        created_by: String containing the user ID who created the model, or None.
        last_updated_by: String containing the user ID who last updated the model, or None.
        secret_name: String containing the secret name (for UI display). Optional.
        provider: String containing the LLM provider (for UI display). Optional.
    """

    model_id: str
    endpoint_id: str
    model_name: str
    secret_id: str
    created_at: int
    last_updated_at: int
    routing_config: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    secret_name: str = ""
    provider: str = ""

    def to_proto(self):
        """Convert EndpointModel entity to protobuf message."""
        proto = ProtoEndpointModel()
        proto.model_id = self.model_id
        proto.endpoint_id = self.endpoint_id
        proto.model_name = self.model_name
        proto.secret_id = self.secret_id
        proto.created_at = self.created_at
        proto.last_updated_at = self.last_updated_at
        if self.routing_config is not None:
            proto.routing_config = self.routing_config
        if self.created_by is not None:
            proto.created_by = self.created_by
        if self.last_updated_by is not None:
            proto.last_updated_by = self.last_updated_by
        proto.secret_name = self.secret_name
        proto.provider = self.provider
        return proto

    @classmethod
    def from_proto(cls, proto):
        """Create EndpointModel entity from protobuf message."""
        return cls(
            model_id=proto.model_id,
            endpoint_id=proto.endpoint_id,
            model_name=proto.model_name,
            secret_id=proto.secret_id,
            created_at=proto.created_at,
            last_updated_at=proto.last_updated_at,
            routing_config=proto.routing_config if proto.HasField("routing_config") else None,
            created_by=proto.created_by if proto.HasField("created_by") else None,
            last_updated_by=proto.last_updated_by if proto.HasField("last_updated_by") else None,
            secret_name=proto.secret_name if proto.HasField("secret_name") else "",
            provider=proto.provider if proto.HasField("provider") else "",
        )
