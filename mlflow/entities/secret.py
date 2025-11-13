from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.service_pb2 import Secret as ProtoSecret
from mlflow.protos.service_pb2 import SecretWithBinding as ProtoSecretWithBinding

if TYPE_CHECKING:
    from mlflow.entities.endpoint import Endpoint
    from mlflow.entities.endpoint_binding import EndpointBinding
    from mlflow.entities.secret_tag import SecretTag


class SecretState(str, Enum):
    """Enum representing the state of a secret.

    - ``ACTIVE``: Secret is active and can be used.
    - ``REVOKED``: Secret has been revoked and can no longer be used.
    - ``ROTATED``: Secret has been rotated to a new version and is inactive.
    """

    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"
    ROTATED = "ROTATED"

    def __str__(self):
        return self.value


@dataclass
class SecretWithEndpointAndBinding(_MlflowObject):
    """
    Result of atomically creating a gateway asset (secret + endpoint + binding).

    This structure represents the complete gateway configuration:
    - Secret: The API key/credential
    - Endpoint: The model configuration (provider + model using that secret)
    - Binding: The resource binding (which service uses this endpoint)

    This ensures that secrets are always created with an endpoint configuration
    and an initial binding, preventing orphaned secrets or endpoints.

    Args:
        secret: The created Secret entity with metadata (API key).
        endpoint: The created Endpoint entity (model configuration).
        binding: The initial EndpointBinding that associates the endpoint with a resource.
    """

    secret: "Secret"
    endpoint: "Endpoint"
    binding: "EndpointBinding"

    def to_proto(self):
        from mlflow.protos.service_pb2 import (
            SecretWithRouteAndBinding as ProtoSecretWithRouteAndBinding,
        )

        proto = ProtoSecretWithRouteAndBinding()
        proto.secret.CopyFrom(self.secret.to_proto())
        proto.route.CopyFrom(self.route.to_proto())
        proto.binding.CopyFrom(self.binding.to_proto())
        return proto

    @classmethod
    def from_proto(cls, proto):
        from mlflow.entities.secret_binding import SecretBinding
        from mlflow.entities.secret_route import SecretRoute

        return cls(
            secret=Secret.from_proto(proto.secret),
            route=SecretRoute.from_proto(proto.route),
            binding=SecretBinding.from_proto(proto.binding),
        )


@dataclass
class Secret(_MlflowObject):
    """
    MLflow entity representing a Secret.

    Secrets store encrypted credentials (e.g., API keys) using envelope encryption.
    This entity contains only metadata about the secret - cryptographic fields
    (encrypted_value, wrapped_dek, etc.) are never exposed outside the store layer.

    Args:
        secret_id: String containing secret ID (UUID).
        secret_name: String containing the unique secret name.
        masked_value: String showing partial secret for identification (e.g., "sk-...xyz123").
            Shows prefix (3-4 chars) and suffix (last 4 chars) with "..." in between.
            Helps users identify shared secrets without exposing the full value.
        is_shared: Boolean indicating if secret can be reused across resources.
            True: Shared secret (can be bound to multiple resources)
            False: Private secret (bound to single resource)
        state: The secret state as a SecretState enum.
            Possible values: SecretState.ACTIVE, SecretState.REVOKED, SecretState.ROTATED
        created_at: Creation timestamp in milliseconds since the UNIX epoch.
        last_updated_at: Last update timestamp in milliseconds since the UNIX epoch.
        created_by: String containing the user ID who created the secret, or None.
        last_updated_by: String containing the user ID who last updated the secret, or None.
        provider: LLM provider identifier (e.g., "anthropic", "openai", "cohere"), or None.
            Used for gateway route filtering. The provider is set at the secret level,
            and all routes for this secret must use models from the same provider.
        binding_count: Number of resource bindings (only populated in list responses).
            For private secrets (is_shared=false), this is always 1.
            For shared secrets (is_shared=true), reflects actual count.
        tags: List of SecretTag objects associated with this secret.
    """

    secret_id: str
    secret_name: str
    masked_value: str
    is_shared: bool
    created_at: int
    last_updated_at: int
    state: SecretState = SecretState.ACTIVE
    created_by: str | None = None
    last_updated_by: str | None = None
    provider: str | None = None
    binding_count: int | None = None
    tags: list[SecretTag] = field(default_factory=list)

    def to_proto(self):
        proto = ProtoSecret()
        proto.secret_id = self.secret_id
        proto.secret_name = self.secret_name
        proto.masked_value = self.masked_value
        proto.is_shared = self.is_shared
        proto.created_at = self.created_at
        proto.last_updated_at = self.last_updated_at
        if self.created_by is not None:
            proto.created_by = self.created_by
        if self.last_updated_by is not None:
            proto.last_updated_by = self.last_updated_by
        if self.provider is not None:
            proto.provider = self.provider
        if self.binding_count is not None:
            proto.binding_count = self.binding_count
        return proto

    @classmethod
    def from_proto(cls, proto):
        return cls(
            secret_id=proto.secret_id,
            secret_name=proto.secret_name,
            masked_value=proto.masked_value,
            is_shared=proto.is_shared,
            created_at=proto.created_at,
            last_updated_at=proto.last_updated_at,
            created_by=proto.created_by if proto.HasField("created_by") else None,
            last_updated_by=proto.last_updated_by if proto.HasField("last_updated_by") else None,
            provider=proto.provider if proto.HasField("provider") else None,
            binding_count=proto.binding_count if proto.HasField("binding_count") else None,
        )


@dataclass
class SecretWithBinding(_MlflowObject):
    """
    DEPRECATED: Use SecretWithRouteAndBinding instead.

    This class is kept for backward compatibility only.
    """

    secret: Secret
    binding: SecretBinding

    def to_proto(self):
        proto = ProtoSecretWithBinding()
        proto.secret.CopyFrom(self.secret.to_proto())
        proto.binding.CopyFrom(self.binding.to_proto())
        return proto

    @classmethod
    def from_proto(cls, proto):
        from mlflow.entities.secret_binding import SecretBinding

        return cls(
            secret=Secret.from_proto(proto.secret),
            binding=SecretBinding.from_proto(proto.binding),
        )
