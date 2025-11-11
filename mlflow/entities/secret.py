from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.service_pb2 import Secret as ProtoSecret
from mlflow.protos.service_pb2 import SecretWithBinding as ProtoSecretWithBinding

if TYPE_CHECKING:
    from mlflow.entities.secret_binding import SecretBinding


@dataclass
class SecretWithBinding(_MlflowObject):
    """
    Result of creating a secret with its initial binding atomically.

    This structure is returned when creating a secret to ensure that every
    secret always has at least one binding, preventing orphaned secrets.

    Args:
        secret: The created Secret entity with metadata.
        binding: The initial SecretBinding that associates the secret with a resource.
    """

    secret: "Secret"
    binding: "SecretBinding"

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
        created_at: Creation timestamp in milliseconds since the UNIX epoch.
        last_updated_at: Last update timestamp in milliseconds since the UNIX epoch.
        created_by: String containing the user ID who created the secret, or None.
        last_updated_by: String containing the user ID who last updated the secret, or None.
        provider: LLM provider identifier (e.g., "anthropic", "openai", "cohere"), or None.
            Used for gateway model metadata.
        model: LLM model identifier (e.g., "claude-3-5-sonnet-20241022", "gpt-4-turbo"), or None.
            Used for gateway model metadata.
        binding_count: Number of resource bindings (only populated in list responses).
            For private secrets (is_shared=false), this is always 1.
            For shared secrets (is_shared=true), reflects actual count.
    """

    secret_id: str
    secret_name: str
    masked_value: str
    is_shared: bool
    created_at: int
    last_updated_at: int
    created_by: str | None = None
    last_updated_by: str | None = None
    provider: str | None = None
    model: str | None = None
    binding_count: int | None = None

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
        if self.model is not None:
            proto.model = self.model
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
            model=proto.model if proto.HasField("model") else None,
            binding_count=proto.binding_count if proto.HasField("binding_count") else None,
        )
