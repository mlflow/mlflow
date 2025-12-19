import json
from dataclasses import dataclass
from typing import Any

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.service_pb2 import GatewaySecretInfo as ProtoGatewaySecretInfo


@dataclass(frozen=True)
class GatewaySecretInfo(_MlflowObject):
    """
    Metadata about an encrypted secret for authenticating with LLM providers.

    This entity contains metadata, masked value, and auth configuration of a secret,
    but NOT the decrypted secret value itself. The actual secret is stored encrypted
    using envelope encryption (DEK encrypted by KEK).

    NB: secret_id and secret_name are IMMUTABLE after creation. They are used as AAD
    (Additional Authenticated Data) during AES-GCM encryption. If either is modified
    in the database, decryption will fail. To "rename" a secret, create a new one with
    the desired name and delete the old one. See mlflow/utils/crypto.py:_create_aad().

    This dataclass is frozen (immutable) because:
    1. It represents a read-only view of database state
    2. secret_id and secret_name must never be modified (used in encryption AAD)
    3. Database triggers also enforce immutability of these fields

    Args:
        secret_id: Unique identifier for this secret. IMMUTABLE - used in AAD for encryption.
        secret_name: User-friendly name for the secret. IMMUTABLE - used in AAD for encryption.
        masked_values: Masked version of the secret values for display as key-value pairs.
            For simple API keys: ``{"api_key": "sk-...xyz123"}``.
            For compound credentials: ``{"aws_access_key_id": "AKI...1234", ...}``.
        created_at: Timestamp (milliseconds) when the secret was created.
        last_updated_at: Timestamp (milliseconds) when the secret was last updated.
        provider: LLM provider this secret is for (e.g., "openai", "anthropic").
        auth_config: Provider-specific configuration (e.g., region, project_id).
            This is non-sensitive metadata useful for UI disambiguation.
        created_by: User ID who created the secret.
        last_updated_by: User ID who last updated the secret.
    """

    secret_id: str
    secret_name: str
    masked_values: dict[str, str]
    created_at: int
    last_updated_at: int
    provider: str | None = None
    auth_config: dict[str, Any] | None = None
    created_by: str | None = None
    last_updated_by: str | None = None

    def to_proto(self):
        proto = ProtoGatewaySecretInfo()
        proto.secret_id = self.secret_id
        proto.secret_name = self.secret_name
        proto.masked_values.update(self.masked_values)
        proto.created_at = self.created_at
        proto.last_updated_at = self.last_updated_at
        if self.provider is not None:
            proto.provider = self.provider
        if self.auth_config is not None:
            proto.auth_config_json = json.dumps(self.auth_config)
        if self.created_by is not None:
            proto.created_by = self.created_by
        if self.last_updated_by is not None:
            proto.last_updated_by = self.last_updated_by
        return proto

    @classmethod
    def from_proto(cls, proto):
        auth_config = None
        if proto.auth_config_json:
            auth_config = json.loads(proto.auth_config_json)
        return cls(
            secret_id=proto.secret_id,
            secret_name=proto.secret_name,
            masked_values=dict(proto.masked_values),
            created_at=proto.created_at,
            last_updated_at=proto.last_updated_at,
            provider=proto.provider or None,
            auth_config=auth_config,
            created_by=proto.created_by or None,
            last_updated_by=proto.last_updated_by or None,
        )
