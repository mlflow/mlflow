from dataclasses import dataclass

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.service_pb2 import GatewaySecret as ProtoGatewaySecret


@dataclass
class GatewaySecret(_MlflowObject):
    """
    Represents an encrypted secret for authenticating with LLM providers.

    Secrets store encrypted API keys and authentication credentials using envelope encryption.
    The actual secret value is encrypted with a DEK (Data Encryption Key), which is itself
    encrypted by a KEK (Key Encryption Key) for secure storage.

    Args:
        secret_id: Unique identifier for this secret.
        secret_name: User-friendly name for the secret (must be unique).
        masked_value: Masked version of the secret for display (e.g., "sk-...xyz123").
        provider: LLM provider this secret is for (e.g., "openai", "anthropic").
        created_at: Timestamp (milliseconds) when the secret was created.
        last_updated_at: Timestamp (milliseconds) when the secret was last updated.
        created_by: User ID who created the secret.
        last_updated_by: User ID who last updated the secret.
    """

    secret_id: str
    secret_name: str
    masked_value: str
    created_at: int
    last_updated_at: int
    provider: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None

    def to_proto(self):
        proto = ProtoGatewaySecret()
        proto.secret_id = self.secret_id
        proto.secret_name = self.secret_name
        proto.masked_value = self.masked_value
        proto.created_at = self.created_at
        proto.last_updated_at = self.last_updated_at
        if self.provider is not None:
            proto.provider = self.provider
        if self.created_by is not None:
            proto.created_by = self.created_by
        if self.last_updated_by is not None:
            proto.last_updated_by = self.last_updated_by
        return proto

    @classmethod
    def from_proto(cls, proto):
        return cls(
            secret_id=proto.secret_id,
            secret_name=proto.secret_name,
            masked_value=proto.masked_value,
            created_at=proto.created_at,
            last_updated_at=proto.last_updated_at,
            provider=proto.provider or None,
            created_by=proto.created_by or None,
            last_updated_by=proto.last_updated_by or None,
        )
