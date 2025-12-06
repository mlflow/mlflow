from dataclasses import dataclass
from typing import Any

from mlflow.entities._mlflow_object import _MlflowObject


@dataclass
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

    Args:
        secret_id: Unique identifier for this secret. IMMUTABLE - used in AAD for encryption.
        secret_name: User-friendly name for the secret. IMMUTABLE - used in AAD for encryption.
        masked_value: Masked version of the secret for display (e.g., "sk-...xyz123").
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
    masked_value: str
    created_at: int
    last_updated_at: int
    provider: str | None = None
    auth_config: dict[str, Any] | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
