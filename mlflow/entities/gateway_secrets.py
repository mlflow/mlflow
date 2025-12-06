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

    Args:
        secret_id: Unique identifier for this secret.
        secret_name: User-friendly name for the secret (must be unique).
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
