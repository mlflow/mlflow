from dataclasses import dataclass

from mlflow.entities._mlflow_object import _MlflowObject


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
