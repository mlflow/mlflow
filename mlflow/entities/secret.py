from dataclasses import dataclass

from mlflow.entities._mlflow_object import _MlflowObject


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
    """

    secret_id: str
    secret_name: str
    masked_value: str
    is_shared: bool
    created_at: int
    last_updated_at: int
    created_by: str | None = None
    last_updated_by: str | None = None
