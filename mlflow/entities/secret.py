from dataclasses import dataclass
from enum import Enum

from mlflow.entities._mlflow_object import _MlflowObject


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
class Secret(_MlflowObject):
    """
    MLflow entity representing a Secret.

    Secrets store encrypted credentials (e.g., API keys) using envelope encryption.
    This entity contains only metadata about the secret - cryptographic fields
    (ciphertext, iv, wrapped_dek, etc.) are never exposed outside the store layer.

    Args:
        secret_id: String containing secret ID (UUID).
        secret_name: String containing the unique secret name.
        is_shared: Boolean indicating if secret can be reused across resources.
            True: Shared secret (can be bound to multiple resources)
            False: Private secret (bound to single resource)
        state: The secret state as a SecretState enum.
            Possible values: SecretState.ACTIVE, SecretState.REVOKED, SecretState.ROTATED
        created_at: Creation timestamp in milliseconds since the UNIX epoch.
        last_updated_at: Last update timestamp in milliseconds since the UNIX epoch.
        created_by: String containing the user ID who created the secret, or None.
        last_updated_by: String containing the user ID who last updated the secret, or None.
    """

    secret_id: str
    secret_name: str
    is_shared: bool
    state: SecretState
    created_at: int
    last_updated_at: int
    created_by: str | None = None
    last_updated_by: str | None = None
