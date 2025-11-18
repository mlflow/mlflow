from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mlflow.entities._mlflow_object import _MlflowObject

if TYPE_CHECKING:
    from mlflow.entities.endpoint import Endpoint
    from mlflow.entities.secret_binding import SecretBinding
    from mlflow.entities.secret_tag import SecretTag


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
        binding: The initial SecretBinding that associates the endpoint with a resource.
    """

    secret: "Secret"
    endpoint: "Endpoint"
    binding: "SecretBinding"


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
        tags: List of SecretTag objects associated with this secret.
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
    tags: list[SecretTag] = field(default_factory=list)
