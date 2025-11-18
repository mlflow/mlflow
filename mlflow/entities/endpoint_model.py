from __future__ import annotations

from dataclasses import dataclass

from mlflow.entities._mlflow_object import _MlflowObject


@dataclass
class EndpointModel(_MlflowObject):
    """
    MLflow entity representing a Model within an Endpoint.

    An Endpoint can contain multiple models for routing/failover scenarios.
    Each model has its own secret (API key) for authentication.

    This entity contains only metadata about the model - cryptographic fields
    (encrypted_model_config, wrapped_model_config_dek) are never exposed outside the store layer.

    Args:
        model_id: String containing model ID (UUID).
        endpoint_id: String containing the parent endpoint ID (UUID).
        model_name: String containing the model identifier.
            E.g., "claude-3-5-sonnet-20241022", "gpt-4-turbo", "gemini-2.5-pro".
        secret_id: String containing the secret ID (UUID) used for this model's authentication.
        routing_config: Optional JSON string containing flexible routing configuration.
            Examples: {"weight": 0.5}, {"priority": 1}, {"strategy": "round_robin"}
        created_at: Creation timestamp in milliseconds since the UNIX epoch.
        last_updated_at: Last update timestamp in milliseconds since the UNIX epoch.
        created_by: String containing the user ID who created the model, or None.
        last_updated_by: String containing the user ID who last updated the model, or None.
        secret_name: String containing the secret name (for UI display). Optional.
        provider: String containing the LLM provider (for UI display). Optional.
    """

    model_id: str
    endpoint_id: str
    model_name: str
    secret_id: str
    created_at: int
    last_updated_at: int
    routing_config: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    secret_name: str = ""
    provider: str = ""
