from dataclasses import dataclass, field
from typing import Any


@dataclass
class GatewayModelConfig:
    """
    Model configuration with decrypted credentials for runtime use.

    This entity contains everything needed to make LLM API calls, including
    the decrypted secret value and optional auth configuration. This is only
    used server-side and should never be exposed to clients.

    Args:
        model_definition_id: Unique identifier for the model definition.
        provider: LLM provider (e.g., "openai", "anthropic", "cohere", "bedrock").
        model_name: Provider-specific model identifier (e.g., "gpt-4o").
        secret_value: Decrypted API key or authentication credential.
        credential_name: Credential identifier (e.g., "OPENAI_API_KEY").
        auth_config: Decrypted provider-specific auth configuration (e.g., project_id, region).
    """

    model_definition_id: str
    provider: str
    model_name: str
    secret_value: str
    credential_name: str | None = None
    auth_config: dict[str, Any] | None = None


@dataclass
class GatewayEndpointConfig:
    """
    Complete endpoint configuration for resource runtime use.

    This entity contains all information needed for a resource to make LLM API calls,
    including decrypted secrets. This is only used server-side and should never be
    exposed to clients.

    Args:
        endpoint_id: Unique identifier for the endpoint.
        endpoint_name: User-friendly name for the endpoint.
        models: List of model configurations with decrypted credentials.
    """

    endpoint_id: str
    endpoint_name: str
    models: list[GatewayModelConfig] = field(default_factory=list)
