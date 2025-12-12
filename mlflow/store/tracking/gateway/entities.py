from dataclasses import dataclass, field
from typing import Any


@dataclass
class GatewayModelConfig:
    """
    Model configuration with decrypted credentials for runtime use.

    This entity contains everything needed to make LLM API calls, including
    the decrypted secrets and auth configuration. This is only used
    server-side and should never be exposed to clients.

    Args:
        model_definition_id: Unique identifier for the model definition.
        provider: LLM provider (e.g., "openai", "anthropic", "cohere", "bedrock").
        model_name: Provider-specific model identifier (e.g., "gpt-4o").
        secret_value: Decrypted secrets as a dict. For providers with multiple
            auth modes, contains all secret fields (e.g., {"aws_access_key_id": "...",
            "aws_secret_access_key": "..."}). For simple providers, contains
            {"api_key": "..."}.
        auth_config: Non-secret configuration including auth_mode (e.g.,
            {"auth_mode": "access_keys", "aws_region_name": "us-east-1"}).
    """

    model_definition_id: str
    provider: str
    model_name: str
    secret_value: dict[str, Any]
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
