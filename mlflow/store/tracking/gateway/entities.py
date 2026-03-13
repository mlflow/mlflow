from dataclasses import dataclass, field
from typing import Any

from mlflow.entities.gateway_endpoint import (
    FallbackConfig,
    FallbackStrategy,
    GatewayModelLinkageType,
    RoutingStrategy,
)


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
        weight: Routing weight for traffic distribution (default 1.0).
        linkage_type: Type of linkage (PRIMARY or FALLBACK).
        fallback_order: Order for fallback attempts (only for FALLBACK linkages, None for PRIMARY).
    """

    model_definition_id: str
    provider: str
    model_name: str
    secret_value: dict[str, Any]
    auth_config: dict[str, Any] | None = None
    weight: float = 1.0
    linkage_type: GatewayModelLinkageType = GatewayModelLinkageType.PRIMARY
    fallback_order: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_definition_id": self.model_definition_id,
            "provider": self.provider,
            "model_name": self.model_name,
            "secret_value": self.secret_value,
            "auth_config": self.auth_config,
            "weight": self.weight,
            "linkage_type": self.linkage_type.value,
            "fallback_order": self.fallback_order,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GatewayModelConfig":
        return cls(
            model_definition_id=data["model_definition_id"],
            provider=data["provider"],
            model_name=data["model_name"],
            secret_value=data["secret_value"],
            auth_config=data["auth_config"],
            weight=data["weight"],
            linkage_type=GatewayModelLinkageType(data["linkage_type"]),
            fallback_order=data["fallback_order"],
        )


@dataclass
class GatewayEndpointConfig:
    """
    Complete endpoint configuration for resource runtime use.

    This entity contains all information needed for a resource to make LLM API calls,
    including decrypted secrets and routing configuration. This is only used server-side
    and should never be exposed to clients.

    Args:
        endpoint_id: Unique identifier for the endpoint.
        endpoint_name: User-friendly name for the endpoint.
        models: List of model configurations with decrypted credentials.
        routing_strategy: Optional routing strategy (e.g., FALLBACK).
        fallback_config: Optional fallback configuration from GatewayEndpoint entity.
        experiment_id: Optional experiment ID for tracing (if usage tracking is enabled).
    """

    endpoint_id: str
    endpoint_name: str
    models: list[GatewayModelConfig] = field(default_factory=list)
    routing_strategy: RoutingStrategy | None = None
    fallback_config: FallbackConfig | None = None
    experiment_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        fallback = None
        if self.fallback_config is not None:
            fallback = {
                "strategy": self.fallback_config.strategy.value
                if self.fallback_config.strategy
                else None,
                "max_attempts": self.fallback_config.max_attempts,
            }
        return {
            "endpoint_id": self.endpoint_id,
            "endpoint_name": self.endpoint_name,
            "models": [m.to_dict() for m in self.models],
            "routing_strategy": self.routing_strategy.value if self.routing_strategy else None,
            "fallback_config": fallback,
            "experiment_id": self.experiment_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GatewayEndpointConfig":
        fallback = None
        if data.get("fallback_config") is not None:
            fc = data["fallback_config"]
            fallback = FallbackConfig(
                strategy=FallbackStrategy(fc["strategy"]) if fc.get("strategy") else None,
                max_attempts=fc.get("max_attempts"),
            )
        return cls(
            endpoint_id=data["endpoint_id"],
            endpoint_name=data["endpoint_name"],
            models=[GatewayModelConfig.from_dict(m) for m in data.get("models", [])],
            routing_strategy=RoutingStrategy(data["routing_strategy"])
            if data.get("routing_strategy")
            else None,
            fallback_config=fallback,
            experiment_id=data.get("experiment_id"),
        )
