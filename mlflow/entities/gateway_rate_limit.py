"""
Entities for Gateway Rate Limiting.

These entities represent rate limit configurations for gateway endpoints,
including default limits and per-user overrides.
"""

from __future__ import annotations

from dataclasses import dataclass

from mlflow.entities._mlflow_object import _MlflowObject


@dataclass
class GatewayRateLimitConfig(_MlflowObject):
    """
    Represents a rate limit configuration for a gateway endpoint.

    Rate limits can be set at two levels:
    1. Default (endpoint-level): Applies to all users without specific overrides
    2. Per-user: Overrides the default for specific users

    Args:
        rate_limit_id: Unique identifier for this rate limit configuration.
        endpoint_id: ID of the gateway endpoint this limit applies to.
        queries_per_minute: Maximum queries allowed per minute.
        username: Username for per-user limits (None for default endpoint limit).
        created_at: Timestamp (milliseconds) when the config was created.
        updated_at: Timestamp (milliseconds) when the config was last updated.
        created_by: User who created the configuration.
        updated_by: User who last updated the configuration.
    """

    rate_limit_id: str
    endpoint_id: str
    queries_per_minute: int
    username: str | None = None
    created_at: int = 0
    updated_at: int = 0
    created_by: str | None = None
    updated_by: str | None = None

    @property
    def is_default(self) -> bool:
        """Returns True if this is a default endpoint limit (not per-user)."""
        return self.username is None


@dataclass
class GatewayRateLimitInput:
    """
    Input data for creating or updating a rate limit configuration.

    Args:
        endpoint_id: ID of the gateway endpoint.
        queries_per_minute: Maximum queries allowed per minute.
        username: Username for per-user limits (None for default endpoint limit).
    """

    endpoint_id: str
    queries_per_minute: int
    username: str | None = None
