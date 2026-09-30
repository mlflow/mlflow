"""Derived capability helpers for Gateway endpoints."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from mlflow.exceptions import MlflowException

SYSTEM_ONE_ACTION = "system_one"


class _GatewayModel(Protocol):
    provider: str
    model_name: str


def model_supports_system_one(provider: str, model_name: str) -> bool:
    """Return whether a provider/model mapping can serve System One."""
    return provider == "typesafe"


def endpoint_system_one_state(models: Iterable[_GatewayModel]) -> tuple[bool, bool]:
    """Return ``(any_supported, all_supported)`` for an endpoint's models."""
    supported = [model_supports_system_one(model.provider, model.model_name) for model in models]
    return any(supported), bool(supported) and all(supported)


def endpoint_supported_actions(models: Iterable[_GatewayModel]) -> list[str]:
    """Return actions that every model mapping on the endpoint can serve."""
    _, all_supported = endpoint_system_one_state(models)
    return [SYSTEM_ONE_ACTION] if all_supported else []


def validate_system_one_endpoint(models: Iterable[_GatewayModel]) -> None:
    """Reject endpoints that mix System One and non-System One models."""
    any_supported, all_supported = endpoint_system_one_state(models)
    if any_supported and not all_supported:
        raise MlflowException.invalid_parameter_value(
            "Gateway endpoints cannot mix System One and chat models. Every primary, fallback, "
            "and traffic-split model must support System One, or none of them may."
        )
