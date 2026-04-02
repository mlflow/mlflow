import logging

from mlflow.environment_variables import (
    MLFLOW_GATEWAY_ALLOWED_PROVIDERS,
    MLFLOW_GATEWAY_BLOCKED_PROVIDERS,
)
from mlflow.exceptions import MlflowException

_logger = logging.getLogger(__name__)

_cached_allowed: set[str] | None = None
_cached_blocked: set[str] | None = None
_cached_allowed_raw: str | None = object()  # sentinel distinct from None/str
_cached_blocked_raw: str | None = object()


def _parse_provider_list(value: str | None) -> set[str]:
    if not value:
        return set()
    return {p.strip().lower() for p in value.split(",") if p.strip()}


def _get_provider_filter() -> tuple[set[str] | None, set[str] | None]:
    global _cached_allowed, _cached_blocked, _cached_allowed_raw, _cached_blocked_raw

    allowed_raw = MLFLOW_GATEWAY_ALLOWED_PROVIDERS.get()
    blocked_raw = MLFLOW_GATEWAY_BLOCKED_PROVIDERS.get()

    if allowed_raw == _cached_allowed_raw and blocked_raw == _cached_blocked_raw:
        return _cached_allowed, _cached_blocked

    if allowed_raw and blocked_raw:
        raise MlflowException.invalid_parameter_value(
            "MLFLOW_GATEWAY_ALLOWED_PROVIDERS and MLFLOW_GATEWAY_BLOCKED_PROVIDERS "
            "cannot be set at the same time. Use one or the other."
        )

    _cached_allowed_raw = allowed_raw
    _cached_blocked_raw = blocked_raw
    _cached_allowed = _parse_provider_list(allowed_raw) or None
    _cached_blocked = _parse_provider_list(blocked_raw) or None
    return _cached_allowed, _cached_blocked


def is_provider_allowed(provider_name: str) -> bool:
    allowed, blocked = _get_provider_filter()
    name = provider_name.lower()

    if allowed is not None:
        return name in allowed
    if blocked is not None:
        return name not in blocked
    return True


def filter_providers(providers: list[str]) -> list[str]:
    allowed, blocked = _get_provider_filter()
    if allowed is None and blocked is None:
        return providers

    result = []
    for p in providers:
        name = p.lower()
        if allowed is not None and name not in allowed:
            _logger.info("Provider '%s' is not in MLFLOW_GATEWAY_ALLOWED_PROVIDERS", p)
            continue
        if blocked is not None and name in blocked:
            _logger.info("Provider '%s' is blocked by MLFLOW_GATEWAY_BLOCKED_PROVIDERS", p)
            continue
        result.append(p)
    return result
