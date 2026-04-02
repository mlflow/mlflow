import logging
import threading

from cachetools import LRUCache
from cachetools.func import cached

from mlflow.environment_variables import (
    MLFLOW_GATEWAY_ALLOWED_PROVIDERS,
    MLFLOW_GATEWAY_BLOCKED_PROVIDERS,
)
from mlflow.exceptions import MlflowException
from mlflow.gateway.config import PROVIDER_ALIASES

_logger = logging.getLogger(__name__)

_provider_filter_cache = LRUCache(maxsize=16)
_provider_filter_cache_lock = threading.RLock()


def _normalize_provider_name(name: str) -> str:
    return PROVIDER_ALIASES.get(name, name)


def _parse_provider_list(value: str | None) -> frozenset[str]:
    if not value:
        return frozenset()
    return frozenset(
        _normalize_provider_name(p.strip().lower()) for p in value.split(",") if p.strip()
    )


@cached(cache=_provider_filter_cache, lock=_provider_filter_cache_lock)
def _parse_provider_filter(
    allowed_raw: str | None, blocked_raw: str | None
) -> tuple[frozenset[str] | None, frozenset[str] | None]:
    if allowed_raw and blocked_raw:
        raise MlflowException.invalid_parameter_value(
            "MLFLOW_GATEWAY_ALLOWED_PROVIDERS and MLFLOW_GATEWAY_BLOCKED_PROVIDERS "
            "cannot be set at the same time. Use one or the other."
        )

    allowed = _parse_provider_list(allowed_raw) or None
    blocked = _parse_provider_list(blocked_raw) or None
    return allowed, blocked


def _get_provider_filter() -> tuple[frozenset[str] | None, frozenset[str] | None]:
    return _parse_provider_filter(
        MLFLOW_GATEWAY_ALLOWED_PROVIDERS.get(),
        MLFLOW_GATEWAY_BLOCKED_PROVIDERS.get(),
    )


def is_provider_allowed(provider_name: str) -> bool:
    allowed, blocked = _get_provider_filter()
    name = _normalize_provider_name(provider_name.lower())

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
        name = _normalize_provider_name(p.lower())
        if allowed is not None and name not in allowed:
            _logger.debug("Provider '%s' is not in MLFLOW_GATEWAY_ALLOWED_PROVIDERS", p)
            continue
        if blocked is not None and name in blocked:
            _logger.debug("Provider '%s' is blocked by MLFLOW_GATEWAY_BLOCKED_PROVIDERS", p)
            continue
        result.append(p)
    return result
