import logging
import threading

from cachetools import LRUCache
from cachetools.func import cached

from mlflow.environment_variables import MLFLOW_GATEWAY_ALLOWED_PROVIDERS

_logger = logging.getLogger(__name__)

# Single source of truth for provider name aliases (string-level).
_PROVIDER_ALIASES: dict[str, str] = {
    "amazon-bedrock": "bedrock",
    "databricks-model-serving": "databricks",
}

_provider_filter_cache = LRUCache(maxsize=16)
_provider_filter_cache_lock = threading.RLock()


def normalize_provider_name(name: str) -> str:
    return _PROVIDER_ALIASES.get(name, name)


def _parse_provider_list(value: str | None) -> frozenset[str]:
    if not value:
        return frozenset()
    return frozenset(
        normalize_provider_name(p.strip().lower()) for p in value.split(",") if p.strip()
    )


@cached(cache=_provider_filter_cache, lock=_provider_filter_cache_lock)
def _parse_allowed_providers(allowed_raw: str | None) -> frozenset[str] | None:
    return _parse_provider_list(allowed_raw) or None


def _get_allowed_providers() -> frozenset[str] | None:
    return _parse_allowed_providers(MLFLOW_GATEWAY_ALLOWED_PROVIDERS.get())


def is_provider_allowed(provider_name: str) -> bool:
    allowed = _get_allowed_providers()
    if allowed is None:
        return True
    name = normalize_provider_name(provider_name.lower())
    return name in allowed


def filter_providers(providers: list[str]) -> list[str]:
    allowed = _get_allowed_providers()
    if allowed is None:
        return providers

    result = []
    for p in providers:
        name = normalize_provider_name(p.lower())
        if name not in allowed:
            _logger.debug("Provider '%s' is not in MLFLOW_GATEWAY_ALLOWED_PROVIDERS", p)
            continue
        result.append(p)
    return result
