"""Central registry of all RateLimitRetryAdapters.

Each provider that has its own internal 429-retry mechanism registers a
RateLimitRetryAdapter here. eval_retry_context() in rate_limiter.py iterates
this registry at call time and disables only the adapters whose
is_adapter_active() returns True, allowing 429s to propagate up to MLflow's
own call_with_retry / AIMD logic.

To register a new adapter, define its provider-specific ``is_adapter_active``
and ``disable_internal_retries`` helpers in that provider's adapter module,
then add a ``register_retry_adapter(...)`` call at the bottom of this file so
all adapters remain visible in a single place.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable

from mlflow.genai.judges.adapters.databricks_adapter import _is_databricks_tracking_uri
from mlflow.genai.judges.adapters.litellm_adapter import (
    _is_litellm_available,
    disable_litellm_rate_limit_retries,
)
from mlflow.utils.rest_utils import disable_429_retry


@dataclass
class RateLimitRetryAdapter:
    """Descriptor for a provider that has its own internal 429-retry mechanism.

    Attributes:
        name: Human-readable identifier (used in debug logging).
        is_adapter_active: Zero-argument callable returning True when this adapter is
            active in the current environment (e.g. package installed and the
            active tracking URI targets that provider).
        disable_internal_retries: Context manager that suppresses the
            provider's own retry loop for the duration of the ``with`` block.
    """

    name: str
    is_adapter_active: Callable[[], bool]
    disable_internal_retries: Callable[[], contextlib.AbstractContextManager]


_RETRY_ADAPTER_REGISTRY: list[RateLimitRetryAdapter] = []


def register_retry_adapter(adapter: RateLimitRetryAdapter) -> None:
    """Register a RateLimitRetryAdapter so eval_retry_context() can disable it.

    Deduplicates by name — registering an adapter with the same name as an
    existing entry replaces the existing entry.
    """
    for i, existing in enumerate(_RETRY_ADAPTER_REGISTRY):
        if existing.name == adapter.name:
            _RETRY_ADAPTER_REGISTRY[i] = adapter
            return
    _RETRY_ADAPTER_REGISTRY.append(adapter)


def get_retry_adapters() -> list[RateLimitRetryAdapter]:
    """Return the registered RateLimitRetryAdapters (read-only view for callers)."""
    return list(_RETRY_ADAPTER_REGISTRY)


# ── Adapter registrations ──
# Provider-specific helpers live in each provider's adapter module; the
# registration calls are centralized here so all adapters are visible at a glance.

# --- LiteLLM ---
register_retry_adapter(
    RateLimitRetryAdapter(
        name="litellm",
        is_adapter_active=_is_litellm_available,
        disable_internal_retries=disable_litellm_rate_limit_retries,
    )
)

# --- Databricks SDK ---
register_retry_adapter(
    RateLimitRetryAdapter(
        name="databricks-sdk",
        is_adapter_active=_is_databricks_tracking_uri,
        disable_internal_retries=disable_429_retry,
    )
)
