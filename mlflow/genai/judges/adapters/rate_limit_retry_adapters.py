"""Central registration of all RateLimitRetryAdapters.

Each provider that has its own internal 429-retry mechanism registers here.
eval_retry_context() in rate_limiter.py iterates this registry at call time
and disables only the adapters whose is_adapter_active() returns True.

To add a new adapter, import register_retry_adapter and RateLimitRetryAdapter
from litellm_adapter and append a new entry below.
"""

from __future__ import annotations

from mlflow.genai.judges.adapters.litellm_adapter import (
    RateLimitRetryAdapter,
    _is_litellm_available,
    disable_litellm_rate_limit_retries,
    register_retry_adapter,
)


def _is_databricks_tracking_uri() -> bool:
    try:
        from mlflow.tracking import get_tracking_uri
        from mlflow.utils.uri import is_databricks_uri

        return is_databricks_uri(get_tracking_uri())
    except Exception:
        return False


def _disable_databricks_429_retry():
    from mlflow.utils.rest_utils import disable_429_retry

    return disable_429_retry()


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
        disable_internal_retries=_disable_databricks_429_retry,
    )
)
