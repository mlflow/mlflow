"""Databricks-specific RateLimitRetryAdapter registration.

Registers a RateLimitRetryAdapter that suppresses the Databricks SDK's own
429-retry loop so that rate-limit errors propagate to MLflow's call_with_retry
/ AIMD logic. Only activates when the active tracking URI is a Databricks URI.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

from mlflow.genai.judges.adapters.litellm_adapter import (
    RateLimitRetryAdapter,
    register_retry_adapter,
)


def _is_databricks_tracking_uri() -> bool:
    try:
        from mlflow.tracking import get_tracking_uri
        from mlflow.utils.uri import is_databricks_uri

        return is_databricks_uri(get_tracking_uri())
    except Exception:
        return False


@contextlib.contextmanager
def _disable_databricks_429_retry() -> Iterator[None]:
    from mlflow.utils.rest_utils import disable_429_retry

    with disable_429_retry():
        yield


register_retry_adapter(
    RateLimitRetryAdapter(
        name="databricks-sdk",
        is_adapter_active=lambda: _is_databricks_tracking_uri(),
        disable_internal_retries=_disable_databricks_429_retry,
    )
)
