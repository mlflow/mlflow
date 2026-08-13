"""Databricks-specific rate-limit retry helpers.

The actual adapter registration lives in rate_limit_retry_adapters.py so that
all adapters are visible in a single place.
"""

from __future__ import annotations

from mlflow.utils.rest_utils import disable_429_retry  # noqa: F401 — re-exported for tests


def _is_databricks_tracking_uri() -> bool:
    try:
        from mlflow.tracking import get_tracking_uri
        from mlflow.utils.uri import is_databricks_uri

        return is_databricks_uri(get_tracking_uri())
    except Exception:
        return False
