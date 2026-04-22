"""
Helpers for making sure a Databricks SQL warehouse is running before MLflow tracing API calls
that require it.
"""

import logging
import time
from datetime import timedelta

from mlflow.environment_variables import (
    MLFLOW_SQL_WAREHOUSE_AUTO_START,
    MLFLOW_SQL_WAREHOUSE_AUTO_START_TIMEOUT_SECONDS,
)
from mlflow.exceptions import MlflowException

_logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 60.0

# warehouse_id -> monotonic deadline by which the "RUNNING" verification expires.
#
# Concurrent callers may race and hit the SDK more than once on a cold cache; that's fine.
# `warehouses.get` is cheap and `start_and_wait` is idempotent on the server. The cache's
# purpose is to eliminate SDK hops in the steady state, not to single-flight the cold path.
_verified_running: dict[str, float] = {}


def _get_workspace_client():
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient()


def ensure_sql_warehouse_running(warehouse_id: str, *, timeout_seconds: int | None = None) -> None:
    """
    Verify the SQL warehouse is in ``RUNNING`` state, starting it and waiting if necessary.

    No-op when ``MLFLOW_SQL_WAREHOUSE_AUTO_START`` is false. Results are cached per-process
    for ``_CACHE_TTL_SECONDS`` to avoid hammering the SDK across closely-spaced calls.

    Args:
        warehouse_id: The Databricks SQL warehouse ID to check.
        timeout_seconds: Override the timeout (seconds) for ``start_and_wait``. When ``None``,
            the value of ``MLFLOW_SQL_WAREHOUSE_AUTO_START_TIMEOUT_SECONDS`` is used.

    Raises:
        MlflowException: When ``start_and_wait`` times out before the warehouse reaches
            ``RUNNING`` state.
    """
    if not MLFLOW_SQL_WAREHOUSE_AUTO_START.get():
        return

    deadline = _verified_running.get(warehouse_id)
    if deadline is not None and time.monotonic() < deadline:
        return

    from databricks.sdk.service.sql import State

    client = _get_workspace_client()
    info = client.warehouses.get(warehouse_id)

    if info.state != State.RUNNING:
        resolved_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else MLFLOW_SQL_WAREHOUSE_AUTO_START_TIMEOUT_SECONDS.get()
        )
        _logger.info(
            "SQL warehouse '%s' is %s; starting it and waiting up to %ds for RUNNING.",
            warehouse_id,
            info.state.value,
            resolved_timeout,
        )
        try:
            client.warehouses.start_and_wait(
                warehouse_id, timeout=timedelta(seconds=resolved_timeout)
            )
        except TimeoutError as e:
            raise MlflowException(
                f"Timed out after {resolved_timeout}s waiting for SQL warehouse "
                f"'{warehouse_id}' to reach RUNNING state. Increase the timeout via "
                f"the `{MLFLOW_SQL_WAREHOUSE_AUTO_START_TIMEOUT_SECONDS.name}` "
                f"environment variable."
            ) from e

    _verified_running[warehouse_id] = time.monotonic() + _CACHE_TTL_SECONDS
