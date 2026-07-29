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


def ensure_sql_warehouse_running(warehouse_id: str) -> None:
    """
    Verify the SQL warehouse is in ``RUNNING`` state, starting it and waiting if necessary.

    No-op when ``MLFLOW_SQL_WAREHOUSE_AUTO_START`` is false. Results are cached per-process
    for ``_CACHE_TTL_SECONDS`` to avoid hammering the SDK across closely-spaced calls.
    The ``start_and_wait`` timeout is taken from
    ``MLFLOW_SQL_WAREHOUSE_AUTO_START_TIMEOUT_SECONDS``.

    Args:
        warehouse_id: The Databricks SQL warehouse ID to check.

    Raises:
        MlflowException: When the warehouse fails to reach ``RUNNING`` (timeout or other
            SDK error).
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
        timeout = MLFLOW_SQL_WAREHOUSE_AUTO_START_TIMEOUT_SECONDS.get()
        _logger.info(
            f"SQL warehouse '{warehouse_id}' is {info.state.value}; starting it and "
            f"waiting up to {timeout}s for RUNNING."
        )
        try:
            client.warehouses.start_and_wait(warehouse_id, timeout=timedelta(seconds=timeout))
        except TimeoutError as e:
            raise MlflowException(
                f"Timed out after {timeout}s waiting for SQL warehouse '{warehouse_id}' to "
                f"reach RUNNING state. Increase the timeout via the "
                f"`{MLFLOW_SQL_WAREHOUSE_AUTO_START_TIMEOUT_SECONDS.name}` environment "
                f"variable, or start the warehouse explicitly and retry."
            ) from e
        except Exception as e:
            raise MlflowException(
                f"Failed to start SQL warehouse '{warehouse_id}': {e}. Start the warehouse "
                f"explicitly and retry, or set `MLFLOW_SQL_WAREHOUSE_AUTO_START=false` to "
                f"disable this preflight."
            ) from e

    _verified_running[warehouse_id] = time.monotonic() + _CACHE_TTL_SECONDS
