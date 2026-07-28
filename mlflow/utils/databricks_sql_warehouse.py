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


# Max wall-clock the synchronous Statement Execution API waits inline before returning; the
# SDK accepts values in the range 5s-50s. Deletes over UC trace tables are small and fast, so
# the inline wait is sufficient and we do not implement async polling here.
_STATEMENT_WAIT_TIMEOUT = "50s"


def execute_sql_statement(warehouse_id: str, statement: str, parameters: list | None = None):
    """
    Run a SQL statement on the given SQL warehouse via the Databricks Statement Execution API.

    Ensures the warehouse is RUNNING first (subject to ``MLFLOW_SQL_WAREHOUSE_AUTO_START``), then
    submits the statement synchronously and waits inline for it to finish.

    Args:
        warehouse_id: The Databricks SQL warehouse ID to run the statement on.
        statement: The SQL statement text. Prefer parameter markers (``:name``) over string
            interpolation for any value that could contain user input.
        parameters: Optional list of ``StatementParameterListItem`` values bound to the markers
            in ``statement``.

    Returns:
        The SDK ``StatementResponse``. Callers that need to confirm a mutation (e.g. a DELETE)
        should inspect the returned rows rather than assume success.

    Raises:
        MlflowException: If the statement does not reach the SUCCEEDED state.
    """
    from databricks.sdk.service.sql import StatementState

    ensure_sql_warehouse_running(warehouse_id)

    client = _get_workspace_client()
    response = client.statement_execution.execute_statement(
        statement=statement,
        warehouse_id=warehouse_id,
        parameters=parameters,
        wait_timeout=_STATEMENT_WAIT_TIMEOUT,
    )
    status = response.status
    if status is None or status.state != StatementState.SUCCEEDED:
        detail = ""
        if status is not None and status.error is not None:
            detail = f": {status.error.error_code}: {status.error.message}"
        state = status.state if status is not None else "UNKNOWN"
        raise MlflowException(
            f"SQL statement did not succeed on warehouse '{warehouse_id}' (state={state}){detail}"
        )
    return response


def num_affected_rows(response) -> int | None:
    """
    Extract the number of rows affected by a DML statement from a ``StatementResponse``.

    Databricks returns the affected-row count for a ``DELETE``/``UPDATE`` as a single-cell result
    set. Returns ``None`` when the count cannot be determined, so callers can distinguish "no rows
    matched" (0) from "the backend did not report a count" (schema/contract drift).
    """
    result = getattr(response, "result", None)
    if result is None or not getattr(result, "data_array", None):
        return None
    try:
        return int(result.data_array[0][0])
    except (ValueError, TypeError, IndexError):
        return None
