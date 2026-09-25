from contextlib import contextmanager
from math import ceil

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from mlflow.store.db.trace_analytics_schema_75868b020152 import (
    _validate_existing_column,
    analytics_columns_by_table,
)

DEFAULT_DDL_LOCK_TIMEOUT_SECONDS = 5


@contextmanager
def _ddl_lock_timeout(connection: sa.Connection, timeout_seconds: float | None):
    """Bound DDL lock acquisition without changing the final migration's wait behavior."""
    if timeout_seconds is None:
        yield
        return
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    dialect_name = connection.dialect.name
    timeout_ms = max(1, ceil(timeout_seconds * 1000))

    if dialect_name == "postgresql":
        # SET LOCAL is transaction-scoped, so the timeout cannot leak into a pooled connection.
        connection.exec_driver_sql(f"SET LOCAL lock_timeout = '{timeout_ms}ms'")
        yield
        return

    if dialect_name == "mysql":
        previous_timeout = connection.exec_driver_sql(
            "SELECT @@SESSION.lock_wait_timeout"
        ).scalar_one()
        connection.exec_driver_sql(
            f"SET SESSION lock_wait_timeout = {max(1, ceil(timeout_seconds))}"
        )
        try:
            yield
        finally:
            connection.exec_driver_sql(f"SET SESSION lock_wait_timeout = {int(previous_timeout)}")
        return

    if dialect_name == "mssql":
        previous_timeout = connection.exec_driver_sql("SELECT @@LOCK_TIMEOUT").scalar_one()
        connection.exec_driver_sql(f"SET LOCK_TIMEOUT {timeout_ms}")
        try:
            yield
        finally:
            connection.exec_driver_sql(f"SET LOCK_TIMEOUT {int(previous_timeout)}")
        return

    # SQLite's connection timeout already bounds acquisition of its database-wide write lock.
    yield


def ensure_analytics_columns(
    connection: sa.Connection,
    *,
    lock_timeout_seconds: float | None = None,
) -> dict[str, list[str]]:
    """Add missing trace analytics columns and validate columns left by a partial run."""
    operations = Operations(MigrationContext.configure(connection))
    added: dict[str, list[str]] = {}

    with _ddl_lock_timeout(connection, lock_timeout_seconds):
        for table_name, expected_columns in analytics_columns_by_table().items():
            inspector = sa.inspect(connection)
            if not inspector.has_table(table_name):
                raise RuntimeError(
                    f"Cannot prepopulate trace analytics: required table {table_name!r} is missing"
                )
            for expected in expected_columns:
                # Reinspect after every DDL operation. Some supported databases implicitly commit
                # DDL, so a later run must discover exactly how far a partial expansion got.
                actual_columns = {
                    column["name"]: column
                    for column in sa.inspect(connection).get_columns(table_name)
                }
                if actual := actual_columns.get(expected.name):
                    _validate_existing_column(table_name, expected, actual, connection.dialect)
                    continue
                operations.add_column(table_name, expected)
                added.setdefault(table_name, []).append(expected.name)

    return added
