from contextlib import contextmanager
from math import ceil
from typing import Any

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy.dialects import mysql

from mlflow.store.tracking.utils.trace_analytics import MODEL_DIMENSION_MAX_LENGTH
from mlflow.tracing.constant import (
    MAX_CHARS_IN_TRACE_INFO_METADATA,
    MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE,
)

DEFAULT_DDL_LOCK_TIMEOUT_SECONDS = 5


def analytics_columns_by_table() -> dict[str, list[sa.Column]]:
    """Return fresh definitions for columns promoted by the trace analytics migration."""
    return {
        "trace_info": [
            sa.Column(
                "trace_name",
                sa.String(length=MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE),
                nullable=True,
            ),
            sa.Column(
                "session_id",
                sa.String(length=MAX_CHARS_IN_TRACE_INFO_METADATA),
                nullable=True,
            ),
            sa.Column("input_tokens", sa.BigInteger(), nullable=True),
            sa.Column("output_tokens", sa.BigInteger(), nullable=True),
            sa.Column("total_tokens", sa.BigInteger(), nullable=True),
            sa.Column("cache_read_input_tokens", sa.BigInteger(), nullable=True),
            sa.Column("cache_creation_input_tokens", sa.BigInteger(), nullable=True),
            sa.Column("input_cost", sa.Float(precision=53), nullable=True),
            sa.Column("output_cost", sa.Float(precision=53), nullable=True),
            sa.Column("total_cost", sa.Float(precision=53), nullable=True),
        ],
        "assessments": [
            sa.Column("experiment_id", sa.Integer(), nullable=True),
            sa.Column("trace_timestamp_ms", sa.BigInteger(), nullable=True),
            sa.Column("aggregate_value", sa.Float(precision=53), nullable=True),
            # Added nullable so the online prepopulation utility expands the schema with a fast
            # metadata-only ALTER. The offline migration tightens it to NOT NULL to match the ORM
            # model; the false server default keeps a concurrent insert from leaving a NULL.
            sa.Column(
                "is_numeric_value",
                sa.Boolean(),
                nullable=True,
                server_default=sa.false(),
            ),
        ],
        "spans": [
            sa.Column("input_cost", sa.Float(precision=53), nullable=True),
            sa.Column("output_cost", sa.Float(precision=53), nullable=True),
            sa.Column("total_cost", sa.Float(precision=53), nullable=True),
            sa.Column("model_name", sa.String(length=MODEL_DIMENSION_MAX_LENGTH), nullable=True),
            sa.Column(
                "model_provider", sa.String(length=MODEL_DIMENSION_MAX_LENGTH), nullable=True
            ),
        ],
    }


def _type_description(type_: sa.types.TypeEngine, dialect: sa.engine.Dialect) -> str:
    return str(type_.dialect_impl(dialect).compile(dialect=dialect))


def types_are_compatible(
    expected: sa.types.TypeEngine,
    actual: sa.types.TypeEngine,
    dialect: sa.engine.Dialect,
) -> bool:
    # Dispatch on the generic expected type rather than dialect_impl(). Some drivers rewrite a
    # generic type into a subclass that breaks the isinstance() family checks below: e.g. psycopg
    # turns sa.Float into _PsycopgFloat, which derives from Numeric (not sa.Float), so a FLOAT(53)
    # column would be wrongly rejected as incompatible with itself.
    if isinstance(expected, sa.Text):
        return isinstance(actual, sa.Text)
    if isinstance(expected, sa.BigInteger):
        return isinstance(actual, sa.BigInteger)
    if isinstance(expected, sa.Integer):
        return isinstance(actual, sa.Integer) and not isinstance(
            actual, (sa.BigInteger, sa.SmallInteger)
        )
    if isinstance(expected, sa.Float):
        return isinstance(actual, sa.Float)
    if isinstance(expected, sa.Boolean):
        if isinstance(actual, sa.Boolean):
            return True
        # MySQL has no native BOOLEAN type: it stores and reflects Boolean columns as TINYINT(1),
        # so a reflected TINYINT(1) is the expected match on that dialect (e.g. on a prepopulation
        # rerun that revalidates the column it already added).
        return isinstance(actual, mysql.TINYINT) and getattr(actual, "display_width", None) == 1
    if isinstance(expected, sa.String):
        # MySQL maps long String columns to TEXT via with_variant(), so a reflected Text column is
        # an acceptable match for an expected String on that dialect.
        if isinstance(actual, sa.Text):
            return True
        return isinstance(actual, sa.String) and expected.length == actual.length
    return type(expected) is type(actual)


def _normalized_false_default(default: Any) -> bool:
    if default is None:
        return False
    normalized = str(default).strip().lower().split("::", maxsplit=1)[0]
    normalized = normalized.strip("()'\"")
    return normalized in {"0", "false"}


def _validate_existing_column(
    table_name: str,
    expected: sa.Column,
    actual: dict[str, Any],
    dialect: sa.engine.Dialect,
) -> None:
    problems = []
    if not types_are_compatible(expected.type, actual["type"], dialect):
        problems.append(
            "type "
            f"{_type_description(actual['type'], dialect)}; expected "
            f"{_type_description(expected.type, dialect)}"
        )
    if expected.nullable != actual["nullable"]:
        problems.append(f"nullable={actual['nullable']}; expected nullable={expected.nullable}")
    if expected.server_default is not None and not _normalized_false_default(actual.get("default")):
        problems.append(
            f"server default {actual.get('default')!r}; expected a false server default"
        )

    if problems:
        raise RuntimeError(
            f"Cannot prepopulate trace analytics: existing column {table_name}.{expected.name} "
            f"has incompatible schema ({'; '.join(problems)})"
        )


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
