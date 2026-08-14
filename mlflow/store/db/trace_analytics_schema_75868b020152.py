"""Frozen trace analytics schema contract for Alembic revision 75868b020152.

IMMUTABLE: do not edit after this module ships. A historical Alembic migration must replay
identically forever, so the offline migration (75868b020152_add_sql_trace_analytics_schema.py)
cannot depend on helpers that a later release is free to change. The online prepopulation utility
(mlflow.store.db.trace_analytics.ensure_analytics_columns) runs on a database sitting at exactly the
revision before that migration and must add byte-for-byte identical columns. Both import the column
definitions, lengths, and validators from this single frozen module, so there is one source of
truth instead of a live helper plus a frozen copy kept in sync by parity tests. A later schema
change needs a new revision and a new frozen module, never an edit to this one.
"""

from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# Column lengths frozen at revision 75868b020152. Kept as literals so a later change to the
# application's live length constants cannot alter this revision's stored schema.
MODEL_DIMENSION_MAX_LENGTH = 500
TRACE_NAME_MAX_LENGTH = 4096
SESSION_ID_MAX_LENGTH = 250


def analytics_columns_by_table() -> dict[str, list[sa.Column]]:
    """Return fresh definitions for the columns promoted by revision 75868b020152."""
    return {
        "trace_info": [
            sa.Column("trace_name", sa.String(length=TRACE_NAME_MAX_LENGTH), nullable=True),
            sa.Column("session_id", sa.String(length=SESSION_ID_MAX_LENGTH), nullable=True),
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
        # rerun that revalidates the is_numeric_value column it already added).
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
    # Reject an already-present analytics column whose type, nullability, or default would let the
    # backfill write values the schema then corrupts. A partial online run or a hand-altered column
    # can leave a same-named column in the wrong shape, so both the migration and the prepopulation
    # utility validate it here before backfilling or dropping the legacy analytics tables.
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
            f"Cannot add trace analytics columns: existing column {table_name}.{expected.name} "
            f"has incompatible schema ({'; '.join(problems)})"
        )
