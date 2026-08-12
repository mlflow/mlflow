import json
from dataclasses import dataclass
from typing import Any, Callable

import sqlalchemy as sa
from alembic.migration import MigrationContext

from mlflow.store.db.trace_analytics import (
    DEFAULT_DDL_LOCK_TIMEOUT_SECONDS,
    ensure_analytics_columns,
)
from mlflow.store.tracking.utils.trace_analytics import (
    COST_COLUMN_BY_KEY,
    TOKEN_COLUMN_BY_KEY,
    _json_object,
    assessment_aggregate,
    bounded_model_dimension,
    finite_float_or_none,
    token_count_or_none,
)
from mlflow.tracing.constant import (
    MAX_CHARS_IN_TRACE_INFO_METADATA,
    MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE,
    SpanAttributeKey,
    TraceMetadataKey,
    TraceTagKey,
)

PREPOPULATION_SCHEMA_REVISION = "a8b9c0d1e2f3"
DEFAULT_BATCH_SIZE = 250
DEFAULT_PROGRESS_EVERY_BATCHES = 100

_TRACE_VALUE_COLUMNS = (
    "trace_name",
    "session_id",
    *TOKEN_COLUMN_BY_KEY.values(),
    *COST_COLUMN_BY_KEY.values(),
)
_SPAN_VALUE_COLUMNS = (
    *COST_COLUMN_BY_KEY.values(),
    "model_name",
    "model_provider",
)
_ASSESSMENT_VALUE_COLUMNS = (
    "experiment_id",
    "trace_timestamp_ms",
    "aggregate_value",
    "is_numeric_value",
)
_SOURCE_COLUMNS_BY_TABLE = {
    "trace_info": {"request_id", "experiment_id", "timestamp_ms"},
    "trace_tags": {"request_id", "key", "value"},
    "trace_request_metadata": {"request_id", "key", "value"},
    "trace_metrics": {"request_id", "key", "value"},
    "spans": {"trace_id", "span_id", "dimension_attributes"},
    "span_metrics": {"trace_id", "span_id", "key", "value"},
    "assessments": {"assessment_id", "trace_id", "value"},
}


@dataclass(frozen=True)
class EntityPrepopulationStats:
    scanned: int = 0
    updated: int = 0


@dataclass(frozen=True)
class TraceAnalyticsPrepopulationStats:
    traces: EntityPrepopulationStats
    spans: EntityPrepopulationStats
    assessments: EntityPrepopulationStats


@dataclass(frozen=True)
class _BatchResult:
    cursor: Any
    scanned: int
    updated: int


ProgressCallback = Callable[[str, EntityPrepopulationStats], None]


def _validate_schema_revision(connection: sa.Connection) -> None:
    revisions = MigrationContext.configure(connection).get_current_heads()
    if revisions != (PREPOPULATION_SCHEMA_REVISION,):
        found = ", ".join(revisions) if revisions else "no Alembic revision"
        raise RuntimeError(
            "Trace analytics prepopulation requires database revision "
            f"{PREPOPULATION_SCHEMA_REVISION!r}; found {found}. The command must run before "
            "the trace analytics migration."
        )


def _validate_source_schema(connection: sa.Connection) -> None:
    inspector = sa.inspect(connection)
    for table_name, required_columns in _SOURCE_COLUMNS_BY_TABLE.items():
        if not inspector.has_table(table_name):
            raise RuntimeError(
                f"Cannot prepopulate trace analytics: required table {table_name!r} is missing"
            )
        actual_columns = {column["name"] for column in inspector.get_columns(table_name)}
        if missing_columns := required_columns - actual_columns:
            raise RuntimeError(
                f"Cannot prepopulate trace analytics: table {table_name!r} is missing required "
                f"columns {sorted(missing_columns)}"
            )


def _reflect_tables(connection: sa.Connection) -> dict[str, sa.Table]:
    metadata = sa.MetaData()
    return {
        name: sa.Table(name, metadata, autoload_with=connection)
        for name in (
            "trace_info",
            "trace_tags",
            "trace_request_metadata",
            "trace_metrics",
            "spans",
            "span_metrics",
            "assessments",
        )
    }


def _validate_required_trace_joins(connection: sa.Connection, tables: dict[str, sa.Table]) -> None:
    trace_info = tables["trace_info"]
    for child_table, trace_column in (("spans", "trace_id"), ("assessments", "trace_id")):
        child = tables[child_table]
        missing_trace_id = connection.execute(
            sa
            .select(child.c[trace_column])
            .outerjoin(trace_info, child.c[trace_column] == trace_info.c.request_id)
            .where(trace_info.c.request_id.is_(None))
            .limit(1)
        ).scalar_one_or_none()
        if missing_trace_id is not None:
            raise RuntimeError(
                f"Cannot prepopulate trace analytics: {child_table} row references missing "
                f"trace_info row {missing_trace_id!r}"
            )


def _values_match(row: sa.Row, expected: dict[str, Any]) -> bool:
    return all(row._mapping[column] == value for column, value in expected.items())


def _bounded_string_or_none(value: Any, max_length: int) -> str | None:
    return value[:max_length] if isinstance(value, str) else None


def analytics_columns_from_metadata(
    metadata: dict[str, str],
) -> dict[str, float | int | str | None]:
    # Mirror the frozen migration's backfill semantics: coerce tokens to integers and truncate an
    # over-length session ID rather than raising. (The authoritative validate_session_id raises,
    # which is right for the live write path but wrong when backfilling pre-existing rows.)
    columns: dict[str, float | int | str | None] = {}
    if TraceMetadataKey.TRACE_SESSION in metadata:
        columns["session_id"] = _bounded_string_or_none(
            metadata[TraceMetadataKey.TRACE_SESSION], MAX_CHARS_IN_TRACE_INFO_METADATA
        )
    if TraceMetadataKey.TOKEN_USAGE in metadata:
        token_usage = _json_object(metadata[TraceMetadataKey.TOKEN_USAGE])
        columns.update({
            column: token_count_or_none(token_usage[key])
            for key, column in TOKEN_COLUMN_BY_KEY.items()
            if key in token_usage
        })
    if TraceMetadataKey.COST in metadata:
        cost = _json_object(metadata[TraceMetadataKey.COST])
        columns.update({
            column: finite_float_or_none(cost[key])
            for key, column in COST_COLUMN_BY_KEY.items()
            if key in cost
        })
    return columns


def _execute_updates(
    connection: sa.Connection,
    statement: sa.Update,
    updates: list[dict[str, Any]],
) -> int:
    if not updates:
        return 0
    result = connection.execute(statement, updates)
    # The old server remains live during prepopulation. A row may be deleted between the page
    # read and this update, so unlike the final offline migration, a short rowcount is expected.
    if result.supports_sane_multi_rowcount() and result.rowcount >= 0:
        return result.rowcount
    return len(updates)


def _trace_batch(
    connection: sa.Connection,
    tables: dict[str, sa.Table],
    cursor: str | None,
    batch_size: int,
) -> _BatchResult:
    trace_info = tables["trace_info"]
    trace_tags = tables["trace_tags"]
    trace_metadata = tables["trace_request_metadata"]
    trace_metrics = tables["trace_metrics"]

    page_statement = sa.select(
        trace_info.c.request_id,
        *(trace_info.c[column] for column in _TRACE_VALUE_COLUMNS),
    ).order_by(trace_info.c.request_id)
    if cursor is not None:
        page_statement = page_statement.where(trace_info.c.request_id > cursor)
    rows = connection.execute(page_statement.limit(batch_size)).all()
    if not rows:
        return _BatchResult(cursor, 0, 0)

    trace_ids = [row.request_id for row in rows]
    trace_names = dict.fromkeys(trace_ids)
    for row in connection.execute(
        sa.select(trace_tags.c.request_id, trace_tags.c.value).where(
            trace_tags.c.request_id.in_(trace_ids),
            trace_tags.c.key == TraceTagKey.TRACE_NAME,
        )
    ):
        trace_names[row.request_id] = row.value

    metadata_by_trace = {trace_id: {} for trace_id in trace_ids}
    for row in connection.execute(
        sa.select(trace_metadata.c.request_id, trace_metadata.c.key, trace_metadata.c.value).where(
            trace_metadata.c.request_id.in_(trace_ids),
            trace_metadata.c.key.in_([
                TraceMetadataKey.TRACE_SESSION,
                TraceMetadataKey.TOKEN_USAGE,
                TraceMetadataKey.COST,
            ]),
        )
    ):
        metadata_by_trace[row.request_id][row.key] = row.value

    metrics_by_trace = {trace_id: {} for trace_id in trace_ids}
    for row in connection.execute(
        sa.select(trace_metrics.c.request_id, trace_metrics.c.key, trace_metrics.c.value).where(
            trace_metrics.c.request_id.in_(trace_ids),
            trace_metrics.c.key.in_(list(TOKEN_COLUMN_BY_KEY)),
        )
    ):
        metrics_by_trace[row.request_id][TOKEN_COLUMN_BY_KEY[row.key]] = token_count_or_none(
            row.value
        )

    update_statement = (
        trace_info
        .update()
        .where(trace_info.c.request_id == sa.bindparam("request_id_param"))
        .values(**{column: sa.bindparam(column) for column in _TRACE_VALUE_COLUMNS})
    )
    updates = []
    for row in rows:
        expected = dict.fromkeys(_TRACE_VALUE_COLUMNS)
        expected["trace_name"] = _bounded_string_or_none(
            trace_names[row.request_id], MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE
        )
        expected.update(analytics_columns_from_metadata(metadata_by_trace[row.request_id]))
        for column, value in metrics_by_trace[row.request_id].items():
            expected[column] = value
        if not _values_match(row, expected):
            updates.append({"request_id_param": row.request_id, **expected})

    updated = _execute_updates(connection, update_statement, updates)
    return _BatchResult(trace_ids[-1], len(rows), updated)


def _span_metrics_for_keys_query(
    span_metrics: sa.Table,
    span_keys: list[tuple[str, str]],
    dialect_name: str,
) -> sa.Select:
    columns = [
        span_metrics.c.trace_id,
        span_metrics.c.span_id,
        span_metrics.c.key,
        span_metrics.c.value,
    ]
    if dialect_name == "mssql":
        batch_keys = sa.values(
            sa.column("trace_id", span_metrics.c.trace_id.type),
            sa.column("span_id", span_metrics.c.span_id.type),
            name="span_keys",
        ).data(span_keys)
        return (
            sa
            .select(*columns)
            .select_from(
                span_metrics.join(
                    batch_keys,
                    sa.and_(
                        span_metrics.c.trace_id == batch_keys.c.trace_id,
                        span_metrics.c.span_id == batch_keys.c.span_id,
                    ),
                )
            )
            .where(span_metrics.c.key.in_(list(COST_COLUMN_BY_KEY)))
        )
    return sa.select(*columns).where(
        sa.tuple_(span_metrics.c.trace_id, span_metrics.c.span_id).in_(span_keys),
        span_metrics.c.key.in_(list(COST_COLUMN_BY_KEY)),
    )


def _span_batch(
    connection: sa.Connection,
    tables: dict[str, sa.Table],
    cursor: tuple[str, str] | None,
    batch_size: int,
) -> _BatchResult:
    spans = tables["spans"]
    span_metrics = tables["span_metrics"]
    page_statement = sa.select(
        spans.c.trace_id,
        spans.c.span_id,
        spans.c.dimension_attributes,
        *(spans.c[column] for column in _SPAN_VALUE_COLUMNS),
    ).order_by(spans.c.trace_id, spans.c.span_id)
    if cursor is not None:
        last_trace_id, last_span_id = cursor
        page_statement = page_statement.where(
            sa.or_(
                spans.c.trace_id > last_trace_id,
                sa.and_(spans.c.trace_id == last_trace_id, spans.c.span_id > last_span_id),
            )
        )
    rows = connection.execute(page_statement.limit(batch_size)).all()
    if not rows:
        return _BatchResult(cursor, 0, 0)

    span_keys = [(row.trace_id, row.span_id) for row in rows]
    metrics_by_span = {span_key: {} for span_key in span_keys}
    for row in connection.execute(
        _span_metrics_for_keys_query(span_metrics, span_keys, connection.dialect.name)
    ):
        span_key = (row.trace_id, row.span_id)
        if span_key in metrics_by_span:
            metrics_by_span[span_key][COST_COLUMN_BY_KEY[row.key]] = finite_float_or_none(row.value)

    update_statement = (
        spans
        .update()
        .where(
            spans.c.trace_id == sa.bindparam("trace_id_param"),
            spans.c.span_id == sa.bindparam("span_id_param"),
        )
        .values(**{column: sa.bindparam(column) for column in _SPAN_VALUE_COLUMNS})
    )
    updates = []
    for row in rows:
        span_key = (row.trace_id, row.span_id)
        dimensions = _json_object(row.dimension_attributes)
        expected = dict.fromkeys(_SPAN_VALUE_COLUMNS)
        expected.update(metrics_by_span[span_key])
        expected["model_name"] = bounded_model_dimension(dimensions.get(SpanAttributeKey.MODEL))
        expected["model_provider"] = bounded_model_dimension(
            dimensions.get(SpanAttributeKey.MODEL_PROVIDER)
        )
        if not _values_match(row, expected):
            updates.append({
                "trace_id_param": row.trace_id,
                "span_id_param": row.span_id,
                **expected,
            })

    updated = _execute_updates(connection, update_statement, updates)
    return _BatchResult(span_keys[-1], len(rows), updated)


def _assessment_batch(
    connection: sa.Connection,
    tables: dict[str, sa.Table],
    cursor: str | None,
    batch_size: int,
) -> _BatchResult:
    trace_info = tables["trace_info"]
    assessments = tables["assessments"]
    page_statement = (
        sa
        .select(
            assessments.c.assessment_id,
            assessments.c.value,
            trace_info.c.experiment_id.label("source_experiment_id"),
            trace_info.c.timestamp_ms.label("source_trace_timestamp_ms"),
            *(assessments.c[column] for column in _ASSESSMENT_VALUE_COLUMNS),
        )
        .join(trace_info, trace_info.c.request_id == assessments.c.trace_id)
        .order_by(assessments.c.assessment_id)
    )
    if cursor is not None:
        page_statement = page_statement.where(assessments.c.assessment_id > cursor)
    rows = connection.execute(page_statement.limit(batch_size)).all()
    if not rows:
        return _BatchResult(cursor, 0, 0)

    update_statement = (
        assessments
        .update()
        .where(assessments.c.assessment_id == sa.bindparam("assessment_id_param"))
        .values(**{column: sa.bindparam(column) for column in _ASSESSMENT_VALUE_COLUMNS})
    )
    updates = []
    for row in rows:
        try:
            value = json.loads(row.value)
        except (TypeError, ValueError):
            value = row.value
        aggregate_value, is_numeric_value = assessment_aggregate(value)
        expected = {
            "experiment_id": row.source_experiment_id,
            "trace_timestamp_ms": row.source_trace_timestamp_ms,
            "aggregate_value": aggregate_value,
            "is_numeric_value": is_numeric_value,
        }
        if not _values_match(row, expected):
            updates.append({"assessment_id_param": row.assessment_id, **expected})

    updated = _execute_updates(connection, update_statement, updates)
    return _BatchResult(rows[-1].assessment_id, len(rows), updated)


def _run_entity_batches(
    engine,
    tables,
    batch_size,
    batch_function,
    entity_name,
    progress_callback,
    progress_every_batches,
) -> EntityPrepopulationStats:
    cursor = None
    scanned = 0
    updated = 0
    batch_count = 0
    while True:
        with engine.begin() as connection:
            result = batch_function(connection, tables, cursor, batch_size)
        if result.scanned == 0:
            return EntityPrepopulationStats(scanned=scanned, updated=updated)
        cursor = result.cursor
        scanned += result.scanned
        updated += result.updated
        batch_count += 1
        if progress_callback is not None and (
            batch_count == 1 or batch_count % progress_every_batches == 0
        ):
            progress_callback(
                entity_name,
                EntityPrepopulationStats(scanned=scanned, updated=updated),
            )


def prepopulate_trace_analytics(
    engine: sa.Engine,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_callback: ProgressCallback | None = None,
    progress_every_batches: int = DEFAULT_PROGRESS_EVERY_BATCHES,
) -> TraceAnalyticsPrepopulationStats:
    """Prepopulate trace analytics columns without applying the final Alembic migration.

    Reruns scan each table from the beginning and skip rows whose destination values are already
    correct. The final migration remains responsible for reconciling writes that race with a run.
    """
    if not 1 <= batch_size <= DEFAULT_BATCH_SIZE:
        raise ValueError(f"batch_size must be between 1 and {DEFAULT_BATCH_SIZE}")
    if progress_every_batches < 1:
        raise ValueError("progress_every_batches must be positive")

    with engine.connect() as connection:
        _validate_schema_revision(connection)
        _validate_source_schema(connection)
        source_tables = _reflect_tables(connection)
        _validate_required_trace_joins(connection, source_tables)

    try:
        with engine.begin() as connection:
            ensure_analytics_columns(
                connection,
                lock_timeout_seconds=DEFAULT_DDL_LOCK_TIMEOUT_SECONDS,
            )
    except sa.exc.OperationalError as e:
        raise RuntimeError(
            "Could not add the trace analytics columns. Verify ALTER permission and retry during a "
            "low-traffic period after long-running transactions finish."
        ) from e

    with engine.connect() as connection:
        tables = _reflect_tables(connection)

    return TraceAnalyticsPrepopulationStats(
        traces=_run_entity_batches(
            engine,
            tables,
            batch_size,
            _trace_batch,
            "traces",
            progress_callback,
            progress_every_batches,
        ),
        spans=_run_entity_batches(
            engine,
            tables,
            batch_size,
            _span_batch,
            "spans",
            progress_callback,
            progress_every_batches,
        ),
        assessments=_run_entity_batches(
            engine,
            tables,
            batch_size,
            _assessment_batch,
            "assessments",
            progress_callback,
            progress_every_batches,
        ),
    )
