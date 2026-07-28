"""add sql trace analytics schema

Revision ID: 75868b020152
Revises: a8b9c0d1e2f3
Create Date: 2026-07-22 00:00:00.000000

"""

import json
import logging
import math

import sqlalchemy as sa
from alembic import op

from mlflow.tracing.constant import (
    CostKey,
    SpanAttributeKey,
    TokenUsageKey,
    TraceMetadataKey,
    TraceTagKey,
)

revision = "75868b020152"
down_revision = "a8b9c0d1e2f3"
branch_labels = None
depends_on = None

_logger = logging.getLogger(__name__)

_BATCH_SIZE = 250
_MODEL_DIMENSION_MAX_LENGTH = 500
_TOKEN_COLUMNS = {
    TokenUsageKey.INPUT_TOKENS: "input_tokens",
    TokenUsageKey.OUTPUT_TOKENS: "output_tokens",
    TokenUsageKey.TOTAL_TOKENS: "total_tokens",
    TokenUsageKey.CACHE_READ_INPUT_TOKENS: "cache_read_input_tokens",
    TokenUsageKey.CACHE_CREATION_INPUT_TOKENS: "cache_creation_input_tokens",
}
_COST_COLUMNS = {
    CostKey.INPUT_COST: "input_cost",
    CostKey.OUTPUT_COST: "output_cost",
    CostKey.TOTAL_COST: "total_cost",
}


def _workspace_column():
    return sa.Column(
        "workspace",
        sa.String(length=63),
        nullable=False,
        server_default=sa.text("'default'"),
    )


def upgrade():
    _validate_required_trace_joins()
    _add_analytics_columns()
    _backfill_trace_analytics()
    _backfill_span_analytics()
    _backfill_assessment_analytics()
    _validate_backfill()
    _create_rollup_tables()
    _create_analytics_indexes()


def downgrade():
    _drop_analytics_indexes()
    op.drop_table("sql_trace_rollup_rebuild_queue")
    op.drop_table("sql_assessment_daily_rollups")
    op.drop_table("sql_span_cost_daily_rollups")
    op.drop_table("sql_trace_metric_daily_rollups")
    _drop_analytics_columns()


def _add_analytics_columns():
    trace_columns = [
        sa.Column(
            "trace_name",
            sa.String(length=8000).with_variant(sa.Text(), "mysql"),
            nullable=True,
        ),
        sa.Column(
            "session_id",
            sa.String(length=8000).with_variant(sa.Text(), "mysql"),
            nullable=True,
        ),
        sa.Column("input_tokens", sa.Float(precision=53), nullable=True),
        sa.Column("output_tokens", sa.Float(precision=53), nullable=True),
        sa.Column("total_tokens", sa.Float(precision=53), nullable=True),
        sa.Column("cache_read_input_tokens", sa.Float(precision=53), nullable=True),
        sa.Column("cache_creation_input_tokens", sa.Float(precision=53), nullable=True),
        sa.Column("input_cost", sa.Float(precision=53), nullable=True),
        sa.Column("output_cost", sa.Float(precision=53), nullable=True),
        sa.Column("total_cost", sa.Float(precision=53), nullable=True),
    ]
    assessment_columns = [
        sa.Column("experiment_id", sa.Integer(), nullable=True),
        sa.Column("trace_timestamp_ms", sa.BigInteger(), nullable=True),
        sa.Column("aggregate_value", sa.Float(precision=53), nullable=True),
        sa.Column("is_numeric_value", sa.Boolean(), nullable=False, server_default=sa.false()),
    ]
    span_columns = [
        sa.Column("input_cost", sa.Float(precision=53), nullable=True),
        sa.Column("output_cost", sa.Float(precision=53), nullable=True),
        sa.Column("total_cost", sa.Float(precision=53), nullable=True),
        sa.Column("model_name", sa.String(length=_MODEL_DIMENSION_MAX_LENGTH), nullable=True),
        sa.Column("model_provider", sa.String(length=_MODEL_DIMENSION_MAX_LENGTH), nullable=True),
    ]

    if op.get_bind().dialect.name == "sqlite":
        for table_name, columns in (
            ("trace_info", trace_columns),
            ("assessments", assessment_columns),
            ("spans", span_columns),
        ):
            with op.batch_alter_table(table_name) as batch_op:
                for column in columns:
                    batch_op.add_column(column)
    else:
        for table_name, columns in (
            ("trace_info", trace_columns),
            ("assessments", assessment_columns),
            ("spans", span_columns),
        ):
            for column in columns:
                op.add_column(table_name, column)


def _drop_analytics_columns():
    columns_by_table = {
        "spans": ["model_provider", "model_name", "total_cost", "output_cost", "input_cost"],
        "assessments": [
            "is_numeric_value",
            "aggregate_value",
            "trace_timestamp_ms",
            "experiment_id",
        ],
        "trace_info": [
            "total_cost",
            "output_cost",
            "input_cost",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "total_tokens",
            "output_tokens",
            "input_tokens",
            "session_id",
            "trace_name",
        ],
    }
    dialect_name = op.get_bind().dialect.name
    if dialect_name == "mssql":
        op.alter_column(
            table_name="assessments",
            column_name="is_numeric_value",
            existing_type=sa.Boolean(),
            existing_nullable=False,
            server_default=None,
        )

    if dialect_name == "sqlite":
        for table_name, columns in columns_by_table.items():
            with op.batch_alter_table(table_name) as batch_op:
                if table_name == "spans":
                    # Alembic copies reflected columns during batch recreation, but SQLite rejects
                    # explicit values for stored generated columns. Recreate duration_ns so SQLite
                    # recomputes it from the copied start and end timestamps.
                    batch_op.drop_column("duration_ns")
                for column in columns:
                    batch_op.drop_column(column)
                if table_name == "spans":
                    batch_op.add_column(
                        sa.Column(
                            "duration_ns",
                            sa.BigInteger(),
                            sa.Computed(
                                "end_time_unix_nano - start_time_unix_nano",
                                persisted=True,
                            ),
                            nullable=True,
                        ),
                        insert_before="content",
                    )
    else:
        for table_name, columns in columns_by_table.items():
            for column in columns:
                op.drop_column(table_name, column)


def _create_rollup_tables():
    op.create_table(
        "sql_trace_metric_daily_rollups",
        sa.Column(
            "id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        _workspace_column(),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("rollup_day", sa.Date(), nullable=False),
        sa.Column("metric_name", sa.String(length=250), nullable=False),
        sa.Column("grouping_set", sa.String(length=50), nullable=False),
        sa.Column("trace_status", sa.String(length=50), nullable=True),
        sa.Column("sample_count", sa.BigInteger(), nullable=False),
        sa.Column("sum_value", sa.Float(precision=53), nullable=True),
        sa.Column("min_value", sa.Float(precision=53), nullable=True),
        sa.Column("max_value", sa.Float(precision=53), nullable=True),
        sa.Column("p50_value", sa.Float(precision=53), nullable=True),
        sa.Column("p90_value", sa.Float(precision=53), nullable=True),
        sa.Column("p99_value", sa.Float(precision=53), nullable=True),
        sa.PrimaryKeyConstraint("id", name="sql_trace_metric_daily_rollups_pk"),
    )
    op.create_table(
        "sql_span_cost_daily_rollups",
        sa.Column(
            "id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        _workspace_column(),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("rollup_day", sa.Date(), nullable=False),
        sa.Column("metric_name", sa.String(length=250), nullable=False),
        sa.Column("grouping_set", sa.String(length=50), nullable=False),
        sa.Column("model_name", sa.String(length=_MODEL_DIMENSION_MAX_LENGTH), nullable=True),
        sa.Column("model_provider", sa.String(length=_MODEL_DIMENSION_MAX_LENGTH), nullable=True),
        sa.Column("sample_count", sa.BigInteger(), nullable=False),
        sa.Column("sum_value", sa.Float(precision=53), nullable=True),
        sa.Column("min_value", sa.Float(precision=53), nullable=True),
        sa.Column("max_value", sa.Float(precision=53), nullable=True),
        sa.PrimaryKeyConstraint("id", name="sql_span_cost_daily_rollups_pk"),
    )
    op.create_table(
        "sql_assessment_daily_rollups",
        sa.Column(
            "id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        _workspace_column(),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("rollup_day", sa.Date(), nullable=False),
        sa.Column("metric_name", sa.String(length=250), nullable=False),
        sa.Column("grouping_set", sa.String(length=50), nullable=False),
        sa.Column("sample_count", sa.BigInteger(), nullable=False),
        sa.Column("sum_value", sa.Float(precision=53), nullable=True),
        sa.Column("min_value", sa.Float(precision=53), nullable=True),
        sa.Column("max_value", sa.Float(precision=53), nullable=True),
        sa.PrimaryKeyConstraint("id", name="sql_assessment_daily_rollups_pk"),
    )
    op.create_table(
        "sql_trace_rollup_rebuild_queue",
        _workspace_column(),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("rollup_day", sa.Date(), nullable=False),
        sa.Column("rollup_family", sa.String(length=50), nullable=False),
        sa.PrimaryKeyConstraint(
            "workspace",
            "experiment_id",
            "rollup_day",
            "rollup_family",
            name="sql_trace_rollup_rebuild_queue_pk",
        ),
    )


def _create_analytics_indexes():
    op.create_index(
        "idx_trace_rollups_lookup",
        "sql_trace_metric_daily_rollups",
        [
            "workspace",
            "experiment_id",
            "rollup_day",
            "metric_name",
            "grouping_set",
            "trace_status",
        ],
    )
    op.create_index(
        "idx_span_cost_rollups_lookup",
        "sql_span_cost_daily_rollups",
        [
            "workspace",
            "experiment_id",
            "rollup_day",
            "metric_name",
            "grouping_set",
            "model_name",
            "model_provider",
        ],
        mysql_length={"model_name": 64, "model_provider": 64},
    )
    op.create_index(
        "idx_assessment_rollups_lookup",
        "sql_assessment_daily_rollups",
        ["workspace", "experiment_id", "rollup_day", "metric_name", "grouping_set"],
    )

    span_index_options = {}
    if op.get_bind().dialect.name == "postgresql":
        span_index_options = {
            "postgresql_include": [
                "input_cost",
                "output_cost",
                "total_cost",
                "model_name",
                "model_provider",
            ],
            "postgresql_where": sa.text(
                "input_cost IS NOT NULL OR output_cost IS NOT NULL OR total_cost IS NOT NULL"
            ),
        }
    op.create_index(
        "idx_spans_cost_trace_time_cover",
        "spans",
        ["trace_id", "start_time_unix_nano"],
        **span_index_options,
    )
    op.create_index(
        "idx_spans_cost_exp_time_cover",
        "spans",
        ["experiment_id", "start_time_unix_nano"],
        **span_index_options,
    )
    op.create_index(
        "idx_assessments_exp_trace_ts",
        "assessments",
        ["experiment_id", "trace_timestamp_ms"],
    )
    op.create_index(
        "idx_assessments_exp_trace_ts_name",
        "assessments",
        ["experiment_id", "trace_timestamp_ms", "name"],
    )
    op.create_index(
        "idx_assessments_exp_name_valid",
        "assessments",
        ["experiment_id", "name", "valid"],
    )


def _drop_analytics_indexes():
    for index_name, table_name in (
        ("idx_assessments_exp_name_valid", "assessments"),
        ("idx_assessments_exp_trace_ts_name", "assessments"),
        ("idx_assessments_exp_trace_ts", "assessments"),
        ("idx_spans_cost_exp_time_cover", "spans"),
        ("idx_spans_cost_trace_time_cover", "spans"),
        ("idx_assessment_rollups_lookup", "sql_assessment_daily_rollups"),
        ("idx_span_cost_rollups_lookup", "sql_span_cost_daily_rollups"),
        ("idx_trace_rollups_lookup", "sql_trace_metric_daily_rollups"),
    ):
        op.drop_index(index_name, table_name=table_name)


def _backfill_trace_analytics():
    bind = op.get_bind()
    metadata = sa.MetaData()
    trace_info = sa.Table("trace_info", metadata, autoload_with=bind)
    trace_tags = sa.Table("trace_tags", metadata, autoload_with=bind)
    trace_metadata = sa.Table("trace_request_metadata", metadata, autoload_with=bind)
    trace_metrics = sa.Table("trace_metrics", metadata, autoload_with=bind)

    trace_name = (
        sa
        .select(trace_tags.c.value)
        .where(
            trace_tags.c.request_id == trace_info.c.request_id,
            trace_tags.c.key == TraceTagKey.TRACE_NAME,
        )
        .correlate(trace_info)
        .scalar_subquery()
    )
    session_id = (
        sa
        .select(trace_metadata.c.value)
        .where(
            trace_metadata.c.request_id == trace_info.c.request_id,
            trace_metadata.c.key == TraceMetadataKey.TRACE_SESSION,
        )
        .correlate(trace_info)
        .scalar_subquery()
    )

    update_stmt = (
        trace_info
        .update()
        .where(trace_info.c.request_id == sa.bindparam("request_id_param"))
        .values(
            input_tokens=sa.bindparam("input_tokens"),
            output_tokens=sa.bindparam("output_tokens"),
            total_tokens=sa.bindparam("total_tokens"),
            cache_read_input_tokens=sa.bindparam("cache_read_input_tokens"),
            cache_creation_input_tokens=sa.bindparam("cache_creation_input_tokens"),
            input_cost=sa.bindparam("input_cost"),
            output_cost=sa.bindparam("output_cost"),
            total_cost=sa.bindparam("total_cost"),
        )
    )
    last_request_id = None
    while True:
        page_stmt = sa.select(trace_info.c.request_id).order_by(trace_info.c.request_id)
        if last_request_id is not None:
            page_stmt = page_stmt.where(trace_info.c.request_id > last_request_id)
        batch = bind.execute(page_stmt.limit(_BATCH_SIZE)).all()
        if not batch:
            break

        batch_ids = [row.request_id for row in batch]
        # Keep large strings out of the numeric executemany while bounding the set-based update.
        bind.execute(
            trace_info
            .update()
            .where(trace_info.c.request_id.in_(batch_ids))
            .values(trace_name=trace_name, session_id=session_id)
        )
        trace_names = dict.fromkeys(batch_ids)
        for row in bind.execute(
            sa.select(trace_tags.c.request_id, trace_tags.c.value).where(
                trace_tags.c.request_id.in_(batch_ids),
                trace_tags.c.key == TraceTagKey.TRACE_NAME,
            )
        ):
            trace_names[row.request_id] = row.value

        metadata_by_trace = {trace_id: {} for trace_id in batch_ids}
        for row in bind.execute(
            sa.select(
                trace_metadata.c.request_id, trace_metadata.c.key, trace_metadata.c.value
            ).where(
                trace_metadata.c.request_id.in_(batch_ids),
                trace_metadata.c.key.in_([
                    TraceMetadataKey.TRACE_SESSION,
                    TraceMetadataKey.TOKEN_USAGE,
                    TraceMetadataKey.COST,
                ]),
            )
        ):
            metadata_by_trace[row.request_id][row.key] = row.value

        metrics_by_trace = {trace_id: {} for trace_id in batch_ids}
        for row in bind.execute(
            sa.select(trace_metrics.c.request_id, trace_metrics.c.key, trace_metrics.c.value).where(
                trace_metrics.c.request_id.in_(batch_ids),
                trace_metrics.c.key.in_(list(_TOKEN_COLUMNS)),
            )
        ):
            metrics_by_trace[row.request_id][_TOKEN_COLUMNS[row.key]] = _finite_float_or_none(
                row.value
            )

        updates = []
        expected = {}
        for trace_id in batch_ids:
            values = metadata_by_trace[trace_id]
            token_usage = _json_object(values.get(TraceMetadataKey.TOKEN_USAGE))
            tokens = {
                column: metrics_by_trace[trace_id].get(
                    column, _finite_float_or_none(token_usage.get(key))
                )
                for key, column in _TOKEN_COLUMNS.items()
            }
            cost = _json_object(values.get(TraceMetadataKey.COST))
            costs = {
                column: _finite_float_or_none(cost.get(key))
                for key, column in _COST_COLUMNS.items()
            }
            update = {
                "request_id_param": trace_id,
                **tokens,
                **{column: costs.get(column) for column in _COST_COLUMNS.values()},
            }
            updates.append(update)
            expected[trace_id] = {
                "trace_name": trace_names[trace_id],
                "session_id": values.get(TraceMetadataKey.TRACE_SESSION),
                **{key: value for key, value in update.items() if key != "request_id_param"},
            }
        bind.execute(update_stmt, updates)
        value_columns = [
            "trace_name",
            "session_id",
            *_TOKEN_COLUMNS.values(),
            *_COST_COLUMNS.values(),
        ]
        actual = {
            row.request_id: {column: getattr(row, column) for column in value_columns}
            for row in bind.execute(
                sa.select(
                    trace_info.c.request_id,
                    *(trace_info.c[column] for column in value_columns),
                ).where(trace_info.c.request_id.in_(batch_ids))
            )
        }
        _validate_backfill_batch("trace", expected, actual)
        last_request_id = batch_ids[-1]


def _backfill_span_analytics():
    bind = op.get_bind()
    metadata = sa.MetaData()
    spans = sa.Table("spans", metadata, autoload_with=bind)
    span_metrics = sa.Table("span_metrics", metadata, autoload_with=bind)
    update_stmt = (
        spans
        .update()
        .where(
            spans.c.trace_id == sa.bindparam("trace_id_param"),
            spans.c.span_id == sa.bindparam("span_id_param"),
        )
        .values(
            input_cost=sa.bindparam("input_cost"),
            output_cost=sa.bindparam("output_cost"),
            total_cost=sa.bindparam("total_cost"),
            model_name=sa.bindparam("model_name"),
            model_provider=sa.bindparam("model_provider"),
        )
    )
    truncation_counts = {"model_name": 0, "model_provider": 0}
    last_span_key = None
    while True:
        page_stmt = sa.select(
            spans.c.trace_id, spans.c.span_id, spans.c.dimension_attributes
        ).order_by(spans.c.trace_id, spans.c.span_id)
        if last_span_key is not None:
            last_trace_id, last_span_id = last_span_key
            page_stmt = page_stmt.where(
                sa.or_(
                    spans.c.trace_id > last_trace_id,
                    sa.and_(
                        spans.c.trace_id == last_trace_id,
                        spans.c.span_id > last_span_id,
                    ),
                )
            )
        batch = bind.execute(page_stmt.limit(_BATCH_SIZE)).all()
        if not batch:
            break

        span_keys = [(row.trace_id, row.span_id) for row in batch]
        span_key_filter = sa.or_(
            *(
                sa.and_(span_metrics.c.trace_id == trace_id, span_metrics.c.span_id == span_id)
                for trace_id, span_id in span_keys
            )
        )
        metrics_by_span = {span_key: {} for span_key in span_keys}
        for row in bind.execute(
            sa.select(
                span_metrics.c.trace_id,
                span_metrics.c.span_id,
                span_metrics.c.key,
                span_metrics.c.value,
            ).where(
                span_key_filter,
                span_metrics.c.key.in_(list(_COST_COLUMNS)),
            )
        ):
            span_key = (row.trace_id, row.span_id)
            if span_key in metrics_by_span:
                metrics_by_span[span_key][_COST_COLUMNS[row.key]] = _finite_float_or_none(row.value)

        updates = []
        expected = {}
        for row in batch:
            dimensions = _json_object(row.dimension_attributes)
            costs = metrics_by_span[(row.trace_id, row.span_id)]
            model_name, model_name_truncated = _bounded_string_or_none(
                dimensions.get(SpanAttributeKey.MODEL), _MODEL_DIMENSION_MAX_LENGTH
            )
            model_provider, model_provider_truncated = _bounded_string_or_none(
                dimensions.get(SpanAttributeKey.MODEL_PROVIDER), _MODEL_DIMENSION_MAX_LENGTH
            )
            truncation_counts["model_name"] += model_name_truncated
            truncation_counts["model_provider"] += model_provider_truncated
            update = {
                "trace_id_param": row.trace_id,
                "span_id_param": row.span_id,
                **{column: costs.get(column) for column in _COST_COLUMNS.values()},
                "model_name": model_name,
                "model_provider": model_provider,
            }
            updates.append(update)
            expected[(row.trace_id, row.span_id)] = {
                key: value
                for key, value in update.items()
                if key not in {"trace_id_param", "span_id_param"}
            }
        bind.execute(update_stmt, updates)
        value_columns = [*_COST_COLUMNS.values(), "model_name", "model_provider"]
        span_key_filter = sa.or_(
            *(
                sa.and_(spans.c.trace_id == trace_id, spans.c.span_id == span_id)
                for trace_id, span_id in span_keys
            )
        )
        actual = {
            (row.trace_id, row.span_id): {column: getattr(row, column) for column in value_columns}
            for row in bind.execute(
                sa.select(
                    spans.c.trace_id,
                    spans.c.span_id,
                    *(spans.c[column] for column in value_columns),
                ).where(span_key_filter)
            )
        }
        _validate_backfill_batch("span", expected, actual)
        last_span_key = span_keys[-1]

    if sum(truncation_counts.values()):
        _logger.warning(
            "Truncated span analytics dimensions to %d characters during backfill: "
            "model_name=%d, model_provider=%d",
            _MODEL_DIMENSION_MAX_LENGTH,
            truncation_counts["model_name"],
            truncation_counts["model_provider"],
        )


def _backfill_assessment_analytics():
    bind = op.get_bind()
    metadata = sa.MetaData()
    trace_info = sa.Table("trace_info", metadata, autoload_with=bind)
    assessments = sa.Table("assessments", metadata, autoload_with=bind)

    update_stmt = (
        assessments
        .update()
        .where(assessments.c.assessment_id == sa.bindparam("assessment_id_param"))
        .values(
            experiment_id=sa.bindparam("experiment_id"),
            trace_timestamp_ms=sa.bindparam("trace_timestamp_ms"),
            aggregate_value=sa.bindparam("aggregate_value"),
            is_numeric_value=sa.bindparam("is_numeric_value"),
        )
    )
    last_assessment_id = None
    while True:
        page_stmt = (
            sa
            .select(
                assessments.c.assessment_id,
                assessments.c.value,
                trace_info.c.experiment_id.label("source_experiment_id"),
                trace_info.c.timestamp_ms.label("source_trace_timestamp_ms"),
            )
            .join(trace_info, trace_info.c.request_id == assessments.c.trace_id)
            .order_by(assessments.c.assessment_id)
        )
        if last_assessment_id is not None:
            page_stmt = page_stmt.where(assessments.c.assessment_id > last_assessment_id)
        batch = bind.execute(page_stmt.limit(_BATCH_SIZE)).all()
        if not batch:
            break

        updates = []
        expected = {}
        for row in batch:
            aggregate_value, is_numeric_value = _assessment_aggregate(row.value)
            updates.append({
                "assessment_id_param": row.assessment_id,
                "experiment_id": row.source_experiment_id,
                "trace_timestamp_ms": row.source_trace_timestamp_ms,
                "aggregate_value": aggregate_value,
                "is_numeric_value": is_numeric_value,
            })
            expected[row.assessment_id] = {
                "experiment_id": row.source_experiment_id,
                "trace_timestamp_ms": row.source_trace_timestamp_ms,
                "aggregate_value": aggregate_value,
                "is_numeric_value": is_numeric_value,
            }
        bind.execute(update_stmt, updates)
        assessment_ids = list(expected)
        value_columns = [
            "experiment_id",
            "trace_timestamp_ms",
            "aggregate_value",
            "is_numeric_value",
        ]
        actual = {
            row.assessment_id: {column: getattr(row, column) for column in value_columns}
            for row in bind.execute(
                sa.select(
                    assessments.c.assessment_id,
                    *(assessments.c[column] for column in value_columns),
                ).where(assessments.c.assessment_id.in_(assessment_ids))
            )
        }
        _validate_backfill_batch("assessment", expected, actual)
        last_assessment_id = updates[-1]["assessment_id_param"]


def _validate_required_trace_joins():
    bind = op.get_bind()
    for child_table, trace_column in (("spans", "trace_id"), ("assessments", "trace_id")):
        missing = bind.execute(
            sa.text(
                f"SELECT COUNT(*) FROM {child_table} child "
                f"LEFT JOIN trace_info trace ON child.{trace_column} = trace.request_id "
                "WHERE trace.request_id IS NULL"
            )
        ).scalar_one()
        if missing:
            raise RuntimeError(
                f"Cannot backfill trace analytics: {missing} {child_table} rows "
                "have no trace_info row"
            )


def _validate_backfill():
    # Exact denormalized values are validated after each bounded backfill batch.
    bind = op.get_bind()
    missing = bind.execute(
        sa.text(
            "SELECT COUNT(*) FROM assessments "
            "WHERE experiment_id IS NULL OR trace_timestamp_ms IS NULL"
        )
    ).scalar_one()
    if missing:
        raise RuntimeError(
            f"Trace analytics assessment backfill left {missing} rows without trace dimensions"
        )


def _validate_backfill_batch(entity, expected, actual):
    for row_key, expected_values in expected.items():
        actual_values = actual.get(row_key)
        mismatched_columns = [
            column
            for column, expected_value in expected_values.items()
            if actual_values is None or actual_values.get(column) != expected_value
        ]
        if mismatched_columns:
            raise RuntimeError(
                f"Trace analytics {entity} backfill validation failed for {row_key!r}: "
                f"mismatched columns {', '.join(mismatched_columns)}"
            )


def _assessment_aggregate(value_json):
    try:
        value = json.loads(value_json)
    except (TypeError, ValueError):
        value = value_json

    if isinstance(value, bool):
        return (1.0 if value else 0.0), False
    if isinstance(value, (int, float)):
        value = float(value)
        return (value, True) if math.isfinite(value) else (None, False)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"yes", "no"}:
            return (1.0 if value == "yes" else 0.0), False
    return None, False


def _json_object(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def _finite_float_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _bounded_string_or_none(value, max_length):
    if not isinstance(value, str):
        return None, False
    return value[:max_length], len(value) > max_length
