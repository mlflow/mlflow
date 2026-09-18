"""Build job for opt-in SQL daily trace analytics rollups.

This module populates the daily rollup tables from raw trace, span, and assessment rows. It runs on
a fully migrated database (the rollup tables already exist) and is meant to be invoked from a single
worker on a schedule; application replicas never call it, they only read rollups and enqueue
rebuild entries. See :mod:`mlflow.store.tracking.utils.sql_trace_rollups` for the read-side planning
layer.

Exactly one maintenance scheduler or external CronJob may invoke this service at a time. The worker
pool below parallelizes distinct partition keys inside that one run; coordinating multiple service
instances would require a database-backed lease/claim protocol and is intentionally unsupported.

Each partition ``(experiment_id, rollup_day, family)`` is rebuilt in its own transaction that first
takes a backend-appropriate write lock on the partition's rebuild-queue entry, then atomically
replaces the rollup rows and deletes the entry. A writer that races with a rebuild locks the same
entry in the transaction that changes the source rows, so it either waits and re-enqueues after
publication or blocks the rebuild until it commits; a stale rollup is never left marked valid.
"""

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Literal

import sqlalchemy as sa
from sqlalchemy import case, func, or_, true
from sqlalchemy.orm import Session, aliased, sessionmaker

from mlflow.entities.trace_metrics import MetricViewType
from mlflow.entities.trace_state import TraceState
from mlflow.entities.trace_status import TraceStatus
from mlflow.store.tracking.dbmodels.models import (
    SqlAssessmentDailyRollup,
    SqlAssessments,
    SqlExperiment,
    SqlSpan,
    SqlSpanCostDailyRollup,
    SqlTraceInfo,
    SqlTraceMetricDailyRollup,
    SqlTraceRollupRebuild,
)
from mlflow.store.tracking.utils.sql_trace_metrics_utils import get_time_bucket_expression
from mlflow.store.tracking.utils.sql_trace_rollups import (
    DAILY_INTERVAL_SECONDS,
    FAMILY_MODEL,
    MS_PER_DAY,
    PERCENTILE_BACKENDS,
    ROLLUP_ELIGIBILITY_LAG_MS,
    GroupingSet,
    RollupFamily,
    _rollup_day_bucket_expression,
    ensure_locked_rebuild_entry,
)
from mlflow.tracing.constant import AssessmentMetricKey, SpanMetricKey, TraceMetricKey

# Exact daily percentiles materialized for trace metrics on supported backends.
_PERCENTILES: tuple[int, int, int] = (50, 90, 99)
_PERCENTILE_COLUMNS = {50: "p50_value", 90: "p90_value", 99: "p99_value"}

_BUILT_FAMILIES: tuple[RollupFamily, ...] = (
    RollupFamily.TRACE_METRIC,
    RollupFamily.SPAN_COST,
    RollupFamily.ASSESSMENT,
)

DEFAULT_MAX_PARTITIONS_PER_RUN = 1000
DEFAULT_MAX_WORKERS = 4
DEFAULT_PARTITIONS_PER_EXPERIMENT = 10
QUEUE_EXAMINATION_MULTIPLIER = 10
MIN_QUEUE_EXAMINATION_LIMIT = 100
_MIN_SOURCE_TIMESTAMP = -(2**63)

_COMPLETE_TRACE_STATUSES = tuple(
    sorted({
        *(status.value for status in TraceStatus.end_statuses()),
        TraceState.STATE_UNSPECIFIED.value,
    })
)

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _MetricSpec:
    """One rollup metric and how to aggregate it.

    Args:
        metric_name: The metric name stored in ``metric_name`` and read back by the reader.
        column: The source column. For ``count_only`` metrics it is the (non-null) column counted
            into ``sample_count``; otherwise it is the value column summed and reduced.
        count_only: ``True`` for pure count metrics (``trace_count``, ``assessment_count``) that
            store only ``sample_count``.
        percentile: ``True`` when the metric materializes daily percentiles (trace metrics only, and
            only on backends in :data:`PERCENTILE_BACKENDS`).
    """

    metric_name: str
    column: sa.Column
    count_only: bool
    percentile: bool


_TRACE_METRIC_SPECS: tuple[_MetricSpec, ...] = (
    _MetricSpec(
        TraceMetricKey.TRACE_COUNT, SqlTraceInfo.request_id, count_only=True, percentile=False
    ),
    _MetricSpec(
        TraceMetricKey.LATENCY, SqlTraceInfo.execution_time_ms, count_only=False, percentile=True
    ),
    _MetricSpec(
        TraceMetricKey.INPUT_TOKENS, SqlTraceInfo.input_tokens, count_only=False, percentile=True
    ),
    _MetricSpec(
        TraceMetricKey.OUTPUT_TOKENS, SqlTraceInfo.output_tokens, count_only=False, percentile=True
    ),
    _MetricSpec(
        TraceMetricKey.TOTAL_TOKENS, SqlTraceInfo.total_tokens, count_only=False, percentile=True
    ),
    _MetricSpec(
        TraceMetricKey.CACHE_READ_INPUT_TOKENS,
        SqlTraceInfo.cache_read_input_tokens,
        count_only=False,
        percentile=True,
    ),
    _MetricSpec(
        TraceMetricKey.CACHE_CREATION_INPUT_TOKENS,
        SqlTraceInfo.cache_creation_input_tokens,
        count_only=False,
        percentile=True,
    ),
)

_ASSESSMENT_METRIC_SPECS: tuple[_MetricSpec, ...] = (
    _MetricSpec(
        AssessmentMetricKey.ASSESSMENT_COUNT,
        SqlAssessments.assessment_id,
        count_only=True,
        percentile=False,
    ),
    _MetricSpec(
        AssessmentMetricKey.ASSESSMENT_VALUE,
        SqlAssessments.aggregate_value,
        count_only=False,
        percentile=False,
    ),
)

_SPAN_COST_METRIC_SPECS: tuple[_MetricSpec, ...] = (
    _MetricSpec(SpanMetricKey.INPUT_COST, SqlSpan.input_cost, count_only=False, percentile=False),
    _MetricSpec(SpanMetricKey.OUTPUT_COST, SqlSpan.output_cost, count_only=False, percentile=False),
    _MetricSpec(SpanMetricKey.TOTAL_COST, SqlSpan.total_cost, count_only=False, percentile=False),
)

_PartitionOutcome = Literal["built", "emptied", "deferred", "failed"]
_Candidate = tuple[RollupFamily, tuple[int, int]]


@dataclass
class RollupFamilyBuildStats:
    built: int = 0
    emptied: int = 0
    deferred: int = 0
    failed: int = 0
    skipped_cap: int = 0


@dataclass(frozen=True)
class RollupBuildStats:
    trace_metric: RollupFamilyBuildStats
    span_cost: RollupFamilyBuildStats
    assessment: RollupFamilyBuildStats


@dataclass(frozen=True)
class RollupDeleteStats:
    trace_metric: int
    span_cost: int
    assessment: int
    rebuild_queue: int


def sql_trace_rollup_rows_exist(engine: sa.Engine) -> bool:
    """Return whether any materialized SQL trace rollup rows exist."""
    table_names = set(sa.inspect(engine).get_table_names())
    models = [
        FAMILY_MODEL[family]
        for family in _BUILT_FAMILIES
        if FAMILY_MODEL[family].__tablename__ in table_names
    ]
    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        return any(
            session.query(sa.literal(1)).select_from(model).first() is not None for model in models
        )


def delete_sql_trace_rollups(engine: sa.Engine) -> RollupDeleteStats:
    """Delete all derived SQL trace rollups and queued rebuild state atomically."""
    table_names = set(sa.inspect(engine).get_table_names())
    session_factory = sessionmaker(bind=engine)
    with session_factory.begin() as session:
        deleted = {
            family: (
                session.query(FAMILY_MODEL[family]).delete(synchronize_session=False)
                if FAMILY_MODEL[family].__tablename__ in table_names
                else 0
            )
            for family in _BUILT_FAMILIES
        }
        rebuild_queue = (
            session.query(SqlTraceRollupRebuild).delete(synchronize_session=False)
            if SqlTraceRollupRebuild.__tablename__ in table_names
            else 0
        )
    return RollupDeleteStats(
        trace_metric=deleted[RollupFamily.TRACE_METRIC],
        span_cost=deleted[RollupFamily.SPAN_COST],
        assessment=deleted[RollupFamily.ASSESSMENT],
        rebuild_queue=rebuild_queue,
    )


def _bucket_to_date(day_bucket: int) -> date:
    return datetime.fromtimestamp(day_bucket * MS_PER_DAY / 1000, tz=timezone.utc).date()


def _date_to_bucket(rollup_day: date) -> int:
    dt = datetime(rollup_day.year, rollup_day.month, rollup_day.day, tzinfo=timezone.utc)
    return int(dt.timestamp()) // (MS_PER_DAY // 1000)


def _day_bucket_expr(timestamp_column: sa.Column):
    return func.floor(timestamp_column / MS_PER_DAY)


def _span_day_start_expr(db_type: str):
    """Return the raw reader's span-start daily bucket expression in epoch milliseconds."""
    return get_time_bucket_expression(MetricViewType.SPANS, DAILY_INTERVAL_SECONDS, db_type)


def _trace_is_unsettled(trace_id_column, cutoff_ms: int):
    """Return a correlated predicate for an incomplete or recently active trace."""
    trace_info = aliased(SqlTraceInfo)
    any_span = aliased(SqlSpan)
    complete_trace = sa.exists().where(
        trace_info.request_id == trace_id_column,
        trace_info.status.in_(_COMPLETE_TRACE_STATUSES),
        trace_info.timestamp_ms <= cutoff_ms,
    )
    active_span = sa.exists().where(
        any_span.trace_id == trace_id_column,
        or_(
            any_span.end_time_unix_nano.is_(None),
            any_span.end_time_unix_nano > cutoff_ms * 1_000_000,
        ),
    )
    return or_(~complete_trace, active_span)


def _trace_partition_state(
    session: Session, experiment_id: int, day_bucket: int, cutoff_ms: int
) -> tuple[bool, bool]:
    """Return ``(eligible, has_rows)`` for a trace partition's contributing traces."""
    lo = day_bucket * MS_PER_DAY
    hi = lo + MS_PER_DAY
    partition_traces = session.query(SqlTraceInfo.request_id).filter(
        SqlTraceInfo.experiment_id == experiment_id,
        SqlTraceInfo.timestamp_ms >= lo,
        SqlTraceInfo.timestamp_ms < hi,
    )
    has_rows = partition_traces.first() is not None
    has_unsettled_trace = (
        session
        .query(sa.literal(1))
        .filter(
            SqlTraceInfo.experiment_id == experiment_id,
            SqlTraceInfo.timestamp_ms >= lo,
            SqlTraceInfo.timestamp_ms < hi,
            _trace_is_unsettled(SqlTraceInfo.request_id, cutoff_ms),
        )
        .first()
        is not None
    )
    return not has_unsettled_trace, has_rows


def _assessment_partition_state(
    session: Session, experiment_id: int, day_bucket: int, cutoff_ms: int
) -> tuple[bool, bool]:
    lo = day_bucket * MS_PER_DAY
    hi = lo + MS_PER_DAY
    assessments = session.query(SqlAssessments.trace_id).filter(
        SqlAssessments.experiment_id == experiment_id,
        SqlAssessments.valid == true(),
        SqlAssessments.trace_timestamp_ms >= lo,
        SqlAssessments.trace_timestamp_ms < hi,
    )
    has_rows = assessments.first() is not None
    has_unsettled_trace = (
        assessments.filter(_trace_is_unsettled(SqlAssessments.trace_id, cutoff_ms)).first()
        is not None
    )
    return not has_unsettled_trace, has_rows


def _span_cost_partition_state(
    session: Session, experiment_id: int, day_bucket: int, cutoff_ms: int
) -> tuple[bool, bool]:
    """Return ``(eligible, has_rows)`` for a span-cost partition."""
    lo_ns = day_bucket * MS_PER_DAY * 1_000_000
    hi_ns = (day_bucket + 1) * MS_PER_DAY * 1_000_000
    partition_spans = session.query(SqlSpan.trace_id).filter(
        SqlSpan.experiment_id == experiment_id,
        SqlSpan.start_time_unix_nano >= lo_ns,
        SqlSpan.start_time_unix_nano < hi_ns,
    )
    count_value = (
        session
        .query(
            func.count(
                case((
                    or_(
                        SqlSpan.input_cost.isnot(None),
                        SqlSpan.output_cost.isnot(None),
                        SqlSpan.total_cost.isnot(None),
                    ),
                    1,
                ))
            )
        )
        .filter(
            SqlSpan.experiment_id == experiment_id,
            SqlSpan.start_time_unix_nano >= lo_ns,
            SqlSpan.start_time_unix_nano < hi_ns,
        )
        .scalar()
    )
    has_unsettled_trace = (
        partition_spans.filter(_trace_is_unsettled(SqlSpan.trace_id, cutoff_ms)).first() is not None
    )
    # Eligibility follows each contributing trace across all of its spans, not just the spans that
    # started in this UTC day. A trace must be complete and every span in that trace must be closed
    # and past the inactivity cutoff before any of its span-cost partitions can be published.
    return not has_unsettled_trace, bool(count_value)


def _partition_state(
    session: Session,
    family: RollupFamily,
    experiment_id: int,
    day_bucket: int,
    cutoff_ms: int,
    frozen_day_bucket: int,
) -> tuple[bool, bool]:
    # Publish only after the entire UTC day is beyond the inactivity cutoff. Consequently every
    # later backdated write into a published day also passes the writer-side invalidation cutoff.
    if day_bucket >= frozen_day_bucket:
        return False, False
    if family == RollupFamily.TRACE_METRIC:
        return _trace_partition_state(session, experiment_id, day_bucket, cutoff_ms)
    if family == RollupFamily.SPAN_COST:
        return _span_cost_partition_state(session, experiment_id, day_bucket, cutoff_ms)
    return _assessment_partition_state(session, experiment_id, day_bucket, cutoff_ms)


def _trace_percentiles(
    session: Session,
    base_filters: list[sa.ColumnElement],
    group_column: sa.Column | None,
) -> dict[str | None, dict[tuple[str, int], float | None]]:
    columns = [
        func
        .percentile_cont(percentile / 100.0)
        .within_group(spec.column)
        .label(f"{spec.metric_name}__p{percentile}")
        for spec in _TRACE_METRIC_SPECS
        if spec.percentile
        for percentile in _PERCENTILES
    ]
    select_columns = ([group_column.label("grp")] if group_column is not None else []) + columns
    query = session.query(*select_columns).filter(*base_filters)
    if group_column is not None:
        query = query.group_by(group_column)

    result: dict[str | None, dict[tuple[str, int], float | None]] = {}
    for row in query:
        key = row.grp if group_column is not None else None
        result[key] = {
            (spec.metric_name, percentile): getattr(row, f"{spec.metric_name}__p{percentile}")
            for spec in _TRACE_METRIC_SPECS
            if spec.percentile
            for percentile in _PERCENTILES
        }
    return result


def _aggregate_columns(specs: tuple[_MetricSpec, ...]) -> list[sa.ColumnElement]:
    columns = []
    for spec in specs:
        columns.append(func.count(spec.column).label(f"{spec.metric_name}__n"))
        if not spec.count_only:
            columns.append(func.sum(spec.column).label(f"{spec.metric_name}__s"))
            columns.append(func.min(spec.column).label(f"{spec.metric_name}__mn"))
            columns.append(func.max(spec.column).label(f"{spec.metric_name}__mx"))
    return columns


def _aggregate_trace(
    session: Session, experiment_id: int, day_bucket: int, db_type: str
) -> list[SqlTraceMetricDailyRollup]:
    lo = day_bucket * MS_PER_DAY
    hi = lo + MS_PER_DAY
    day = _bucket_to_date(day_bucket)
    base_filters = [
        SqlTraceInfo.experiment_id == experiment_id,
        SqlTraceInfo.timestamp_ms >= lo,
        SqlTraceInfo.timestamp_ms < hi,
    ]
    rows: list[SqlTraceMetricDailyRollup] = []
    groupings = ((GroupingSet.GLOBAL, None), (GroupingSet.STATUS, SqlTraceInfo.status))
    for grouping_set, group_column in groupings:
        select_columns = (
            [group_column.label("grp")] if group_column is not None else []
        ) + _aggregate_columns(_TRACE_METRIC_SPECS)
        query = session.query(*select_columns).filter(*base_filters)
        if group_column is not None:
            query = query.group_by(group_column)

        percentiles = (
            _trace_percentiles(session, base_filters, group_column)
            if db_type in PERCENTILE_BACKENDS
            else {}
        )
        for row in query:
            status_value = row.grp if group_column is not None else None
            # Raw grouped queries drop rows whose grouping dimension is null; mirror that here.
            if group_column is not None and status_value is None:
                continue
            group_percentiles = percentiles.get(
                status_value if group_column is not None else None, {}
            )
            for spec in _TRACE_METRIC_SPECS:
                rollup = SqlTraceMetricDailyRollup(
                    experiment_id=experiment_id,
                    rollup_day=day,
                    metric_name=spec.metric_name,
                    grouping_set=grouping_set.value,
                    trace_status=status_value,
                    sample_count=getattr(row, f"{spec.metric_name}__n") or 0,
                    sum_value=None if spec.count_only else getattr(row, f"{spec.metric_name}__s"),
                    min_value=None if spec.count_only else getattr(row, f"{spec.metric_name}__mn"),
                    max_value=None if spec.count_only else getattr(row, f"{spec.metric_name}__mx"),
                )
                if spec.percentile:
                    for percentile in _PERCENTILES:
                        setattr(
                            rollup,
                            _PERCENTILE_COLUMNS[percentile],
                            group_percentiles.get((spec.metric_name, percentile)),
                        )
                rows.append(rollup)
    return rows


def _aggregate_assessment(
    session: Session, experiment_id: int, day_bucket: int, db_type: str
) -> list[SqlAssessmentDailyRollup]:
    lo = day_bucket * MS_PER_DAY
    hi = lo + MS_PER_DAY
    day = _bucket_to_date(day_bucket)
    row = (
        session
        .query(*_aggregate_columns(_ASSESSMENT_METRIC_SPECS))
        .filter(
            SqlAssessments.experiment_id == experiment_id,
            SqlAssessments.valid == true(),
            SqlAssessments.trace_timestamp_ms >= lo,
            SqlAssessments.trace_timestamp_ms < hi,
        )
        .one()
    )
    return [
        SqlAssessmentDailyRollup(
            experiment_id=experiment_id,
            rollup_day=day,
            metric_name=spec.metric_name,
            grouping_set=GroupingSet.GLOBAL.value,
            sample_count=getattr(row, f"{spec.metric_name}__n") or 0,
            sum_value=None if spec.count_only else getattr(row, f"{spec.metric_name}__s"),
            min_value=None if spec.count_only else getattr(row, f"{spec.metric_name}__mn"),
            max_value=None if spec.count_only else getattr(row, f"{spec.metric_name}__mx"),
        )
        for spec in _ASSESSMENT_METRIC_SPECS
    ]


def _aggregate_span_cost(
    session: Session, experiment_id: int, day_bucket: int
) -> list[SqlSpanCostDailyRollup]:
    lo_ns = day_bucket * MS_PER_DAY * 1_000_000
    hi_ns = (day_bucket + 1) * MS_PER_DAY * 1_000_000
    day = _bucket_to_date(day_bucket)
    base_filters = [
        SqlSpan.experiment_id == experiment_id,
        SqlSpan.start_time_unix_nano >= lo_ns,
        SqlSpan.start_time_unix_nano < hi_ns,
    ]
    groupings = (
        (GroupingSet.GLOBAL, ()),
        (GroupingSet.MODEL, (SqlSpan.model_name,)),
        (GroupingSet.PROVIDER, (SqlSpan.model_provider,)),
        (GroupingSet.MODEL_PROVIDER, (SqlSpan.model_name, SqlSpan.model_provider)),
    )
    rows: list[SqlSpanCostDailyRollup] = []
    for grouping_set, group_columns in groupings:
        select_columns = [column.label(f"group_{i}") for i, column in enumerate(group_columns)]
        query = session.query(*select_columns, *_aggregate_columns(_SPAN_COST_METRIC_SPECS)).filter(
            *base_filters
        )
        if group_columns:
            # The raw metric path drops groups with a null requested dimension.
            query = query.filter(*(column.isnot(None) for column in group_columns)).group_by(
                *group_columns
            )

        for result in query:
            model_name = None
            model_provider = None
            if grouping_set == GroupingSet.MODEL:
                model_name = result.group_0
            elif grouping_set == GroupingSet.PROVIDER:
                model_provider = result.group_0
            elif grouping_set == GroupingSet.MODEL_PROVIDER:
                model_name = result.group_0
                model_provider = result.group_1

            for spec in _SPAN_COST_METRIC_SPECS:
                sample_count = getattr(result, f"{spec.metric_name}__n") or 0
                # Do not publish an empty row for a metric with no source values. Readers will
                # correctly fall back to raw for that metric/day.
                if sample_count == 0:
                    continue
                rows.append(
                    SqlSpanCostDailyRollup(
                        experiment_id=experiment_id,
                        rollup_day=day,
                        metric_name=spec.metric_name,
                        grouping_set=grouping_set.value,
                        model_name=model_name,
                        model_provider=model_provider,
                        sample_count=sample_count,
                        sum_value=getattr(result, f"{spec.metric_name}__s"),
                        min_value=getattr(result, f"{spec.metric_name}__mn"),
                        max_value=getattr(result, f"{spec.metric_name}__mx"),
                    )
                )
    return rows


def _aggregate(
    session: Session, family: RollupFamily, experiment_id: int, day_bucket: int, db_type: str
):
    if family == RollupFamily.TRACE_METRIC:
        return _aggregate_trace(session, experiment_id, day_bucket, db_type)
    if family == RollupFamily.SPAN_COST:
        return _aggregate_span_cost(session, experiment_id, day_bucket)
    return _aggregate_assessment(session, experiment_id, day_bucket, db_type)


def _rebuild_partition(
    session_factory,
    family: RollupFamily,
    partition: tuple[int, int],
    cutoff_ms: int,
    frozen_day_bucket: int,
) -> _PartitionOutcome:
    experiment_id, day_bucket = partition
    day = _bucket_to_date(day_bucket)
    model = FAMILY_MODEL[family]
    with session_factory() as session, session.begin():
        db_type = session.get_bind().dialect.name
        entry = ensure_locked_rebuild_entry(session, family, experiment_id, day)
        eligible, has_rows = _partition_state(
            session,
            family,
            experiment_id,
            day_bucket,
            cutoff_ms,
            frozen_day_bucket,
        )
        if not eligible:
            # The day is incomplete or contributing traces are still active. Leave it queued and
            # keep serving raw rows.
            return "deferred"
        session.query(model).filter(
            model.experiment_id == experiment_id, model.rollup_day == day
        ).delete(synchronize_session=False)
        if has_rows:
            session.add_all(_aggregate(session, family, experiment_id, day_bucket, db_type))
        session.delete(entry)
        return "emptied" if not has_rows else "built"


def _bounded_query_rows(query, limit: int) -> tuple[list[Any], bool]:
    """Fetch at most ``limit`` candidates plus one bounded overflow sentinel."""
    rows = query.limit(limit + 1).all()
    return rows[:limit], len(rows) > limit


def _queued_candidates(
    session_factory,
    limit: int,
    cursor: tuple[int, date, str] | None,
) -> tuple[list[_Candidate], tuple[int, date, str] | None, RollupFamily | None]:
    with session_factory() as session:
        query = (
            session
            .query(
                SqlTraceRollupRebuild.rollup_family,
                SqlTraceRollupRebuild.experiment_id,
                SqlTraceRollupRebuild.rollup_day,
            )
            .filter(SqlTraceRollupRebuild.rollup_family.in_([f.value for f in _BUILT_FAMILIES]))
            .order_by(
                SqlTraceRollupRebuild.experiment_id,
                SqlTraceRollupRebuild.rollup_day,
                SqlTraceRollupRebuild.rollup_family,
            )
        )
        if cursor is not None:
            cursor_experiment_id, cursor_day, cursor_family = cursor
            query = query.filter(
                or_(
                    SqlTraceRollupRebuild.experiment_id > cursor_experiment_id,
                    sa.and_(
                        SqlTraceRollupRebuild.experiment_id == cursor_experiment_id,
                        SqlTraceRollupRebuild.rollup_day > cursor_day,
                    ),
                    sa.and_(
                        SqlTraceRollupRebuild.experiment_id == cursor_experiment_id,
                        SqlTraceRollupRebuild.rollup_day == cursor_day,
                        SqlTraceRollupRebuild.rollup_family > cursor_family,
                    ),
                )
            )
        rows, overflow = _bounded_query_rows(query, limit)
    candidates = [
        (RollupFamily(family), (int(experiment_id), _date_to_bucket(rollup_day)))
        for family, experiment_id, rollup_day in rows
    ]
    next_cursor = (int(rows[-1][1]), rows[-1][2], str(rows[-1][0])) if rows else cursor
    overflow_family = RollupFamily(rows[-1][0]) if overflow and rows else None
    return candidates, next_cursor, overflow_family


def _source_discovery_parts(family: RollupFamily, db_type: str):
    if family == RollupFamily.TRACE_METRIC:
        source_model = SqlTraceInfo
        experiment_id = SqlTraceInfo.experiment_id
        source_timestamp = SqlTraceInfo.timestamp_ms
        timestamp_units_per_ms = 1
        day_start_ms = _day_bucket_expr(SqlTraceInfo.timestamp_ms) * MS_PER_DAY
        source_filters = [SqlTraceInfo.timestamp_ms.isnot(None)]
    elif family == RollupFamily.SPAN_COST:
        source_model = SqlSpan
        experiment_id = SqlSpan.experiment_id
        source_timestamp = SqlSpan.start_time_unix_nano
        timestamp_units_per_ms = 1_000_000
        day_start_ms = _span_day_start_expr(db_type)
        source_filters = [
            SqlSpan.experiment_id.isnot(None),
            SqlSpan.start_time_unix_nano.isnot(None),
            or_(
                SqlSpan.input_cost.isnot(None),
                SqlSpan.output_cost.isnot(None),
                SqlSpan.total_cost.isnot(None),
            ),
        ]
    else:
        source_model = SqlAssessments
        experiment_id = SqlAssessments.experiment_id
        source_timestamp = SqlAssessments.trace_timestamp_ms
        timestamp_units_per_ms = 1
        day_start_ms = _day_bucket_expr(SqlAssessments.trace_timestamp_ms) * MS_PER_DAY
        source_filters = [
            SqlAssessments.valid == true(),
            SqlAssessments.experiment_id.isnot(None),
            SqlAssessments.trace_timestamp_ms.isnot(None),
        ]
    return (
        source_model,
        experiment_id,
        source_timestamp,
        timestamp_units_per_ms,
        day_start_ms,
        source_filters,
    )


def _candidate_experiment_ids(
    session_factory,
    family: RollupFamily,
    frozen_day_bucket: int,
) -> list[int]:
    """Find experiments with source data newer than their latest persisted rollup."""
    with session_factory() as session:
        db_type = session.get_bind().dialect.name
        (
            _,
            source_experiment_id,
            source_timestamp,
            timestamp_units_per_ms,
            _,
            source_filters,
        ) = _source_discovery_parts(family, db_type)
        rollup_model = FAMILY_MODEL[family]
        latest_day_start_ms = (
            sa
            .select(func.max(_rollup_day_bucket_expression(db_type, rollup_model.rollup_day)))
            .where(rollup_model.experiment_id == SqlExperiment.experiment_id)
            .correlate(SqlExperiment)
            .scalar_subquery()
        )
        source_probe = (
            sa
            .exists()
            .where(
                sa.and_(
                    source_experiment_id == SqlExperiment.experiment_id,
                    *source_filters,
                    source_timestamp < frozen_day_bucket * MS_PER_DAY * timestamp_units_per_ms,
                    source_timestamp
                    >= func.coalesce(
                        (latest_day_start_ms + MS_PER_DAY) * timestamp_units_per_ms,
                        _MIN_SOURCE_TIMESTAMP,
                    ),
                )
            )
            .correlate(SqlExperiment)
        )
        rows = session.query(SqlExperiment.experiment_id).filter(source_probe).all()
    return [int(experiment_id) for (experiment_id,) in rows]


def _new_candidates_for_experiment(
    session_factory,
    family: RollupFamily,
    experiment_id: int,
    frozen_day_bucket: int,
    limit: int,
) -> tuple[list[_Candidate], bool]:
    """Discover the next unbuilt days for one experiment using its source-time index."""
    with session_factory() as session:
        db_type = session.get_bind().dialect.name
        (
            source_model,
            source_experiment_id,
            source_timestamp,
            timestamp_units_per_ms,
            day_start_ms,
            source_filters,
        ) = _source_discovery_parts(family, db_type)
        rollup_model = FAMILY_MODEL[family]
        queued_days = (
            session
            .query(
                _rollup_day_bucket_expression(db_type, SqlTraceRollupRebuild.rollup_day).label(
                    "day_start_ms"
                )
            )
            .filter(
                SqlTraceRollupRebuild.rollup_family == family.value,
                SqlTraceRollupRebuild.experiment_id == experiment_id,
            )
            .subquery()
        )
        latest_day_start_ms = (
            session
            .query(func.max(_rollup_day_bucket_expression(db_type, rollup_model.rollup_day)))
            .filter(rollup_model.experiment_id == experiment_id)
            .scalar()
        )
        query = (
            session
            .query(day_start_ms.label("day_start_ms"))
            .select_from(source_model)
            .outerjoin(
                queued_days,
                queued_days.c.day_start_ms == day_start_ms,
            )
            .filter(
                source_experiment_id == experiment_id,
                *source_filters,
                source_timestamp < frozen_day_bucket * MS_PER_DAY * timestamp_units_per_ms,
                queued_days.c.day_start_ms.is_(None),
            )
            .group_by(day_start_ms)
            .order_by(day_start_ms)
        )
        if latest_day_start_ms is not None:
            query = query.filter(
                source_timestamp >= (int(latest_day_start_ms) + MS_PER_DAY) * timestamp_units_per_ms
            )
        rows, overflow = _bounded_query_rows(query, limit)
    return (
        [
            (family, (experiment_id, int(day_start_ms_value) // MS_PER_DAY))
            for (day_start_ms_value,) in rows
        ],
        overflow,
    )


def _group_candidates_by_experiment(candidates: list[_Candidate]):
    grouped: dict[int, list[_Candidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate[1][0], []).append(candidate)
    return list(grouped.items())


def _process_candidate_groups(
    session_factory,
    candidate_groups: list[tuple[int, list[_Candidate]]],
    cutoff_ms: int,
    frozen_day_bucket: int,
    max_workers: int,
    db_type: str,
) -> list[tuple[_Candidate, _PartitionOutcome]]:
    if not candidate_groups:
        return []

    def rebuild_experiment(group: tuple[int, list[_Candidate]]):
        _, candidates = group
        outcomes = []
        for candidate in candidates:
            family, partition = candidate
            try:
                outcome = _rebuild_partition(
                    session_factory,
                    family,
                    partition,
                    cutoff_ms,
                    frozen_day_bucket,
                )
            except Exception:
                experiment_id, day_bucket = partition
                _logger.exception(
                    "Failed to rebuild SQL trace rollup partition "
                    "(%s, %s, %s); skipping later partitions for this experiment.",
                    experiment_id,
                    _bucket_to_date(day_bucket),
                    family.value,
                )
                # A queued candidate remains durable because its rebuild transaction rolled back.
                # A newly discovered candidate may have failed before its marker committed, so
                # record it in a fresh transaction for a later maintenance pass.
                try:
                    with session_factory() as session, session.begin():
                        ensure_locked_rebuild_entry(
                            session,
                            family,
                            experiment_id,
                            _bucket_to_date(day_bucket),
                        )
                except Exception:
                    _logger.exception(
                        "Failed to retain SQL trace rollup rebuild marker for (%s, %s, %s).",
                        experiment_id,
                        _bucket_to_date(day_bucket),
                        family.value,
                    )
                outcomes.append((candidate, "failed"))
                break
            outcomes.append((candidate, outcome))
        return outcomes

    worker_count = 1 if db_type == "sqlite" else min(max_workers, len(candidate_groups))
    if worker_count == 1:
        return [item for group in map(rebuild_experiment, candidate_groups) for item in group]

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="trace-rollup",
    ) as executor:
        return [
            item for group in executor.map(rebuild_experiment, candidate_groups) for item in group
        ]


def _record_outcomes(
    outcomes: list[tuple[_Candidate, _PartitionOutcome]],
    family_stats: dict[RollupFamily, RollupFamilyBuildStats],
) -> tuple[int, list[str]]:
    successful = 0
    deferred_samples = []
    for (family, (experiment_id, day_bucket)), outcome in outcomes:
        stats = family_stats[family]
        match outcome:
            case "built":
                stats.built += 1
                successful += 1
            case "emptied":
                stats.emptied += 1
                successful += 1
            case "deferred":
                stats.deferred += 1
                if len(deferred_samples) < 5:
                    deferred_samples.append(
                        f"({experiment_id}, {_bucket_to_date(day_bucket)}, {family.value})"
                    )
            case "failed":
                stats.failed += 1
    return successful, deferred_samples


def run_sql_trace_rollups(
    engine: sa.Engine,
    *,
    now_ms: int | None = None,
    max_partitions_per_run: int = DEFAULT_MAX_PARTITIONS_PER_RUN,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> RollupBuildStats:
    """Build eligible daily rollups and drain the rebuild queue.

    Rebuilds each eligible or queued ``(experiment_id, rollup_day)`` partition for the trace-metric,
    span-cost, and assessment families in its own locked transaction. Callers must ensure that only
    one maintenance service instance runs at a time.

    Args:
        engine: A SQLAlchemy engine bound to a fully migrated tracking database.
        now_ms: Job start time in epoch milliseconds; defaults to the current time. Injectable so
            eligibility (the 24-hour inactivity rule) is deterministic in tests.
        max_partitions_per_run: Maximum number of successfully built or emptied partitions across
            all families in one run. Deferred queue entries do not consume this publication cap.
        max_workers: Maximum number of experiments processed concurrently. Days within one
            experiment are always processed serially. SQLite always uses one worker.

    Returns:
        Per-family build statistics.
    """
    if max_partitions_per_run < 1:
        raise ValueError("max_partitions_per_run must be positive")
    if max_workers < 1:
        raise ValueError("max_workers must be positive")

    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    cutoff_ms = now_ms - ROLLUP_ELIGIBILITY_LAG_MS
    frozen_day_bucket = cutoff_ms // MS_PER_DAY
    session_factory = sessionmaker(bind=engine)
    db_type = engine.dialect.name
    examination_limit = max(
        MIN_QUEUE_EXAMINATION_LIMIT,
        max_partitions_per_run * QUEUE_EXAMINATION_MULTIPLIER,
    )

    family_stats = {family: RollupFamilyBuildStats() for family in _BUILT_FAMILIES}
    published = 0
    examined = 0
    deferred_samples: list[str] = []
    overflow_families_seen: set[RollupFamily] = set()
    examined_candidates: set[_Candidate] = set()
    failed_experiment_ids: set[int] = set()

    # Drain the durable queue first. A keyset cursor skips deferred rows within this run, while
    # the separate examination bound prevents permanently active traces from causing an
    # unbounded scan. Only built/emptied outcomes consume the publication cap.
    queue_cursor = None
    queue_exhausted = False
    while published < max_partitions_per_run and examined < examination_limit:
        page_limit = min(
            max_partitions_per_run - published,
            examination_limit - examined,
        )
        queued, queue_cursor, overflow_family = _queued_candidates(
            session_factory,
            page_limit,
            queue_cursor,
        )
        if not queued:
            queue_exhausted = True
            break
        skipped_after_failure = [
            candidate for candidate in queued if candidate[1][0] in failed_experiment_ids
        ]
        examined += len(skipped_after_failure)
        examined_candidates.update(skipped_after_failure)
        queued = [candidate for candidate in queued if candidate[1][0] not in failed_experiment_ids]
        if not queued:
            if overflow_family is None:
                queue_exhausted = True
                break
            continue
        outcomes = _process_candidate_groups(
            session_factory,
            _group_candidates_by_experiment(queued),
            cutoff_ms,
            frozen_day_bucket,
            max_workers,
            db_type,
        )
        successful, samples = _record_outcomes(outcomes, family_stats)
        examined_candidates.update(candidate for candidate, _ in outcomes)
        failed_experiment_ids.update(
            candidate[1][0] for candidate, outcome in outcomes if outcome == "failed"
        )
        published += successful
        examined += len(outcomes)
        deferred_samples.extend(samples[: 5 - len(deferred_samples)])
        if overflow_family is None:
            queue_exhausted = True
            break
        if published >= max_partitions_per_run:
            overflow_families_seen.add(overflow_family)

    # New coverage is discovered only after the queue scan is exhausted. Discovery starts from
    # each experiment's latest persisted day and uses indexed source probes instead of grouping
    # complete source history. Experiments are shuffled once per pass, then processed in small
    # serial quotas so a large experiment cannot starve the rest.
    if queue_exhausted and published < max_partitions_per_run and examined < examination_limit:
        candidate_families: dict[int, list[RollupFamily]] = {}
        for family in _BUILT_FAMILIES:
            for experiment_id in _candidate_experiment_ids(
                session_factory,
                family,
                frozen_day_bucket,
            ):
                if experiment_id not in failed_experiment_ids:
                    candidate_families.setdefault(experiment_id, []).append(family)
        experiment_ids = list(candidate_families)
        random.shuffle(experiment_ids)

        while (
            experiment_ids and published < max_partitions_per_run and examined < examination_limit
        ):
            slots = min(
                max_partitions_per_run - published,
                examination_limit - examined,
            )
            candidate_groups = []
            for experiment_id in experiment_ids:
                if experiment_id in failed_experiment_ids:
                    continue
                if slots <= 0:
                    break
                quota = min(DEFAULT_PARTITIONS_PER_EXPERIMENT, slots)
                candidates = []
                overflow_families = []
                for family in candidate_families[experiment_id]:
                    family_candidates, overflow = _new_candidates_for_experiment(
                        session_factory,
                        family,
                        experiment_id,
                        frozen_day_bucket,
                        quota,
                    )
                    candidates.extend(
                        candidate
                        for candidate in family_candidates
                        if candidate not in examined_candidates
                    )
                    if overflow:
                        overflow_families.append(family)
                candidates.sort(
                    key=lambda item: (
                        item[1][1],
                        _BUILT_FAMILIES.index(item[0]),
                    )
                )
                if selected := candidates[:quota]:
                    candidate_groups.append((experiment_id, selected))
                    slots -= len(selected)
                overflow_families_seen.update(overflow_families)

            if not candidate_groups:
                break
            outcomes = _process_candidate_groups(
                session_factory,
                candidate_groups,
                cutoff_ms,
                frozen_day_bucket,
                max_workers,
                db_type,
            )
            successful, samples = _record_outcomes(outcomes, family_stats)
            examined_candidates.update(candidate for candidate, _ in outcomes)
            failed_experiment_ids.update(
                candidate[1][0] for candidate, outcome in outcomes if outcome == "failed"
            )
            published += successful
            examined += len(outcomes)
            deferred_samples.extend(samples[: 5 - len(deferred_samples)])

    if deferred_samples and (published < max_partitions_per_run or examined >= examination_limit):
        _logger.warning(
            "SQL trace rollup maintenance deferred %d partition(s); sample keys: %s",
            sum(stats.deferred for stats in family_stats.values()),
            ", ".join(deferred_samples),
        )
    if published >= max_partitions_per_run:
        for family in overflow_families_seen:
            family_stats[family].skipped_cap += 1
        _logger.warning(
            "SQL trace rollup maintenance reached the %d-partition publication cap; "
            "additional work may remain.",
            max_partitions_per_run,
        )
    elif examined >= examination_limit:
        for family in overflow_families_seen:
            family_stats[family].skipped_cap += 1
        _logger.warning(
            "SQL trace rollup maintenance reached the %d-partition examination limit; "
            "deferred or additional work remains.",
            examination_limit,
        )

    return RollupBuildStats(
        trace_metric=family_stats[RollupFamily.TRACE_METRIC],
        span_cost=family_stats[RollupFamily.SPAN_COST],
        assessment=family_stats[RollupFamily.ASSESSMENT],
    )
