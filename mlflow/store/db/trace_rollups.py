"""Build job for opt-in SQL daily trace analytics rollups.

This module populates the daily rollup tables from raw trace and assessment rows. It runs on a
fully migrated database (the rollup tables already exist) and is meant to be invoked from a single
worker on a schedule; application replicas never call it, they only read rollups and enqueue rebuild
entries. See :mod:`mlflow.store.tracking.utils.sql_trace_rollups` for the read-side planning layer
and the reasons span-cost rollups are neither built nor served yet.

Each partition ``(experiment_id, rollup_day, family)`` is rebuilt in its own transaction that first
takes a ``SELECT FOR UPDATE`` lock on the partition's rebuild-queue entry, then atomically replaces
the rollup rows and deletes the entry. A writer that races with a rebuild locks the same entry in
the transaction that changes the source rows, so it either waits and re-enqueues after publication
or blocks the rebuild until it commits; a stale rollup is never left marked valid.
"""

import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable, Literal

import sqlalchemy as sa
from sqlalchemy import case, func, true
from sqlalchemy.orm import Session, sessionmaker

from mlflow.store.tracking.dbmodels.models import (
    SqlAssessmentDailyRollup,
    SqlAssessments,
    SqlSpan,
    SqlTraceInfo,
    SqlTraceMetricDailyRollup,
    SqlTraceRollupRebuild,
)
from mlflow.store.tracking.utils.sql_trace_rollups import (
    FAMILY_MODEL,
    MS_PER_DAY,
    PERCENTILE_BACKENDS,
    GroupingSet,
    RollupFamily,
    ensure_locked_rebuild_entry,
)
from mlflow.tracing.constant import AssessmentMetricKey, TraceMetricKey

# A trace is eligible for rollup only once it has been inactive for this long; see the 24-hour rule
# in RFC 0006. The rule controls eligibility, not correctness: late writes re-enqueue a rebuild and
# readers fall back to raw for any queued or unbuilt day.
ROLLUP_ELIGIBILITY_LAG_MS = 24 * 60 * 60 * 1000

# Larger than any real settle time (epoch ms), used to force a trace with an open span to exceed the
# inactivity cutoff so its day is never considered eligible while a span is still running.
_ACTIVE_SENTINEL_MS = 1 << 62

# Exact daily percentiles materialized for trace metrics on supported backends.
_PERCENTILES: tuple[int, int, int] = (50, 90, 99)
_PERCENTILE_COLUMNS = {50: "p50_value", 90: "p90_value", 99: "p99_value"}

_BUILT_FAMILIES: tuple[RollupFamily, RollupFamily] = (
    RollupFamily.TRACE_METRIC,
    RollupFamily.ASSESSMENT,
)

DEFAULT_PROGRESS_EVERY_PARTITIONS = 100


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

_PartitionOutcome = Literal["built", "emptied", "deferred"]


@dataclass
class RollupFamilyBuildStats:
    built: int = 0
    emptied: int = 0
    deferred: int = 0
    skipped_cap: int = 0


@dataclass(frozen=True)
class RollupBuildStats:
    trace_metric: RollupFamilyBuildStats
    assessment: RollupFamilyBuildStats


ProgressCallback = Callable[[str, RollupFamilyBuildStats], None]


def _bucket_to_date(day_bucket: int) -> date:
    return datetime.fromtimestamp(day_bucket * MS_PER_DAY / 1000, tz=timezone.utc).date()


def _date_to_bucket(rollup_day: date) -> int:
    dt = datetime(rollup_day.year, rollup_day.month, rollup_day.day, tzinfo=timezone.utc)
    return int(dt.timestamp()) // (MS_PER_DAY // 1000)


def _day_bucket_expr(timestamp_column: sa.Column):
    return func.floor(timestamp_column / MS_PER_DAY)


def _span_settle_subquery(session: Session):
    """Per-trace span settle state: latest span end (ms) and whether any span is still open."""
    return (
        session
        .query(
            SqlSpan.trace_id.label("trace_id"),
            func.max(SqlSpan.end_time_unix_nano).label("max_end_nano"),
            func.max(case((SqlSpan.end_time_unix_nano.is_(None), 1), else_=0)).label("has_open"),
        )
        .group_by(SqlSpan.trace_id)
        .subquery()
    )


def _trace_settle_expr(span_stat) -> sa.ColumnElement:
    # A trace settles at its latest span end; a trace without spans settles at its own timestamp; a
    # trace with any open span is treated as active (sentinel) so its day cannot become eligible.
    return case(
        (span_stat.c.trace_id.is_(None), SqlTraceInfo.timestamp_ms),
        (span_stat.c.has_open == 1, _ACTIVE_SENTINEL_MS),
        else_=span_stat.c.max_end_nano / 1_000_000.0,
    )


def _eligible_trace_days(
    session: Session, cutoff_ms: int, current_day_bucket: int
) -> set[tuple[int, int]]:
    """Return ``(experiment_id, day_bucket)`` trace partitions whose every trace is inactive."""
    span_stat = _span_settle_subquery(session)
    settle = _trace_settle_expr(span_stat)
    day_bucket = _day_bucket_expr(SqlTraceInfo.timestamp_ms)
    rows = (
        session
        .query(SqlTraceInfo.experiment_id, day_bucket.label("day_bucket"))
        .outerjoin(span_stat, span_stat.c.trace_id == SqlTraceInfo.request_id)
        .filter(day_bucket < current_day_bucket)
        .group_by(SqlTraceInfo.experiment_id, day_bucket)
        .having(func.max(settle) <= cutoff_ms)
    )
    return {(int(exp), int(bucket)) for exp, bucket in rows}


def _assessment_days_with_data(session: Session) -> set[tuple[int, int]]:
    day_bucket = _day_bucket_expr(SqlAssessments.trace_timestamp_ms)
    rows = (
        session
        .query(SqlAssessments.experiment_id, day_bucket.label("day_bucket"))
        .filter(
            SqlAssessments.valid == true(),
            # The denormalized columns are nullable (online prepopulation adds them before
            # backfill, and orphaned assessments never get backfilled). Such rows are excluded from
            # every other assessment path (aggregate + has_rows filter on experiment_id/range), so
            # skip them here too rather than let ``int(None)`` abort the whole run.
            SqlAssessments.experiment_id.isnot(None),
            SqlAssessments.trace_timestamp_ms.isnot(None),
        )
        .group_by(SqlAssessments.experiment_id, day_bucket)
    )
    return {(int(exp), int(bucket)) for exp, bucket in rows}


def _built_partitions(session: Session, family: RollupFamily) -> set[tuple[int, int]]:
    model = FAMILY_MODEL[family]
    rows = session.query(model.experiment_id, model.rollup_day).distinct()
    return {(int(exp), _date_to_bucket(rollup_day)) for exp, rollup_day in rows}


def _queued_partitions(session: Session, family: RollupFamily) -> list[tuple[int, int]]:
    rows = (
        session
        .query(SqlTraceRollupRebuild.experiment_id, SqlTraceRollupRebuild.rollup_day)
        .filter(SqlTraceRollupRebuild.rollup_family == family.value)
        .order_by(SqlTraceRollupRebuild.experiment_id, SqlTraceRollupRebuild.rollup_day)
    )
    return [(int(exp), _date_to_bucket(rollup_day)) for exp, rollup_day in rows]


def _trace_partition_state(
    session: Session, experiment_id: int, day_bucket: int, cutoff_ms: int
) -> tuple[bool, bool]:
    """Return ``(eligible, has_rows)`` for a trace partition's contributing traces."""
    lo = day_bucket * MS_PER_DAY
    hi = lo + MS_PER_DAY
    span_stat = _span_settle_subquery(session)
    settle = _trace_settle_expr(span_stat)
    count_value, max_settle = (
        session
        .query(func.count(SqlTraceInfo.request_id), func.max(settle))
        .outerjoin(span_stat, span_stat.c.trace_id == SqlTraceInfo.request_id)
        .filter(
            SqlTraceInfo.experiment_id == experiment_id,
            SqlTraceInfo.timestamp_ms >= lo,
            SqlTraceInfo.timestamp_ms < hi,
        )
        .one()
    )
    has_rows = bool(count_value)
    eligible = not has_rows or (max_settle is not None and max_settle <= cutoff_ms)
    return eligible, has_rows


def _assessment_has_rows(session: Session, experiment_id: int, day_bucket: int) -> bool:
    lo = day_bucket * MS_PER_DAY
    hi = lo + MS_PER_DAY
    return (
        session
        .query(sa.literal(1))
        .filter(
            SqlAssessments.experiment_id == experiment_id,
            SqlAssessments.valid == true(),
            SqlAssessments.trace_timestamp_ms >= lo,
            SqlAssessments.trace_timestamp_ms < hi,
        )
        .first()
        is not None
    )


def _partition_state(
    session: Session,
    family: RollupFamily,
    experiment_id: int,
    day_bucket: int,
    cutoff_ms: int,
    current_day_bucket: int,
) -> tuple[bool, bool]:
    # A rollup is valid only for a complete UTC day. Keep current/future partitions queued so
    # readers continue to use raw rows until a later maintenance run can publish them safely.
    if day_bucket >= current_day_bucket:
        return False, False
    if family == RollupFamily.TRACE_METRIC:
        return _trace_partition_state(session, experiment_id, day_bucket, cutoff_ms)
    # Assessments inherit their trace's timestamp, so contributing traces are those of the same day.
    trace_eligible, trace_has_rows = _trace_partition_state(
        session, experiment_id, day_bucket, cutoff_ms
    )
    has_rows = _assessment_has_rows(session, experiment_id, day_bucket)
    eligible = trace_eligible if trace_has_rows else True
    return eligible, has_rows


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


def _aggregate(
    session: Session, family: RollupFamily, experiment_id: int, day_bucket: int, db_type: str
):
    if family == RollupFamily.TRACE_METRIC:
        return _aggregate_trace(session, experiment_id, day_bucket, db_type)
    return _aggregate_assessment(session, experiment_id, day_bucket, db_type)


def _rebuild_partition(
    session_factory,
    family: RollupFamily,
    partition: tuple[int, int],
    cutoff_ms: int,
    current_day_bucket: int,
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
            current_day_bucket,
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


def _family_candidates(
    session_factory,
    family: RollupFamily,
    eligible: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    with session_factory() as session:
        queued = _queued_partitions(session, family)
        built = _built_partitions(session, family)
    queued_set = set(queued)
    new = sorted(partition for partition in eligible if partition not in built | queued_set)
    return queued, new


def _process_candidates(
    session_factory,
    family: RollupFamily,
    candidates: list[tuple[int, int]],
    cutoff_ms: int,
    current_day_bucket: int,
    remaining_cap: int | None,
    stats: RollupFamilyBuildStats,
    progress_callback: ProgressCallback | None,
    progress_every: int,
) -> int | None:
    processed = 0
    for partition in candidates:
        if remaining_cap is not None and remaining_cap <= 0:
            stats.skipped_cap += 1
            continue
        outcome = _rebuild_partition(
            session_factory, family, partition, cutoff_ms, current_day_bucket
        )
        match outcome:
            case "built":
                stats.built += 1
            case "emptied":
                stats.emptied += 1
            case "deferred":
                stats.deferred += 1
        if outcome != "deferred" and remaining_cap is not None:
            remaining_cap -= 1
        processed += 1
        if progress_callback is not None and processed % progress_every == 0:
            progress_callback(family.value, stats)
    if progress_callback is not None:
        progress_callback(family.value, stats)
    return remaining_cap


def run_sql_trace_rollups(
    engine: sa.Engine,
    *,
    now_ms: int | None = None,
    max_partitions_per_run: int | None = None,
    progress_callback: ProgressCallback | None = None,
    progress_every: int = DEFAULT_PROGRESS_EVERY_PARTITIONS,
) -> RollupBuildStats:
    """Build eligible daily rollups and drain the rebuild queue.

    Rebuilds each eligible or queued ``(experiment_id, rollup_day)`` partition for the trace-metric
    and assessment families in its own locked transaction. Span-cost rollups are intentionally not
    built; see :data:`mlflow.store.tracking.utils.sql_trace_rollups.SERVABLE_FAMILIES`.

    Args:
        engine: A SQLAlchemy engine bound to a fully migrated tracking database.
        now_ms: Job start time in epoch milliseconds; defaults to the current time. Injectable so
            eligibility (the 24-hour inactivity rule) is deterministic in tests.
        max_partitions_per_run: Optional cap on the number of partitions actually rebuilt across all
            families in one run. Deferred partitions do not count against the cap.
        progress_callback: Optional callback invoked with ``(family, stats)`` during the run.
        progress_every: Invoke ``progress_callback`` every this many processed partitions.

    Returns:
        Per-family build statistics.
    """
    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    if max_partitions_per_run is not None and max_partitions_per_run < 1:
        raise ValueError("max_partitions_per_run must be positive")

    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    cutoff_ms = now_ms - ROLLUP_ELIGIBILITY_LAG_MS
    current_day_bucket = now_ms // MS_PER_DAY
    session_factory = sessionmaker(bind=engine)

    with session_factory() as session:
        eligible_trace_days = _eligible_trace_days(session, cutoff_ms, current_day_bucket)
        assessment_data_days = _assessment_days_with_data(session)

    eligible_by_family = {
        RollupFamily.TRACE_METRIC: eligible_trace_days,
        RollupFamily.ASSESSMENT: eligible_trace_days & assessment_data_days,
    }

    candidates_by_family = {
        family: _family_candidates(session_factory, family, eligible_by_family[family])
        for family in _BUILT_FAMILIES
    }
    family_stats = {family: RollupFamilyBuildStats() for family in _BUILT_FAMILIES}
    remaining_cap = max_partitions_per_run
    # RFC 0006 requires queued rebuilds to take priority globally. Drain both families' queues
    # before using any remaining capacity for newly eligible, previously unbuilt partitions.
    for candidate_index in (0, 1):
        for family in _BUILT_FAMILIES:
            remaining_cap = _process_candidates(
                session_factory,
                family,
                candidates_by_family[family][candidate_index],
                cutoff_ms,
                current_day_bucket,
                remaining_cap,
                family_stats[family],
                progress_callback,
                progress_every,
            )

    return RollupBuildStats(
        trace_metric=family_stats[RollupFamily.TRACE_METRIC],
        assessment=family_stats[RollupFamily.ASSESSMENT],
    )
