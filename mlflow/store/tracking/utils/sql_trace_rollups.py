"""Opt-in SQL daily rollup planning for trace analytics.

Daily rollups precompute per-day aggregates (``sample_count``, ``sum``, ``min``, ``max``, and, for
trace metrics on PostgreSQL, daily ``p50``/``p90``/``p99``) keyed by
``(experiment_id, rollup_day, metric_name, grouping_set, <dimension columns>)``. Eligible dashboard
queries then read a handful of daily summary rows instead of scanning millions of raw rows. Rollups
are opt-in via :data:`MLFLOW_SQL_TRACE_ROLLUPS_ENABLED` and always fall back to the raw query path,
so an absent, stale, or partially built rollup can never change a query result, only its speed.

This module holds the backend-agnostic planning layer plus the read-serving helpers: the
family/grouping/metric taxonomy, the runtime-config gate, the read-eligibility decision, the
UTC-midnight range split, and reading rollup rows to assemble a response. Populating the rollup
tables lives in the separate rollup maintenance job.

The reader trusts a full UTC day only when a matching rollup row exists for the metric and grouping
set and no rebuild is queued for that day and family; it also demotes a covered day back to the raw
path when the row is missing a column the request needs (a null aggregate value). The single-row
``global`` grouping set is self-verifying. Trace-status rows are served only when their distinct
status counts add up to the corresponding global count, which proves the materialized breakdown is
complete. Span model/provider grouping sets rely on the maintenance contract that replacement and
rebuild-queue removal publish a complete partition atomically.

The rollup tables carry no ``workspace`` column (the shipped analytics schema scopes purely on
``experiment_id``), so workspace isolation is preserved by the caller only routing experiment ids
the requester may access. Hard experiment deletion removes these non-FK rows before an integer id
can be reused.
"""

from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum

from sqlalchemy.orm import Session

from mlflow.entities.trace_metrics import (
    AggregationType,
    MetricAggregation,
    MetricDataPoint,
    MetricViewType,
)
from mlflow.environment_variables import MLFLOW_SQL_TRACE_ROLLUPS_ENABLED
from mlflow.store.db import db_types
from mlflow.store.tracking.dbmodels.models import (
    SqlAssessmentDailyRollup,
    SqlSpanCostDailyRollup,
    SqlTraceMetricDailyRollup,
    SqlTraceRollupRebuild,
)
from mlflow.store.tracking.utils.sql_trace_metrics_utils import TIME_BUCKET_LABEL
from mlflow.tracing.constant import (
    AssessmentMetricKey,
    SpanMetricDimensionKey,
    SpanMetricKey,
    TraceMetricDimensionKey,
    TraceMetricKey,
)

MS_PER_DAY = 86_400_000
DAILY_INTERVAL_SECONDS = 86_400

# Keep every date ``IN`` predicate comfortably below MSSQL's 2,100-parameter statement limit and
# bound planner memory for caller-controlled timestamp ranges. A larger request remains valid; it
# simply uses the authoritative raw query.
MAX_ROLLUP_DAYS = 1_000

# Scattered missing or invalidated days can otherwise produce a very large OR expression even
# though the raw portions are executed in one statement. Above this threshold the single full raw
# query is both simpler and predictably cheaper.
MAX_RAW_RANGES = 8

# Daily percentiles are not composable into coarser buckets, so rollups only serve these three exact
# daily percentile values, and only for single-experiment complete-UTC-day requests.
SUPPORTED_PERCENTILE_VALUES = frozenset({50.0, 90.0, 99.0})


class RollupFamily(str, Enum):
    """Rollup table family, persisted in ``sql_trace_rollup_rebuild_queue.rollup_family``."""

    TRACE_METRIC = "trace_metric"
    SPAN_COST = "span_cost"
    ASSESSMENT = "assessment"

    def __str__(self) -> str:
        return self.value


class GroupingSet(str, Enum):
    """Dimension set a rollup row aggregates. Persisted in ``grouping_set`` columns.

    ``global`` aggregates across all rows for the experiment and day without an optional dimension;
    global rows are identified by this value, not by null dimension columns.
    """

    GLOBAL = "global"
    STATUS = "status"
    MODEL = "model"
    PROVIDER = "provider"
    MODEL_PROVIDER = "model_provider"

    def __str__(self) -> str:
        return self.value


VIEW_TYPE_TO_FAMILY: dict[MetricViewType, RollupFamily] = {
    MetricViewType.TRACES: RollupFamily.TRACE_METRIC,
    MetricViewType.SPANS: RollupFamily.SPAN_COST,
    MetricViewType.ASSESSMENTS: RollupFamily.ASSESSMENT,
}

# Metrics each family materializes into rollups. Metrics outside these sets always use the raw path:
# session counts (not composable across days/experiments), span counts and span latency (the span
# rollup table is cost-only), and any assessment metric grouped by name or value.
ROLLUP_METRICS: dict[RollupFamily, frozenset[str]] = {
    RollupFamily.TRACE_METRIC: frozenset({
        TraceMetricKey.TRACE_COUNT,
        TraceMetricKey.LATENCY,
        TraceMetricKey.INPUT_TOKENS,
        TraceMetricKey.OUTPUT_TOKENS,
        TraceMetricKey.TOTAL_TOKENS,
        TraceMetricKey.CACHE_READ_INPUT_TOKENS,
        TraceMetricKey.CACHE_CREATION_INPUT_TOKENS,
    }),
    RollupFamily.SPAN_COST: frozenset({
        SpanMetricKey.INPUT_COST,
        SpanMetricKey.OUTPUT_COST,
        SpanMetricKey.TOTAL_COST,
    }),
    RollupFamily.ASSESSMENT: frozenset({
        AssessmentMetricKey.ASSESSMENT_COUNT,
        AssessmentMetricKey.ASSESSMENT_VALUE,
    }),
}

# Only trace-metric rollups store daily percentile columns, and only PostgreSQL computes them.
PERCENTILE_FAMILIES = frozenset({RollupFamily.TRACE_METRIC})
PERCENTILE_BACKENDS = frozenset({db_types.POSTGRES})

# Raw span ranges and buckets use span start time, the same timestamp used to assign span-cost
# rollup days. All three materialized families can therefore be reproduced exactly.
SERVABLE_FAMILIES = frozenset(RollupFamily)

# Publication is atomic with rebuild-queue removal, making every materialized grouping set
# readable. Trace status additionally has a cheap global-count completeness check.
SERVABLE_GROUPING_SETS = frozenset(GroupingSet)

_ROLLUP_READ_ISOLATION_LEVEL = {
    db_types.POSTGRES: "REPEATABLE READ",
    db_types.MYSQL: "REPEATABLE READ",
    db_types.MSSQL: "REPEATABLE READ",
}


def rollups_enabled() -> bool:
    """Whether opt-in SQL trace rollups are enabled for this deployment."""
    return MLFLOW_SQL_TRACE_ROLLUPS_ENABLED.get()


def configure_rollup_read_snapshot(session: Session, db_type: str) -> None:
    """Start eligible multi-statement reads at a stable backend isolation level.

    PostgreSQL and MySQL use a repeatable snapshot. MSSQL repeatable-read locks stable rows; a
    final rebuild-queue check catches queue-row phantoms. SQLite's managed session already starts
    reads in its serializable transaction mode, so it needs no per-connection override.
    """
    if rollups_enabled() and (isolation_level := _ROLLUP_READ_ISOLATION_LEVEL.get(db_type)):
        session.connection(execution_options={"isolation_level": isolation_level})


@dataclass
class UtcDaySplit:
    """Result of splitting an inclusive ``[start, end]`` ms range at UTC-midnight boundaries.

    Args:
        covered_day_starts_ms: Epoch-ms UTC-midnight starts of full days entirely inside the range.
            These are candidates that a reader may serve from rollups if matching rows exist and no
            rebuild is queued; any that are not covered fall back to their own raw day range.
        raw_ranges: Inclusive ``(start_ms, end_ms)`` sub-ranges for the partial first/last days that
            can never be served by whole-day rollups.
    """

    covered_day_starts_ms: list[int]
    raw_ranges: list[tuple[int, int]]


def split_range_into_utc_days(start_time_ms: int | None, end_time_ms: int | None) -> UtcDaySplit:
    """Split an inclusive ``[start_time_ms, end_time_ms]`` range into full UTC days and raw edges.

    A full UTC day ``[D, D + MS_PER_DAY)`` is a covered candidate only when the request range fully
    contains it, i.e. ``start_time_ms <= D`` and ``end_time_ms >= D + MS_PER_DAY - 1``. This matches
    the raw query filter (``timestamp >= start AND timestamp <= end``) so a rollup-served day is
    exactly equivalent to the raw aggregate for that day. Partial first/last days become raw ranges.
    """
    if start_time_ms is None or end_time_ms is None or start_time_ms > end_time_ms:
        has_range = start_time_ms is not None and end_time_ms is not None
        raw = [(start_time_ms, end_time_ms)] if has_range else []
        return UtcDaySplit(covered_day_starts_ms=[], raw_ranges=raw)

    # Integer ceiling division avoids float overflow and precision loss for caller-controlled epoch
    # values.
    first_full_day_start = -(-start_time_ms // MS_PER_DAY) * MS_PER_DAY
    # Greatest UTC midnight D such that the whole day [D, D + MS_PER_DAY) fits inside the range,
    # i.e. D + MS_PER_DAY - 1 <= end_time_ms.
    last_full_day_start = ((end_time_ms + 1) // MS_PER_DAY) * MS_PER_DAY - MS_PER_DAY

    full_day_count = max(0, (last_full_day_start - first_full_day_start) // MS_PER_DAY + 1)
    if full_day_count > MAX_ROLLUP_DAYS:
        return UtcDaySplit(covered_day_starts_ms=[], raw_ranges=[(start_time_ms, end_time_ms)])

    covered = list(range(first_full_day_start, last_full_day_start + MS_PER_DAY, MS_PER_DAY))

    raw_ranges: list[tuple[int, int]] = []
    if not covered:
        raw_ranges.append((start_time_ms, end_time_ms))
    else:
        if start_time_ms < first_full_day_start:
            raw_ranges.append((start_time_ms, first_full_day_start - 1))
        covered_end_exclusive = last_full_day_start + MS_PER_DAY
        if covered_end_exclusive <= end_time_ms:
            raw_ranges.append((covered_end_exclusive, end_time_ms))

    return UtcDaySplit(covered_day_starts_ms=covered, raw_ranges=raw_ranges)


def resolve_grouping_set(
    view_type: MetricViewType, dimensions: list[str] | None
) -> GroupingSet | None:
    """Map requested grouping dimensions to a materialized grouping set, or ``None`` if unsupported.

    User-controlled high-cardinality dimensions (trace name, assessment name/value, span name/type/
    status) have no rollup grouping set and always return ``None`` so the request stays raw.
    """
    dims = set(dimensions or [])
    match view_type:
        case MetricViewType.TRACES:
            if not dims:
                return GroupingSet.GLOBAL
            if dims == {TraceMetricDimensionKey.TRACE_STATUS}:
                return GroupingSet.STATUS
        case MetricViewType.SPANS:
            if not dims:
                return GroupingSet.GLOBAL
            if dims == {SpanMetricDimensionKey.SPAN_MODEL_NAME}:
                return GroupingSet.MODEL
            if dims == {SpanMetricDimensionKey.SPAN_MODEL_PROVIDER}:
                return GroupingSet.PROVIDER
            if dims == {
                SpanMetricDimensionKey.SPAN_MODEL_NAME,
                SpanMetricDimensionKey.SPAN_MODEL_PROVIDER,
            }:
                return GroupingSet.MODEL_PROVIDER
        case MetricViewType.ASSESSMENTS:
            if not dims:
                return GroupingSet.GLOBAL
    return None


@dataclass
class RollupReadPlan:
    """A validated plan describing how to serve a query from rollups.

    Args:
        family: The rollup table family to read.
        metric_name: The metric whose rollup rows to read (matches the request metric name).
        grouping_set: The materialized grouping set to read.
        dimensions: Requested grouping dimensions in API order.
        aggregations: The requested aggregations, all confirmed rollup-servable.
        bucketed: ``True`` for one data point per UTC day (daily interval); ``False`` for a single
            aggregate over the whole range.
        experiment_id: The single experiment id to read (rollup serving is single-experiment only).
        covered_day_starts_ms: Candidate full UTC days that may be served from rollups.
        raw_ranges: Inclusive ``(start_ms, end_ms)`` partial-day ranges that must be served raw.
        uses_percentiles: Whether any requested aggregation is a percentile.
    """

    family: RollupFamily
    metric_name: str
    grouping_set: GroupingSet
    dimensions: list[str]
    aggregations: list[MetricAggregation]
    bucketed: bool
    experiment_id: int
    covered_day_starts_ms: list[int]
    raw_ranges: list[tuple[int, int]]
    uses_percentiles: bool


def _aggregation_servable(
    aggregation: MetricAggregation, family: RollupFamily, db_type: str, bucketed: bool
) -> bool:
    if aggregation.aggregation_type != AggregationType.PERCENTILE:
        # COUNT/SUM/AVG/MIN/MAX derive from sample_count/sum_value/min_value/max_value in every
        # family.
        return True
    # Percentiles are only materialized for trace metrics on supported backends, only for the three
    # exact daily percentile values, and only when bucketed by day (a daily percentile is not
    # composable across days into a single range aggregate).
    return (
        bucketed
        and family in PERCENTILE_FAMILIES
        and db_type in PERCENTILE_BACKENDS
        and aggregation.percentile_value in SUPPORTED_PERCENTILE_VALUES
    )


def resolve_rollup_read(
    view_type: MetricViewType,
    metric_name: str,
    aggregations: list[MetricAggregation],
    dimensions: list[str] | None,
    filters: list[str] | None,
    time_interval_seconds: int | None,
    start_time_ms: int | None,
    end_time_ms: int | None,
    experiment_ids: list[int],
    db_type: str,
) -> RollupReadPlan | None:
    """Decide whether a trace metrics query can be served from rollups.

    Returns a :class:`RollupReadPlan` when at least one full UTC day of the request is rollup
    eligible, or ``None`` to signal the caller to use the raw path. Returning ``None`` is always a
    safe, correct choice; this function never changes results, only whether rollups are consulted.

    Requests fall back to raw for any of: rollups disabled, an unsupported view/metric/dimension/
    aggregation, a grouping set not yet servable from rollups, any filter (rollups are unfiltered
    aggregates), a non-daily bucket width, a missing time range, more than one experiment, or a
    range containing no full UTC day.
    """
    if not rollups_enabled():
        return None

    family = VIEW_TYPE_TO_FAMILY.get(view_type)
    if family is None or family not in SERVABLE_FAMILIES:
        return None
    if metric_name not in ROLLUP_METRICS[family]:
        return None

    # ``all([])`` is true, but an empty aggregation request produces no raw data points and must
    # not be turned into rollup points with empty value maps.
    if not aggregations:
        return None

    # Rollups are unfiltered daily aggregates; any request-level filter must use the raw path.
    if filters:
        return None

    grouping_set = resolve_grouping_set(view_type, dimensions)
    if grouping_set is None or grouping_set not in SERVABLE_GROUPING_SETS:
        return None

    # Rollups are UTC-day aligned: serve only single-day buckets or a single whole-range aggregate.
    if time_interval_seconds is not None and time_interval_seconds != DAILY_INTERVAL_SECONDS:
        return None
    bucketed = time_interval_seconds == DAILY_INTERVAL_SECONDS

    if start_time_ms is None or end_time_ms is None:
        return None

    # Rollup serving is single-experiment: multi-experiment coverage cannot distinguish "no data
    # that day" from "not built yet", and daily percentiles are single-experiment by definition.
    if len(experiment_ids) != 1:
        return None

    if not all(_aggregation_servable(agg, family, db_type, bucketed) for agg in aggregations):
        return None

    split = split_range_into_utc_days(start_time_ms, end_time_ms)
    if not split.covered_day_starts_ms:
        return None

    # Date columns cannot represent arbitrary integer epochs. Out-of-range requests remain valid
    # raw SQL predicates, but must never fail while the optional planner converts day starts.
    try:
        _day_start_ms_to_date(split.covered_day_starts_ms[0])
        _day_start_ms_to_date(split.covered_day_starts_ms[-1])
    except (OverflowError, OSError, ValueError):
        return None

    return RollupReadPlan(
        family=family,
        metric_name=metric_name,
        grouping_set=grouping_set,
        dimensions=list(dimensions or []),
        aggregations=aggregations,
        bucketed=bucketed,
        experiment_id=experiment_ids[0],
        covered_day_starts_ms=split.covered_day_starts_ms,
        raw_ranges=split.raw_ranges,
        uses_percentiles=any(
            agg.aggregation_type == AggregationType.PERCENTILE for agg in aggregations
        ),
    )


FAMILY_MODEL = {
    RollupFamily.TRACE_METRIC: SqlTraceMetricDailyRollup,
    RollupFamily.SPAN_COST: SqlSpanCostDailyRollup,
    RollupFamily.ASSESSMENT: SqlAssessmentDailyRollup,
}


def _day_start_ms_to_date(day_start_ms: int) -> date:
    return datetime.fromtimestamp(day_start_ms / 1000, tz=timezone.utc).date()


def _day_start_ms_to_iso(day_start_ms: int) -> str:
    # Matches the raw path's time_bucket rendering in convert_results_to_metric_data_points.
    return datetime.fromtimestamp(day_start_ms / 1000, tz=timezone.utc).isoformat()


def _rollup_row_dimensions(plan: RollupReadPlan, row) -> dict[str, str]:
    match plan.grouping_set:
        case GroupingSet.STATUS:
            return {TraceMetricDimensionKey.TRACE_STATUS: row.trace_status}
        case GroupingSet.MODEL:
            return {SpanMetricDimensionKey.SPAN_MODEL_NAME: row.model_name}
        case GroupingSet.PROVIDER:
            return {SpanMetricDimensionKey.SPAN_MODEL_PROVIDER: row.model_provider}
        case GroupingSet.MODEL_PROVIDER:
            values = {
                SpanMetricDimensionKey.SPAN_MODEL_NAME: row.model_name,
                SpanMetricDimensionKey.SPAN_MODEL_PROVIDER: row.model_provider,
            }
            return {dimension: values[dimension] for dimension in plan.dimensions}
        case _:
            return {}


def _rollup_row_value(aggregation: MetricAggregation, row) -> float | None:
    match aggregation.aggregation_type:
        case AggregationType.COUNT:
            return row.sample_count
        case AggregationType.SUM:
            return row.sum_value
        case AggregationType.AVG:
            if row.sample_count and row.sum_value is not None:
                return row.sum_value / row.sample_count
            return None
        case AggregationType.MIN:
            return row.min_value
        case AggregationType.MAX:
            return row.max_value
        case AggregationType.PERCENTILE:
            match aggregation.percentile_value:
                case 50.0:
                    return row.p50_value
                case 90.0:
                    return row.p90_value
                case 99.0:
                    return row.p99_value
    return None


def raw_aggregations_for_plan(plan: RollupReadPlan) -> list[MetricAggregation]:
    """Return raw aggregations needed to merge an unbucketed result exactly.

    Daily averages are not composable, so an unbucketed AVG is represented by SUM and COUNT
    contributions from both rollup rows and the combined raw-range query.
    """
    if plan.bucketed:
        return plan.aggregations

    aggregation_types = []
    for aggregation in plan.aggregations:
        match aggregation.aggregation_type:
            case AggregationType.AVG:
                aggregation_types.extend([AggregationType.SUM, AggregationType.COUNT])
            case aggregation_type:
                aggregation_types.append(aggregation_type)

    unique_types = dict.fromkeys(aggregation_types)
    return [
        MetricAggregation(aggregation_type=aggregation_type) for aggregation_type in unique_types
    ]


def compute_covered_day_starts(session: Session, plan: RollupReadPlan) -> list[int]:
    """Return the candidate days actually servable from rollups.

    A day is covered when a matching rollup row exists for the metric and grouping set and no
    rebuild entry is queued for the family and day. Missing or queued days fall back to the raw
    path.
    """
    date_to_ms = {_day_start_ms_to_date(ms): ms for ms in plan.covered_day_starts_ms}
    candidate_dates = list(date_to_ms.keys())
    model = FAMILY_MODEL[plan.family]

    built = {
        row_day
        for (row_day,) in session
        .query(model.rollup_day)
        .filter(
            model.experiment_id == plan.experiment_id,
            model.metric_name == plan.metric_name,
            model.grouping_set == plan.grouping_set.value,
            model.rollup_day.in_(candidate_dates),
        )
        .distinct()
    }
    queued = {
        row_day
        for (row_day,) in session.query(SqlTraceRollupRebuild.rollup_day).filter(
            SqlTraceRollupRebuild.experiment_id == plan.experiment_id,
            SqlTraceRollupRebuild.rollup_family == plan.family.value,
            SqlTraceRollupRebuild.rollup_day.in_(candidate_dates),
        )
    }
    covered = [ms for d, ms in date_to_ms.items() if d in built and d not in queued]
    return sorted(covered)


def read_rollup_data_points(
    session: Session, plan: RollupReadPlan, covered_day_starts: list[int]
) -> tuple[list[MetricDataPoint], list[int]]:
    """Assemble daily data points from rollup rows, with the day starts actually served.

    A covered day is served only when its rollup row yields a non-null value for every requested
    aggregation. A day whose row is missing a needed column (a null aggregate) is left out of the
    served set so the caller re-queries it raw rather than returning a crashing or short response,
    keeping a served result identical to the rollups-disabled result.
    """
    date_to_ms = {_day_start_ms_to_date(ms): ms for ms in covered_day_starts}
    model = FAMILY_MODEL[plan.family]
    rows = session.query(model).filter(
        model.experiment_id == plan.experiment_id,
        model.metric_name == plan.metric_name,
        model.grouping_set == plan.grouping_set.value,
        model.rollup_day.in_(list(date_to_ms.keys())),
    )

    rows_by_day: dict[date, list] = {}
    for row in rows:
        rows_by_day.setdefault(row.rollup_day, []).append(row)

    global_rows_by_day: dict[date, list] = {}
    if plan.grouping_set == GroupingSet.STATUS:
        global_rows = session.query(model).filter(
            model.experiment_id == plan.experiment_id,
            model.metric_name == plan.metric_name,
            model.grouping_set == GroupingSet.GLOBAL.value,
            model.rollup_day.in_(list(date_to_ms.keys())),
        )
        for row in global_rows:
            global_rows_by_day.setdefault(row.rollup_day, []).append(row)

    contribution_aggregations = raw_aggregations_for_plan(plan)
    data_points = []
    served_day_starts: set[int] = set()
    for rollup_day, day_rows in rows_by_day.items():
        # A global aggregate has exactly one row. Duplicate rows can occur while an external
        # publisher is replacing a day's rollup, so treat them as incomplete rather than risking
        # a duplicate data point.
        if plan.grouping_set == GroupingSet.GLOBAL and len(day_rows) != 1:
            continue

        if plan.grouping_set == GroupingSet.STATUS:
            global_rows = global_rows_by_day.get(rollup_day, [])
            statuses = [row.trace_status for row in day_rows]
            # The global row is the publication/completeness marker for a materialized status
            # breakdown. It must be unique and equal to the sum of the distinct status rows.
            if (
                len(global_rows) != 1
                or any(status is None for status in statuses)
                or len(set(statuses)) != len(statuses)
                or sum(row.sample_count for row in day_rows) != global_rows[0].sample_count
            ):
                continue

        if plan.grouping_set != GroupingSet.GLOBAL:
            dimension_keys = [tuple(_rollup_row_dimensions(plan, row).values()) for row in day_rows]
            if len(set(dimension_keys)) != len(dimension_keys):
                # Duplicate dimension rows mean publication is not a complete atomic replacement.
                continue

        day_points = []
        for row in day_rows:
            group_dims = _rollup_row_dimensions(plan, row)
            # Mirror the raw path, which drops rows whose grouping dimension is null.
            if any(v is None for v in group_dims.values()):
                break

            values = {}
            for agg in contribution_aggregations:
                value = _rollup_row_value(agg, row)
                if value is None:
                    # A requested aggregation has no value for this day; demote the whole day to
                    # raw so a grouped result cannot be only partially served.
                    break
                values[str(agg)] = value
            else:
                day_start_ms = date_to_ms[rollup_day]
                dimensions = (
                    {TIME_BUCKET_LABEL: _day_start_ms_to_iso(day_start_ms)} if plan.bucketed else {}
                )
                dimensions.update(group_dims)
                day_points.append(
                    MetricDataPoint(
                        metric_name=plan.metric_name, dimensions=dimensions, values=values
                    )
                )
                continue
            break
        else:
            served_day_starts.add(date_to_ms[rollup_day])
            data_points.extend(day_points)
    return data_points, sorted(served_day_starts)


def _coalesce_day_starts_to_ranges(day_starts: list[int]) -> list[tuple[int, int]]:
    """Merge consecutive UTC-day starts into inclusive ``(start_ms, end_ms)`` ranges."""
    ranges: list[tuple[int, int]] = []
    for day_start in sorted(day_starts):
        day_end = day_start + MS_PER_DAY - 1
        if ranges and day_start == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], day_end)
        else:
            ranges.append((day_start, day_end))
    return ranges


def remaining_raw_ranges(
    plan: RollupReadPlan, covered_day_starts: list[int]
) -> list[tuple[int, int]]:
    """Inclusive ms ranges still requiring the raw path: partial edges plus uncovered full days."""
    covered = set(covered_day_starts)
    uncovered_days = [ms for ms in plan.covered_day_starts_ms if ms not in covered]
    return plan.raw_ranges + _coalesce_day_starts_to_ranges(uncovered_days)


@dataclass
class RollupReadResult:
    """Rollup contributions plus the raw ranges needed to complete a response."""

    data_points: list[MetricDataPoint]
    raw_ranges: list[tuple[int, int]]
    served_day_starts_ms: list[int]


def serve_rollup_read(session: Session, plan: RollupReadPlan) -> RollupReadResult | None:
    """Serve the rollup-eligible portion of a query.

    Returns rollup contributions and inclusive raw ranges, or ``None`` to signal a full raw
    fallback. Unbucketed contributions use composable backing aggregates (for example SUM and
    COUNT for AVG), which the caller merges after querying all raw gaps in one statement.

    Days whose rollup row is missing a requested aggregation are demoted back into ``raw_ranges``,
    so a served response always matches the raw one.
    """
    covered = compute_covered_day_starts(session, plan)
    if not covered:
        return None
    data_points, served_day_starts = read_rollup_data_points(session, plan, covered)
    if not served_day_starts:
        return None
    raw_ranges = remaining_raw_ranges(plan, served_day_starts)
    if len(raw_ranges) > MAX_RAW_RANGES:
        return None
    return RollupReadResult(data_points, raw_ranges, served_day_starts)


def rollup_read_is_current(
    session: Session, plan: RollupReadPlan, served_day_starts_ms: list[int]
) -> bool:
    """Recheck invalidation after raw-gap reads for backends without snapshot phantoms.

    Repeatable snapshots make this a stable re-read on PostgreSQL/MySQL. On MSSQL, a rebuild row
    inserted after the initial absence check is a permitted repeatable-read phantom; seeing it here
    makes the caller abandon the mixed result and execute one authoritative raw query.
    """
    served_dates = [_day_start_ms_to_date(ms) for ms in served_day_starts_ms]
    return (
        session
        .query(SqlTraceRollupRebuild.experiment_id)
        .filter(
            SqlTraceRollupRebuild.experiment_id == plan.experiment_id,
            SqlTraceRollupRebuild.rollup_family == plan.family.value,
            SqlTraceRollupRebuild.rollup_day.in_(served_dates),
        )
        .first()
        is None
    )


def merge_unbucketed_data_points(
    plan: RollupReadPlan,
    rollup_points: list[MetricDataPoint],
    raw_points: list[MetricDataPoint],
    max_results: int,
) -> list[MetricDataPoint]:
    """Merge composable unbucketed rollup and raw contributions by grouping dimensions."""
    support_aggregations = raw_aggregations_for_plan(plan)
    grouped: dict[tuple[tuple[str, str], ...], dict[str, float]] = {}
    dimensions_by_key: dict[tuple[tuple[str, str], ...], dict[str, str]] = {}

    for point in [*rollup_points, *raw_points]:
        dimensions = point.dimensions or {}
        key = tuple(dimensions.items())
        dimensions_by_key[key] = dimensions
        accumulated = grouped.setdefault(key, {})
        for aggregation in support_aggregations:
            label = str(aggregation)
            if label not in point.values:
                continue
            value = point.values[label]
            match aggregation.aggregation_type:
                case AggregationType.COUNT | AggregationType.SUM:
                    accumulated[label] = accumulated.get(label, 0) + value
                case AggregationType.MIN:
                    accumulated[label] = min(accumulated.get(label, value), value)
                case AggregationType.MAX:
                    accumulated[label] = max(accumulated.get(label, value), value)

    data_points = []
    for key, support_values in grouped.items():
        values = {}
        for aggregation in plan.aggregations:
            label = str(aggregation)
            if aggregation.aggregation_type == AggregationType.AVG:
                count_label = str(MetricAggregation(aggregation_type=AggregationType.COUNT))
                sum_label = str(MetricAggregation(aggregation_type=AggregationType.SUM))
                sample_count = support_values.get(count_label, 0)
                sum_value = support_values.get(sum_label)
                if sample_count and sum_value is not None:
                    values[label] = sum_value / sample_count
            elif label in support_values:
                values[label] = support_values[label]
        if values:
            data_points.append(
                MetricDataPoint(
                    metric_name=plan.metric_name,
                    dimensions=dimensions_by_key[key],
                    values=values,
                )
            )
    return order_and_limit_data_points(data_points, max_results)


def _data_point_ordering(point: MetricDataPoint) -> tuple[str, tuple[str, ...]]:
    dims = point.dimensions
    time_bucket = dims.get(TIME_BUCKET_LABEL) or ""
    grouping = tuple(v or "" for k, v in dims.items() if k != TIME_BUCKET_LABEL)
    return (time_bucket, grouping)


def order_and_limit_data_points(
    data_points: list[MetricDataPoint], max_results: int
) -> list[MetricDataPoint]:
    """Order merged rollup and raw data points, then cap to ``max_results``.

    Reproduces the single-query raw path's ``ORDER BY <time_bucket>, <dimensions>`` then
    ``LIMIT max_results`` so a rollup-served response is row-for-row identical to the
    rollups-disabled response.
    """
    return sorted(data_points, key=_data_point_ordering)[:max_results]
