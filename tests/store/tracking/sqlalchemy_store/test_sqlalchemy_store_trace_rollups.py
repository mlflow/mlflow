# Read-path routing tests for opt-in SQL daily trace analytics rollups.
#
# Rollup rows are seeded directly through the ORM (the rollup build job lives elsewhere), so these
# tests cover only what the query planner owns: routing an eligible request to the rollup tables and
# falling back to the raw path for every ineligible case. Seeding by hand also lets a test store a
# deliberately wrong ("sentinel") value: a query that falls back returns the raw result and ignores
# the sentinel, while a served query returns the sentinel, pinning down exactly when rollups are
# consulted.

import uuid
from datetime import datetime, timezone

import pytest

from mlflow.entities import AssessmentSource, AssessmentSourceType, Feedback, trace_location
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_metrics import AggregationType, MetricAggregation, MetricViewType
from mlflow.entities.trace_state import TraceState
from mlflow.environment_variables import MLFLOW_SQL_TRACE_ROLLUPS_ENABLED
from mlflow.store.tracking.dbmodels.models import (
    SqlAssessmentDailyRollup,
    SqlTraceMetricDailyRollup,
    SqlTraceRollupRebuild,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.tracking.utils.sql_trace_rollups import (
    DAILY_INTERVAL_SECONDS,
    GroupingSet,
    RollupFamily,
)
from mlflow.tracing.constant import (
    AssessmentMetricKey,
    TraceMetricDimensionKey,
    TraceMetricKey,
    TraceTagKey,
)

pytestmark = pytest.mark.notrackingurimock

MS_PER_DAY = 86_400_000
# Two full UTC days, aligned to midnight so a whole-day request is exactly rollup-servable.
DAY_A_START = 20_000 * MS_PER_DAY
DAY_B_START = DAY_A_START + MS_PER_DAY
DAY_A_RANGE = (DAY_A_START, DAY_A_START + MS_PER_DAY - 1)

# A value no raw aggregate in these tests can produce, so it reveals whether a rollup was consulted.
SENTINEL_COUNT = 987_654

SOURCE = AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="tester")

_COUNT = [MetricAggregation(aggregation_type=AggregationType.COUNT)]
_AVG = [MetricAggregation(aggregation_type=AggregationType.AVG)]
_P50 = [MetricAggregation(aggregation_type=AggregationType.PERCENTILE, percentile_value=50)]


def _day_of(timestamp_ms: int):
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).date()


def _new_trace(store, exp_id, timestamp_ms, duration_ms=100, state=TraceState.OK):
    trace_id = f"tr-{uuid.uuid4()}"
    store.start_trace(
        TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=timestamp_ms,
            execution_duration=duration_ms,
            state=state,
            tags={TraceTagKey.TRACE_NAME: "rollup-test"},
        )
    )
    return trace_id


def _add_feedback(store, trace_id, value=0.5, name="quality"):
    return store.create_assessment(
        Feedback(trace_id=trace_id, name=name, value=value, source=SOURCE)
    )


def _seed(store) -> str:
    """Two OK/ERROR traces on day A, plus one trace and feedback on day B, in a new experiment."""
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    t1 = _new_trace(store, exp_id, DAY_A_START + 5_000, duration_ms=100, state=TraceState.OK)
    _new_trace(store, exp_id, DAY_A_START + 6_000, duration_ms=300, state=TraceState.ERROR)
    t3 = _new_trace(store, exp_id, DAY_B_START + 7_000, duration_ms=500, state=TraceState.OK)
    _add_feedback(store, t1, value=0.4)
    _add_feedback(store, t3, value=0.9)
    # These read-routing tests seed rollup rows directly. Maintenance-owned invalidations created
    # by the source writes are cleared so each test can explicitly control coverage and queue state.
    with store.ManagedSessionMaker(read_only=False) as session:
        session.query(SqlTraceRollupRebuild).delete()
    return exp_id


def _set_enabled(monkeypatch, enabled: bool):
    monkeypatch.setenv(MLFLOW_SQL_TRACE_ROLLUPS_ENABLED.name, "true" if enabled else "false")


def _insert_trace_metric_rollup(
    store,
    exp_id,
    day_start_ms,
    metric_name,
    grouping_set,
    *,
    sample_count,
    sum_value=None,
    trace_status=None,
    p50_value=None,
):
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add(
            SqlTraceMetricDailyRollup(
                experiment_id=int(exp_id),
                rollup_day=_day_of(day_start_ms),
                metric_name=metric_name,
                grouping_set=grouping_set,
                trace_status=trace_status,
                sample_count=sample_count,
                sum_value=sum_value,
                p50_value=p50_value,
            )
        )
        session.commit()


def _insert_assessment_rollup(
    store, exp_id, day_start_ms, metric_name, grouping_set, *, sample_count, sum_value=None
):
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add(
            SqlAssessmentDailyRollup(
                experiment_id=int(exp_id),
                rollup_day=_day_of(day_start_ms),
                metric_name=metric_name,
                grouping_set=grouping_set,
                sample_count=sample_count,
                sum_value=sum_value,
            )
        )
        session.commit()


def _invalidate_day(store, family, exp_id, day_start_ms):
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add(
            SqlTraceRollupRebuild(
                experiment_id=int(exp_id),
                rollup_day=_day_of(day_start_ms),
                rollup_family=family.value,
            )
        )
        session.commit()


def _normalize(points):
    return sorted(
        (
            tuple(sorted((p.dimensions or {}).items())),
            tuple(sorted((k, round(float(v), 6)) for k, v in p.values.items())),
        )
        for p in points
    )


def _query(
    store,
    exp_id,
    view,
    metric,
    aggs,
    dims,
    *,
    start=DAY_A_RANGE[0],
    end=DAY_A_RANGE[1],
    time_interval=DAILY_INTERVAL_SECONDS,
    filters=None,
    experiment_ids=None,
):
    return _normalize(
        store.query_trace_metrics(
            experiment_ids=experiment_ids or [exp_id],
            view_type=view,
            metric_name=metric,
            aggregations=aggs,
            dimensions=dims,
            filters=filters,
            time_interval_seconds=time_interval,
            start_time_ms=start,
            end_time_ms=end,
        )
    )


def _raw_day_points(store, exp_id, view, metric, dims, day_start_ms):
    """Raw single-day aggregate (no bucketing) used to seed correct rollup sample counts."""
    return store.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=view,
        metric_name=metric,
        aggregations=_COUNT,
        dimensions=dims,
        start_time_ms=day_start_ms,
        end_time_ms=day_start_ms + MS_PER_DAY - 1,
    )


def _assert_enabled_equals_raw(
    store, monkeypatch, exp_id, view, metric, aggs, dims, **query_kwargs
):
    _set_enabled(monkeypatch, False)
    raw = _query(store, exp_id, view, metric, aggs, dims, **query_kwargs)
    _set_enabled(monkeypatch, True)
    fast = _query(store, exp_id, view, metric, aggs, dims, **query_kwargs)
    assert raw == fast


def test_covered_bucketed_day_is_served_from_rollup(store: SqlAlchemyStore, monkeypatch):
    exp_id = _seed(store)
    _insert_trace_metric_rollup(
        store,
        exp_id,
        DAY_A_START,
        TraceMetricKey.TRACE_COUNT,
        GroupingSet.GLOBAL.value,
        sample_count=SENTINEL_COUNT,
    )
    _set_enabled(monkeypatch, False)
    raw = _query(store, exp_id, MetricViewType.TRACES, TraceMetricKey.TRACE_COUNT, _COUNT, None)
    _set_enabled(monkeypatch, True)
    served = _query(store, exp_id, MetricViewType.TRACES, TraceMetricKey.TRACE_COUNT, _COUNT, None)

    # Same day bucket as raw, but the count comes from the rollup row rather than the raw scan.
    assert [dims for dims, _ in served] == [dims for dims, _ in raw]
    assert served == [(raw[0][0], (("COUNT", float(SENTINEL_COUNT)),))]


def test_incomplete_status_grouping_falls_back_to_raw(store: SqlAlchemyStore, monkeypatch):
    exp_id = _seed(store)
    # A status breakdown without its global publication row is incomplete and must stay raw.
    for status in ("OK", "ERROR"):
        _insert_trace_metric_rollup(
            store,
            exp_id,
            DAY_A_START,
            TraceMetricKey.TRACE_COUNT,
            GroupingSet.STATUS.value,
            sample_count=SENTINEL_COUNT,
            trace_status=status,
        )
    _assert_enabled_equals_raw(
        store,
        monkeypatch,
        exp_id,
        MetricViewType.TRACES,
        TraceMetricKey.TRACE_COUNT,
        _COUNT,
        [TraceMetricDimensionKey.TRACE_STATUS],
    )


def test_complete_status_grouping_is_served_from_rollup(store: SqlAlchemyStore, monkeypatch):
    exp_id = _seed(store)
    _insert_trace_metric_rollup(
        store,
        exp_id,
        DAY_A_START,
        TraceMetricKey.TRACE_COUNT,
        GroupingSet.GLOBAL.value,
        sample_count=SENTINEL_COUNT * 2,
    )
    for status in ("OK", "ERROR"):
        _insert_trace_metric_rollup(
            store,
            exp_id,
            DAY_A_START,
            TraceMetricKey.TRACE_COUNT,
            GroupingSet.STATUS.value,
            sample_count=SENTINEL_COUNT,
            trace_status=status,
        )

    _set_enabled(monkeypatch, True)
    served = _query(
        store,
        exp_id,
        MetricViewType.TRACES,
        TraceMetricKey.TRACE_COUNT,
        _COUNT,
        [TraceMetricDimensionKey.TRACE_STATUS],
    )

    counts_by_status = {
        dict(dimensions)[TraceMetricDimensionKey.TRACE_STATUS]: dict(values)["COUNT"]
        for dimensions, values in served
    }
    assert counts_by_status == {"OK": float(SENTINEL_COUNT), "ERROR": float(SENTINEL_COUNT)}


def test_duplicate_global_rollup_falls_back_to_raw(store: SqlAlchemyStore, monkeypatch):
    exp_id = _seed(store)
    for _ in range(2):
        _insert_trace_metric_rollup(
            store,
            exp_id,
            DAY_A_START,
            TraceMetricKey.TRACE_COUNT,
            GroupingSet.GLOBAL.value,
            sample_count=SENTINEL_COUNT,
        )
    _assert_enabled_equals_raw(
        store, monkeypatch, exp_id, MetricViewType.TRACES, TraceMetricKey.TRACE_COUNT, _COUNT, None
    )


def test_latency_avg_is_computed_from_rollup(store: SqlAlchemyStore, monkeypatch):
    exp_id = _seed(store)
    _insert_trace_metric_rollup(
        store,
        exp_id,
        DAY_A_START,
        TraceMetricKey.LATENCY,
        GroupingSet.GLOBAL.value,
        sample_count=4,
        sum_value=1000.0,
    )
    _set_enabled(monkeypatch, True)
    served = _query(store, exp_id, MetricViewType.TRACES, TraceMetricKey.LATENCY, _AVG, None)

    # AVG is derived from the stored sum and sample count: 1000 / 4.
    assert served == [(served[0][0], (("AVG", 250.0),))]


def test_assessment_value_avg_is_computed_from_rollup(store: SqlAlchemyStore, monkeypatch):
    exp_id = _seed(store)
    _insert_assessment_rollup(
        store,
        exp_id,
        DAY_A_START,
        AssessmentMetricKey.ASSESSMENT_VALUE,
        GroupingSet.GLOBAL.value,
        sample_count=4,
        sum_value=2.0,
    )
    _set_enabled(monkeypatch, True)
    served = _query(
        store, exp_id, MetricViewType.ASSESSMENTS, AssessmentMetricKey.ASSESSMENT_VALUE, _AVG, None
    )

    # AVG is derived from the stored sum and sample count: 2.0 / 4.
    assert served == [(served[0][0], (("AVG", 0.5),))]


def test_latency_avg_falls_back_when_rollup_sum_is_null(store: SqlAlchemyStore, monkeypatch):
    exp_id = _seed(store)
    # A covered day whose AVG backing column is null must demote to the raw path, not divide by the
    # sample count and raise.
    _insert_trace_metric_rollup(
        store,
        exp_id,
        DAY_A_START,
        TraceMetricKey.LATENCY,
        GroupingSet.GLOBAL.value,
        sample_count=4,
        sum_value=None,
    )
    _assert_enabled_equals_raw(
        store, monkeypatch, exp_id, MetricViewType.TRACES, TraceMetricKey.LATENCY, _AVG, None
    )


@pytest.mark.parametrize(
    ("view", "metric"),
    [
        (MetricViewType.TRACES, TraceMetricKey.TRACE_COUNT),
        (MetricViewType.ASSESSMENTS, AssessmentMetricKey.ASSESSMENT_COUNT),
    ],
    ids=["trace_count global", "assessment_count global"],
)
def test_enabled_matches_raw_for_correct_count_rollups(
    store: SqlAlchemyStore, monkeypatch, view, metric
):
    exp_id = _seed(store)
    for day_start in (DAY_A_START, DAY_B_START):
        for point in _raw_day_points(store, exp_id, view, metric, None, day_start):
            sample_count = int(point.values["COUNT"])
            if view == MetricViewType.ASSESSMENTS:
                _insert_assessment_rollup(
                    store,
                    exp_id,
                    day_start,
                    metric,
                    GroupingSet.GLOBAL.value,
                    sample_count=sample_count,
                )
            else:
                _insert_trace_metric_rollup(
                    store,
                    exp_id,
                    day_start,
                    metric,
                    GroupingSet.GLOBAL.value,
                    sample_count=sample_count,
                )

    _assert_enabled_equals_raw(
        store,
        monkeypatch,
        exp_id,
        view,
        metric,
        _COUNT,
        None,
        start=DAY_A_START,
        end=DAY_B_START + MS_PER_DAY - 1,
    )


def test_partial_edges_query_raw_around_covered_day(store: SqlAlchemyStore, monkeypatch):
    exp_id = _seed(store)
    _insert_trace_metric_rollup(
        store,
        exp_id,
        DAY_A_START,
        TraceMetricKey.TRACE_COUNT,
        GroupingSet.GLOBAL.value,
        sample_count=SENTINEL_COUNT,
    )
    # Whole day A is covered; day B is only partially in range, so its edge stays on the raw path.
    end = DAY_B_START + 10_000
    _set_enabled(monkeypatch, False)
    raw = _query(
        store, exp_id, MetricViewType.TRACES, TraceMetricKey.TRACE_COUNT, _COUNT, None, end=end
    )
    _set_enabled(monkeypatch, True)
    merged = _query(
        store, exp_id, MetricViewType.TRACES, TraceMetricKey.TRACE_COUNT, _COUNT, None, end=end
    )

    raw_by_bucket = {dims: dict(vals)["COUNT"] for dims, vals in raw}
    merged_by_bucket = {dims: dict(vals)["COUNT"] for dims, vals in merged}
    assert set(merged_by_bucket) == set(raw_by_bucket)
    day_a_bucket = min(raw_by_bucket)
    day_b_bucket = max(raw_by_bucket)
    assert merged_by_bucket[day_a_bucket] == float(SENTINEL_COUNT)
    assert merged_by_bucket[day_b_bucket] == raw_by_bucket[day_b_bucket]


def test_merged_result_matches_raw_with_correct_rollup_values(store: SqlAlchemyStore, monkeypatch):
    exp_id = _seed(store)
    # Seed day A's real count, then request a range covering day A fully and only part of day B so a
    # rollup-served day and a raw edge must merge into a result identical to the raw path.
    for point in _raw_day_points(
        store, exp_id, MetricViewType.TRACES, TraceMetricKey.TRACE_COUNT, None, DAY_A_START
    ):
        _insert_trace_metric_rollup(
            store,
            exp_id,
            DAY_A_START,
            TraceMetricKey.TRACE_COUNT,
            GroupingSet.GLOBAL.value,
            sample_count=int(point.values["COUNT"]),
        )
    _assert_enabled_equals_raw(
        store,
        monkeypatch,
        exp_id,
        MetricViewType.TRACES,
        TraceMetricKey.TRACE_COUNT,
        _COUNT,
        None,
        end=DAY_B_START + 10_000,
    )


def test_disabled_ignores_rollups(store: SqlAlchemyStore, monkeypatch):
    exp_id = _seed(store)
    _insert_trace_metric_rollup(
        store,
        exp_id,
        DAY_A_START,
        TraceMetricKey.TRACE_COUNT,
        GroupingSet.GLOBAL.value,
        sample_count=SENTINEL_COUNT,
    )
    _set_enabled(monkeypatch, False)
    result = _query(store, exp_id, MetricViewType.TRACES, TraceMetricKey.TRACE_COUNT, _COUNT, None)

    assert dict(result[0][1])["COUNT"] == 2.0


def test_invalidated_day_falls_back_to_raw(store: SqlAlchemyStore, monkeypatch):
    exp_id = _seed(store)
    _insert_trace_metric_rollup(
        store,
        exp_id,
        DAY_A_START,
        TraceMetricKey.TRACE_COUNT,
        GroupingSet.GLOBAL.value,
        sample_count=SENTINEL_COUNT,
    )
    _invalidate_day(store, RollupFamily.TRACE_METRIC, exp_id, DAY_A_START)
    _assert_enabled_equals_raw(
        store, monkeypatch, exp_id, MetricViewType.TRACES, TraceMetricKey.TRACE_COUNT, _COUNT, None
    )


def test_uncovered_day_falls_back_to_raw(store: SqlAlchemyStore, monkeypatch):
    exp_id = _seed(store)
    # A rollup exists for day B, but the request targets day A, which has none.
    _insert_trace_metric_rollup(
        store,
        exp_id,
        DAY_B_START,
        TraceMetricKey.TRACE_COUNT,
        GroupingSet.GLOBAL.value,
        sample_count=SENTINEL_COUNT,
    )
    _assert_enabled_equals_raw(
        store, monkeypatch, exp_id, MetricViewType.TRACES, TraceMetricKey.TRACE_COUNT, _COUNT, None
    )


def test_partial_day_only_range_falls_back_to_raw(store: SqlAlchemyStore, monkeypatch):
    exp_id = _seed(store)
    _insert_trace_metric_rollup(
        store,
        exp_id,
        DAY_A_START,
        TraceMetricKey.TRACE_COUNT,
        GroupingSet.GLOBAL.value,
        sample_count=SENTINEL_COUNT,
    )
    # A range that never spans a full UTC day cannot use whole-day rollups.
    _assert_enabled_equals_raw(
        store,
        monkeypatch,
        exp_id,
        MetricViewType.TRACES,
        TraceMetricKey.TRACE_COUNT,
        _COUNT,
        None,
        start=DAY_A_START + 100,
        end=DAY_A_START + MS_PER_DAY - 200,
    )


def test_non_daily_interval_falls_back_to_raw(store: SqlAlchemyStore, monkeypatch):
    exp_id = _seed(store)
    _insert_trace_metric_rollup(
        store,
        exp_id,
        DAY_A_START,
        TraceMetricKey.TRACE_COUNT,
        GroupingSet.GLOBAL.value,
        sample_count=SENTINEL_COUNT,
    )
    _assert_enabled_equals_raw(
        store,
        monkeypatch,
        exp_id,
        MetricViewType.TRACES,
        TraceMetricKey.TRACE_COUNT,
        _COUNT,
        None,
        time_interval=3_600,
    )


def test_filtered_request_falls_back_to_raw(store: SqlAlchemyStore, monkeypatch):
    exp_id = _seed(store)
    _insert_trace_metric_rollup(
        store,
        exp_id,
        DAY_A_START,
        TraceMetricKey.TRACE_COUNT,
        GroupingSet.GLOBAL.value,
        sample_count=SENTINEL_COUNT,
    )
    _assert_enabled_equals_raw(
        store,
        monkeypatch,
        exp_id,
        MetricViewType.TRACES,
        TraceMetricKey.TRACE_COUNT,
        _COUNT,
        None,
        filters=["trace.status = 'OK'"],
    )


def test_high_cardinality_grouping_falls_back_to_raw(store: SqlAlchemyStore, monkeypatch):
    exp_id = _seed(store)
    _insert_trace_metric_rollup(
        store,
        exp_id,
        DAY_A_START,
        TraceMetricKey.TRACE_COUNT,
        GroupingSet.GLOBAL.value,
        sample_count=SENTINEL_COUNT,
    )
    # trace_name is not a materialized grouping set, so the request stays raw.
    _assert_enabled_equals_raw(
        store,
        monkeypatch,
        exp_id,
        MetricViewType.TRACES,
        TraceMetricKey.TRACE_COUNT,
        _COUNT,
        [TraceMetricDimensionKey.TRACE_NAME],
    )


def test_multi_experiment_request_falls_back_to_raw(store: SqlAlchemyStore, monkeypatch):
    exp_id = _seed(store)
    other_exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    _insert_trace_metric_rollup(
        store,
        exp_id,
        DAY_A_START,
        TraceMetricKey.TRACE_COUNT,
        GroupingSet.GLOBAL.value,
        sample_count=SENTINEL_COUNT,
    )
    _assert_enabled_equals_raw(
        store,
        monkeypatch,
        exp_id,
        MetricViewType.TRACES,
        TraceMetricKey.TRACE_COUNT,
        _COUNT,
        None,
        experiment_ids=[exp_id, other_exp_id],
    )


def test_rollup_for_other_experiment_is_ignored(store: SqlAlchemyStore, monkeypatch):
    exp_id = _seed(store)
    other_exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    # Rollup rows are keyed by experiment id, so another experiment's row must never satisfy this
    # experiment's request. This is what keeps rollups workspace-isolated: the caller only ever
    # routes accessible experiment ids into the planner.
    _insert_trace_metric_rollup(
        store,
        other_exp_id,
        DAY_A_START,
        TraceMetricKey.TRACE_COUNT,
        GroupingSet.GLOBAL.value,
        sample_count=SENTINEL_COUNT,
    )
    _assert_enabled_equals_raw(
        store, monkeypatch, exp_id, MetricViewType.TRACES, TraceMetricKey.TRACE_COUNT, _COUNT, None
    )


def test_percentiles_fall_back_to_raw_off_postgres(store: SqlAlchemyStore, monkeypatch):
    if store.engine.dialect.name == "postgresql":
        pytest.skip("percentiles are served from rollups on postgres")
    exp_id = _seed(store)
    _insert_trace_metric_rollup(
        store,
        exp_id,
        DAY_A_START,
        TraceMetricKey.LATENCY,
        GroupingSet.GLOBAL.value,
        sample_count=4,
        p50_value=float(SENTINEL_COUNT),
    )
    _assert_enabled_equals_raw(
        store, monkeypatch, exp_id, MetricViewType.TRACES, TraceMetricKey.LATENCY, _P50, None
    )
