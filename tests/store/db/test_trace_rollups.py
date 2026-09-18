import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Barrier, Event, get_ident
from unittest.mock import Mock

import pytest
from click.testing import CliRunner
from sqlalchemy import event
from sqlalchemy.dialects import mssql
from sqlalchemy.orm import Session

import mlflow.db
import mlflow.store.tracking.sqlalchemy_store as sqlalchemy_store_module
from mlflow.entities import AssessmentSource, AssessmentSourceType, Feedback, trace_location
from mlflow.entities.assessment import FeedbackValue
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_metrics import AggregationType, MetricAggregation, MetricViewType
from mlflow.entities.trace_state import TraceState
from mlflow.entities.trace_status import TraceStatus
from mlflow.environment_variables import MLFLOW_SQL_TRACE_ROLLUPS_ENABLED
from mlflow.store.db import trace_rollups
from mlflow.store.db.trace_rollups import (
    ROLLUP_ELIGIBILITY_LAG_MS,
    RollupBuildStats,
    RollupDeleteStats,
    delete_sql_trace_rollups,
    run_sql_trace_rollups,
    sql_trace_rollup_rows_exist,
)
from mlflow.store.tracking.dbmodels.models import (
    SqlAssessmentDailyRollup,
    SqlAssessments,
    SqlSpan,
    SqlSpanCostDailyRollup,
    SqlTraceInfo,
    SqlTraceMetricDailyRollup,
    SqlTraceRollupRebuild,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.tracking.utils import sql_trace_rollups as sql_trace_rollup_utils
from mlflow.store.tracking.utils.sql_trace_rollups import (
    RollupFamily,
    _lock_rebuild_entry_query,
    enqueue_rollup_rebuild_partitions,
    enqueue_rollup_rebuilds,
)
from mlflow.tracing.constant import (
    AssessmentMetadataKey,
    AssessmentMetricKey,
    SpanAttributeKey,
    SpanMetricKey,
    TraceMetricKey,
    TraceTagKey,
)

from tests.store.tracking.sqlalchemy_store.conftest import create_test_span

pytestmark = pytest.mark.notrackingurimock

MS_PER_DAY = 86_400_000
# A fixed instant on an arbitrary past UTC day, safely more than 24 hours before FUTURE_NOW_MS.
DAY_A_MS = 20_000 * MS_PER_DAY + 5_000
DAY_B_MS = DAY_A_MS + MS_PER_DAY
# Far enough in the future that every seeded day clears the 24-hour inactivity lag.
FUTURE_NOW_MS = 30_000 * MS_PER_DAY

SOURCE = AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="tester")


@pytest.fixture
def store(tmp_path: Path, db_uri: str) -> SqlAlchemyStore:
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir()
    return SqlAlchemyStore(db_uri, artifact_uri.as_uri())


@pytest.fixture(autouse=True)
def enable_rollups(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(MLFLOW_SQL_TRACE_ROLLUPS_ENABLED.name, "true")


def _day_of(timestamp_ms: int):
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).date()


def _new_trace(store, exp_id, timestamp_ms, duration_ms=100, state=TraceStatus.OK):
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


def _count(store, model, **filters):
    with store.ManagedSessionMaker() as session:
        query = session.query(model)
        if filters:
            query = query.filter_by(**filters)
        return query.count()


def _enqueue_entry(store, family, experiment_id, timestamp_ms):
    with store.ManagedSessionMaker(read_only=False) as session:
        enqueue_rollup_rebuilds(session, family, int(experiment_id), [timestamp_ms])


def _query_daily_trace_count(store: SqlAlchemyStore, experiment_id: str):
    day_start_ms = DAY_A_MS - 5_000
    return store.query_trace_metrics(
        experiment_ids=[experiment_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.TRACE_COUNT,
        aggregations=[MetricAggregation(AggregationType.COUNT)],
        time_interval_seconds=MS_PER_DAY // 1_000,
        start_time_ms=day_start_ms,
        end_time_ms=day_start_ms + MS_PER_DAY - 1,
    )


def _assert_rollup_matches_raw(store: SqlAlchemyStore, monkeypatch, experiment_id: str):
    monkeypatch.setenv(MLFLOW_SQL_TRACE_ROLLUPS_ENABLED.name, "false")
    raw = _query_daily_trace_count(store, experiment_id)
    monkeypatch.setenv(MLFLOW_SQL_TRACE_ROLLUPS_ENABLED.name, "true")
    rollup = _query_daily_trace_count(store, experiment_id)
    assert rollup == raw


def test_build_populates_trace_metric_and_assessment_rollups(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    trace_id = _new_trace(store, exp_id, DAY_A_MS, duration_ms=100)
    _new_trace(store, exp_id, DAY_A_MS + 1000, duration_ms=300, state=TraceStatus.ERROR)
    _add_feedback(store, trace_id, value=0.8)

    stats = run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)

    assert isinstance(stats, RollupBuildStats)
    assert stats.trace_metric.built == 1
    assert stats.assessment.built == 1
    assert stats.trace_metric.deferred == 0
    assert _count(store, SqlTraceMetricDailyRollup) > 0
    assert _count(store, SqlAssessmentDailyRollup) > 0

    with store.ManagedSessionMaker() as session:
        # trace_count global sample_count equals the number of traces that day.
        trace_count = (
            session
            .query(SqlTraceMetricDailyRollup)
            .filter_by(
                experiment_id=int(exp_id),
                metric_name=TraceMetricKey.TRACE_COUNT,
                grouping_set="global",
            )
            .one()
        )
        assert trace_count.sample_count == 2
        # assessment_count global sample_count equals the number of assessments that day.
        assessment_count = (
            session
            .query(SqlAssessmentDailyRollup)
            .filter_by(
                experiment_id=int(exp_id),
                metric_name=AssessmentMetricKey.ASSESSMENT_COUNT,
                grouping_set="global",
            )
            .one()
        )
        assert assessment_count.sample_count == 1


def test_build_is_idempotent(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    _new_trace(store, exp_id, DAY_A_MS)

    run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)
    first = _count(store, SqlTraceMetricDailyRollup)

    second_stats = run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)

    # Nothing new is eligible and the queue is empty, so the rerun is a no-op.
    assert second_stats.trace_metric.built == 0
    assert second_stats.span_cost.built == 0
    assert second_stats.assessment.built == 0
    assert _count(store, SqlTraceMetricDailyRollup) == first


def test_deprecated_start_trace_v2_enqueues_and_rebuilds_trace_metric_rollup(
    store: SqlAlchemyStore, monkeypatch: pytest.MonkeyPatch
):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    _new_trace(store, exp_id, DAY_A_MS)
    run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)

    trace = store.deprecated_start_trace_v2(exp_id, DAY_A_MS + 1_000, {}, {})

    assert (
        _count(
            store,
            SqlTraceRollupRebuild,
            experiment_id=int(exp_id),
            rollup_day=_day_of(DAY_A_MS),
            rollup_family=RollupFamily.TRACE_METRIC.value,
        )
        == 1
    )

    # Complete the V2 trace so the partition is eligible, then verify the published rollup is
    # indistinguishable from the authoritative raw query.
    store.deprecated_end_trace_v2(
        trace.request_id,
        DAY_A_MS + 1_100,
        TraceStatus.OK,
        {},
        {},
    )
    run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)
    _assert_rollup_matches_raw(store, monkeypatch, exp_id)


def test_deprecated_end_trace_v2_enqueues_and_rebuilds_trace_metric_rollup(
    store: SqlAlchemyStore, monkeypatch: pytest.MonkeyPatch
):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    trace = store.deprecated_start_trace_v2(exp_id, DAY_A_MS, {}, {})
    store.deprecated_end_trace_v2(
        trace.request_id,
        DAY_A_MS + 100,
        TraceStatus.OK,
        {},
        {},
    )
    run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)
    assert _count(store, SqlTraceRollupRebuild) == 0

    store.deprecated_end_trace_v2(
        trace.request_id,
        DAY_A_MS + 250,
        TraceStatus.ERROR,
        {},
        {},
    )

    assert (
        _count(
            store,
            SqlTraceRollupRebuild,
            experiment_id=int(exp_id),
            rollup_day=_day_of(DAY_A_MS),
            rollup_family=RollupFamily.TRACE_METRIC.value,
        )
        == 1
    )

    run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)
    _assert_rollup_matches_raw(store, monkeypatch, exp_id)


def test_build_populates_span_cost_rollups_for_supported_groupings(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    trace_id = _new_trace(store, exp_id, DAY_A_MS)
    store.log_spans(
        exp_id,
        [
            create_test_span(
                trace_id=trace_id,
                span_id=1,
                start_ns=DAY_A_MS * 1_000_000,
                end_ns=(DAY_A_MS + 10) * 1_000_000,
                attributes={
                    SpanAttributeKey.LLM_COST: {
                        "input_cost": 0.2,
                        "output_cost": 0.3,
                        "total_cost": 0.5,
                    },
                    SpanAttributeKey.MODEL: "gpt-test",
                    SpanAttributeKey.MODEL_PROVIDER: "openai",
                },
            ),
            create_test_span(
                trace_id=trace_id,
                span_id=2,
                start_ns=(DAY_A_MS + 20) * 1_000_000,
                end_ns=(DAY_A_MS + 30) * 1_000_000,
                attributes={
                    SpanAttributeKey.LLM_COST: {
                        "input_cost": 0.4,
                        "output_cost": 0.6,
                        "total_cost": 1.0,
                    },
                    SpanAttributeKey.MODEL: "gpt-test",
                },
            ),
        ],
    )

    stats = run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)

    assert stats.span_cost.built == 1
    with store.ManagedSessionMaker() as session:
        total_cost_global = (
            session
            .query(SqlSpanCostDailyRollup)
            .filter_by(
                experiment_id=int(exp_id),
                rollup_day=_day_of(DAY_A_MS),
                metric_name=SpanMetricKey.TOTAL_COST,
                grouping_set="global",
            )
            .one()
        )
        assert total_cost_global.sample_count == 2
        assert total_cost_global.sum_value == pytest.approx(1.5)
        assert total_cost_global.min_value == pytest.approx(0.5)
        assert total_cost_global.max_value == pytest.approx(1.0)

        model = (
            session
            .query(SqlSpanCostDailyRollup)
            .filter_by(metric_name=SpanMetricKey.TOTAL_COST, grouping_set="model")
            .one()
        )
        assert (model.model_name, model.model_provider, model.sample_count) == (
            "gpt-test",
            None,
            2,
        )

        provider = (
            session
            .query(SqlSpanCostDailyRollup)
            .filter_by(metric_name=SpanMetricKey.TOTAL_COST, grouping_set="provider")
            .one()
        )
        assert (provider.model_name, provider.model_provider, provider.sample_count) == (
            None,
            "openai",
            1,
        )

        model_provider = (
            session
            .query(SqlSpanCostDailyRollup)
            .filter_by(metric_name=SpanMetricKey.TOTAL_COST, grouping_set="model_provider")
            .one()
        )
        assert (
            model_provider.model_name,
            model_provider.model_provider,
            model_provider.sample_count,
        ) == ("gpt-test", "openai", 1)


def test_recent_day_is_not_eligible(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    _new_trace(store, exp_id, DAY_A_MS)

    # now is only 1 second after the trace, well inside the 24-hour inactivity lag.
    stats = run_sql_trace_rollups(store.engine, now_ms=DAY_A_MS + 1000)

    assert stats.trace_metric.built == 0
    assert stats.assessment.built == 0
    assert _count(store, SqlTraceMetricDailyRollup) == 0


def test_empty_current_day_remains_queued(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    current_day_ms = FUTURE_NOW_MS
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add(
            SqlTraceRollupRebuild(
                experiment_id=int(exp_id),
                rollup_day=_day_of(current_day_ms),
                rollup_family=RollupFamily.TRACE_METRIC.value,
            )
        )

    stats = run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS + 1000)

    assert stats.trace_metric.deferred == 1
    assert _count(store, SqlTraceRollupRebuild, rollup_family=RollupFamily.TRACE_METRIC.value) == 1


def test_day_becomes_eligible_exactly_after_lag(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    _new_trace(store, exp_id, DAY_A_MS)
    day_end_ms = (DAY_A_MS // MS_PER_DAY + 1) * MS_PER_DAY

    # The complete UTC day, not merely its first trace, must be beyond the lag.
    before = run_sql_trace_rollups(store.engine, now_ms=day_end_ms + ROLLUP_ELIGIBILITY_LAG_MS - 1)
    assert before.trace_metric.built == 0

    at = run_sql_trace_rollups(store.engine, now_ms=day_end_ms + ROLLUP_ELIGIBILITY_LAG_MS)
    assert at.trace_metric.built == 1


def test_backdated_write_after_publication_queues_and_serves_raw(
    store: SqlAlchemyStore, monkeypatch
):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    _new_trace(store, exp_id, DAY_A_MS)
    day_end_ms = (DAY_A_MS // MS_PER_DAY + 1) * MS_PER_DAY
    eligible_now_ms = day_end_ms + ROLLUP_ELIGIBILITY_LAG_MS
    run_sql_trace_rollups(store.engine, now_ms=eligible_now_ms)

    monkeypatch.setattr(sql_trace_rollup_utils, "get_current_time_millis", lambda: eligible_now_ms)
    _new_trace(store, exp_id, day_end_ms - 1)

    assert (
        _count(
            store,
            SqlTraceRollupRebuild,
            rollup_family=RollupFamily.TRACE_METRIC.value,
        )
        == 1
    )
    _assert_rollup_matches_raw(store, monkeypatch, exp_id)


def test_open_span_defers_queued_partition(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    trace_id = _new_trace(store, exp_id, DAY_A_MS)
    # An unfinished span (no end time) keeps the trace active, so its day must never be built.
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add(
            SqlSpan(
                trace_id=trace_id,
                experiment_id=int(exp_id),
                span_id="span-open",
                status="UNSET",
                start_time_unix_nano=DAY_A_MS * 1_000_000,
                end_time_unix_nano=None,
                content="{}",
            )
        )
        session.commit()
    _enqueue_entry(store, RollupFamily.TRACE_METRIC, exp_id, DAY_A_MS)
    _enqueue_entry(store, RollupFamily.SPAN_COST, exp_id, DAY_A_MS)

    stats = run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)

    assert stats.trace_metric.deferred == 1
    assert stats.trace_metric.built == 0
    assert stats.span_cost.deferred == 1
    assert stats.span_cost.built == 0
    # The queue entry survives so a later run retries once the span closes.
    assert _count(store, SqlTraceRollupRebuild, rollup_family=RollupFamily.TRACE_METRIC.value) == 1
    assert _count(store, SqlTraceRollupRebuild, rollup_family=RollupFamily.SPAN_COST.value) == 1
    assert _count(store, SqlTraceMetricDailyRollup) == 0


def test_spanless_in_progress_trace_defers_partition(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    store.deprecated_start_trace_v2(exp_id, DAY_A_MS, {}, {})

    stats = run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)

    assert stats.trace_metric.deferred == 1
    assert _count(store, SqlTraceMetricDailyRollup) == 0
    assert _count(store, SqlTraceRollupRebuild, rollup_family=RollupFamily.TRACE_METRIC.value) == 1


@pytest.mark.parametrize(
    "status",
    [TraceStatus.UNSPECIFIED.value, TraceState.STATE_UNSPECIFIED.value],
)
def test_spanless_unspecified_trace_is_terminal(store: SqlAlchemyStore, status: str):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add_all([
            SqlTraceInfo(
                request_id=f"tr-{uuid.uuid4()}",
                experiment_id=int(exp_id),
                timestamp_ms=DAY_A_MS,
                execution_time_ms=100,
                status=status,
            ),
            SqlTraceRollupRebuild(
                experiment_id=int(exp_id),
                rollup_day=_day_of(DAY_A_MS),
                rollup_family=RollupFamily.TRACE_METRIC.value,
            ),
        ])

    stats = run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)

    assert stats.trace_metric.built == 1
    assert stats.trace_metric.deferred == 0
    assert _count(store, SqlTraceMetricDailyRollup) > 0
    assert _count(store, SqlTraceRollupRebuild, rollup_family=RollupFamily.TRACE_METRIC.value) == 0


def test_open_span_on_another_day_defers_span_cost_partition(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    trace_id = _new_trace(store, exp_id, DAY_A_MS)
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add_all([
            SqlSpan(
                trace_id=trace_id,
                experiment_id=int(exp_id),
                span_id="span-day-a",
                status="OK",
                start_time_unix_nano=DAY_A_MS * 1_000_000,
                end_time_unix_nano=(DAY_A_MS + 10) * 1_000_000,
                total_cost=0.5,
                content="{}",
            ),
            SqlSpan(
                trace_id=trace_id,
                experiment_id=int(exp_id),
                span_id="span-day-b-open",
                status="UNSET",
                start_time_unix_nano=DAY_B_MS * 1_000_000,
                end_time_unix_nano=None,
                content="{}",
            ),
        ])
    _enqueue_entry(store, RollupFamily.SPAN_COST, exp_id, DAY_A_MS)

    stats = run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)

    assert stats.span_cost.deferred == 1
    assert _count(store, SqlSpanCostDailyRollup) == 0
    assert _count(store, SqlTraceRollupRebuild, rollup_family=RollupFamily.SPAN_COST.value) == 1


def test_emptied_partition_removes_rollups_and_queue_entry(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    trace_id = _new_trace(store, exp_id, DAY_A_MS)
    run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)
    assert _count(store, SqlTraceMetricDailyRollup) > 0

    # Delete the source rows out from under the rollups, then enqueue the day.
    with store.ManagedSessionMaker(read_only=False) as session:
        session.query(SqlTraceInfo).filter_by(request_id=trace_id).delete()
        session.commit()
    _enqueue_entry(store, RollupFamily.TRACE_METRIC, exp_id, DAY_A_MS)

    stats = run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)

    assert stats.trace_metric.emptied == 1
    assert _count(store, SqlTraceMetricDailyRollup) == 0
    assert _count(store, SqlTraceRollupRebuild, rollup_family=RollupFamily.TRACE_METRIC.value) == 0


def test_queue_entry_is_drained_on_rebuild(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    _new_trace(store, exp_id, DAY_A_MS)
    _enqueue_entry(store, RollupFamily.TRACE_METRIC, exp_id, DAY_A_MS)
    assert _count(store, SqlTraceRollupRebuild, rollup_family=RollupFamily.TRACE_METRIC.value) == 1

    run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)

    assert _count(store, SqlTraceRollupRebuild) == 0
    assert _count(store, SqlTraceMetricDailyRollup) > 0


def test_failed_rebuild_keeps_previous_rollup_and_queue_entry(
    store: SqlAlchemyStore, monkeypatch: pytest.MonkeyPatch
):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    _new_trace(store, exp_id, DAY_A_MS)
    run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)
    previous_rollup_count = _count(store, SqlTraceMetricDailyRollup)

    _new_trace(store, exp_id, DAY_A_MS + 1000)
    monkeypatch.setattr(
        trace_rollups,
        "_aggregate",
        Mock(side_effect=RuntimeError("aggregation failed")),
    )

    stats = run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)

    assert stats.trace_metric.failed == 1
    assert _count(store, SqlTraceMetricDailyRollup) == previous_rollup_count
    assert _count(store, SqlTraceRollupRebuild, rollup_family=RollupFamily.TRACE_METRIC.value) == 1


def test_source_writes_do_not_enqueue_while_rollups_are_disabled(
    store: SqlAlchemyStore, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(MLFLOW_SQL_TRACE_ROLLUPS_ENABLED.name, "false")
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")

    trace_id = _new_trace(store, exp_id, DAY_A_MS)
    store.log_spans(
        exp_id,
        [
            create_test_span(
                trace_id=trace_id,
                start_ns=DAY_A_MS * 1_000_000,
                end_ns=(DAY_A_MS + 10) * 1_000_000,
            )
        ],
    )

    assert _count(store, SqlTraceRollupRebuild) == 0


def test_rebuild_enqueue_requires_an_eligible_timestamp(
    store: SqlAlchemyStore, monkeypatch: pytest.MonkeyPatch
):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    monkeypatch.setattr(
        sql_trace_rollup_utils,
        "get_current_time_millis",
        lambda: DAY_A_MS + ROLLUP_ELIGIBILITY_LAG_MS - 1,
    )

    _enqueue_entry(store, RollupFamily.TRACE_METRIC, exp_id, DAY_A_MS)
    assert _count(store, SqlTraceRollupRebuild) == 0

    monkeypatch.setattr(
        sql_trace_rollup_utils,
        "get_current_time_millis",
        lambda: DAY_A_MS + ROLLUP_ELIGIBILITY_LAG_MS,
    )
    _enqueue_entry(store, RollupFamily.TRACE_METRIC, exp_id, DAY_A_MS)
    assert _count(store, SqlTraceRollupRebuild) == 1


def test_late_span_write_enqueues_trace_partition(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    trace_id = _new_trace(store, exp_id, DAY_A_MS)
    run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)
    assert _count(store, SqlTraceRollupRebuild) == 0

    store.log_spans(
        exp_id,
        [
            create_test_span(
                trace_id=trace_id,
                start_ns=DAY_B_MS * 1_000_000,
                end_ns=(DAY_B_MS + 50) * 1_000_000,
            )
        ],
    )

    assert _count(store, SqlTraceRollupRebuild, rollup_family=RollupFamily.TRACE_METRIC.value) == 1
    with store.ManagedSessionMaker() as session:
        span_cost_entry = (
            session
            .query(SqlTraceRollupRebuild)
            .filter_by(rollup_family=RollupFamily.SPAN_COST.value)
            .one()
        )
        assert span_cost_entry.rollup_day == _day_of(DAY_B_MS)


def test_span_upsert_invalidates_old_and_new_cost_partitions(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    trace_id = _new_trace(store, exp_id, DAY_A_MS)
    store.log_spans(
        exp_id,
        [
            create_test_span(
                trace_id=trace_id,
                span_id=1,
                start_ns=DAY_A_MS * 1_000_000,
                end_ns=(DAY_A_MS + 50) * 1_000_000,
                attributes={SpanAttributeKey.LLM_COST: {"total_cost": 0.5}},
            )
        ],
    )
    run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)
    assert _count(store, SqlSpanCostDailyRollup) > 0

    # Re-sending the same span moves it to day B and clears its cost. Both the old materialized
    # partition and its new source partition must be queued, then emptied safely on rebuild.
    store.log_spans(
        exp_id,
        [
            create_test_span(
                trace_id=trace_id,
                span_id=1,
                start_ns=DAY_B_MS * 1_000_000,
                end_ns=(DAY_B_MS + 50) * 1_000_000,
            )
        ],
    )
    with store.ManagedSessionMaker() as session:
        queued_days = {
            row.rollup_day
            for row in session.query(SqlTraceRollupRebuild).filter_by(
                rollup_family=RollupFamily.SPAN_COST.value
            )
        }
    assert queued_days == {_day_of(DAY_A_MS), _day_of(DAY_B_MS)}

    stats = run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)

    assert stats.span_cost.emptied == 2
    assert _count(store, SqlSpanCostDailyRollup) == 0
    assert _count(store, SqlTraceRollupRebuild, rollup_family=RollupFamily.SPAN_COST.value) == 0


def test_trace_move_enqueues_old_and_new_partitions(store: SqlAlchemyStore):
    old_exp_id = store.create_experiment(f"old-{uuid.uuid4()}")
    new_exp_id = store.create_experiment(f"new-{uuid.uuid4()}")
    trace_id = _new_trace(store, old_exp_id, DAY_A_MS)
    store.log_spans(
        old_exp_id,
        [
            create_test_span(
                trace_id=trace_id,
                start_ns=DAY_A_MS * 1_000_000,
                end_ns=(DAY_A_MS + 50) * 1_000_000,
                attributes={SpanAttributeKey.LLM_COST: {"total_cost": 0.5}},
            )
        ],
    )
    run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)
    assert _count(store, SqlTraceRollupRebuild) == 0

    store.start_trace(
        TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(new_exp_id),
            request_time=DAY_B_MS,
            execution_duration=250,
            state=TraceStatus.OK,
            tags={TraceTagKey.TRACE_NAME: "moved-trace"},
        )
    )

    with store.ManagedSessionMaker() as session:
        queued = {
            (row.experiment_id, row.rollup_day, row.rollup_family)
            for row in session.query(SqlTraceRollupRebuild)
        }
    assert queued == {
        (int(old_exp_id), _day_of(DAY_A_MS), RollupFamily.TRACE_METRIC.value),
        (int(old_exp_id), _day_of(DAY_A_MS), RollupFamily.ASSESSMENT.value),
        (int(new_exp_id), _day_of(DAY_B_MS), RollupFamily.TRACE_METRIC.value),
        (int(new_exp_id), _day_of(DAY_B_MS), RollupFamily.ASSESSMENT.value),
        (int(old_exp_id), _day_of(DAY_A_MS), RollupFamily.SPAN_COST.value),
        (int(new_exp_id), _day_of(DAY_A_MS), RollupFamily.SPAN_COST.value),
    }


def test_trace_delete_enqueues_span_cost_partition(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    trace_id = _new_trace(store, exp_id, DAY_A_MS)
    store.log_spans(
        exp_id,
        [
            create_test_span(
                trace_id=trace_id,
                start_ns=DAY_B_MS * 1_000_000,
                end_ns=(DAY_B_MS + 50) * 1_000_000,
                attributes={SpanAttributeKey.LLM_COST: {"total_cost": 0.5}},
            )
        ],
    )
    run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)
    assert _count(store, SqlTraceRollupRebuild) == 0

    store.delete_traces(exp_id, trace_ids=[trace_id])

    with store.ManagedSessionMaker() as session:
        span_cost_entry = (
            session
            .query(SqlTraceRollupRebuild)
            .filter_by(rollup_family=RollupFamily.SPAN_COST.value)
            .one()
        )
        assert span_cost_entry.rollup_day == _day_of(DAY_B_MS)


@pytest.mark.parametrize("mutation", ["create", "update", "delete"])
def test_assessment_mutations_enqueue_rebuild(store: SqlAlchemyStore, mutation: str):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    trace_id = _new_trace(store, exp_id, DAY_A_MS)
    assessment = _add_feedback(store, trace_id, value=0.2)
    run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)
    assert _count(store, SqlTraceRollupRebuild) == 0

    if mutation == "create":
        _add_feedback(store, trace_id, value=0.8, name="late")
    elif mutation == "update":
        store.update_assessment(
            trace_id=trace_id,
            assessment_id=assessment.assessment_id,
            feedback=FeedbackValue(value=0.8),
        )
    else:
        store.delete_assessment(trace_id, assessment.assessment_id)

    assert _count(store, SqlTraceRollupRebuild, rollup_family=RollupFamily.ASSESSMENT.value) == 1


def test_run_deletion_enqueues_assessment_rebuild(store: SqlAlchemyStore):
    # Deleting a run hard-deletes its source-run assessments via _mark_run_deleted, which must
    # invalidate the assessment partitions those rows fell in so their rollups are rebuilt.
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    run = store.create_run(
        experiment_id=exp_id,
        user_id="tester",
        start_time=DAY_A_MS,
        tags=[],
        run_name="eval-run",
    )
    trace_id = _new_trace(store, exp_id, DAY_A_MS)
    store.create_assessment(
        Feedback(
            trace_id=trace_id,
            name="quality",
            value=0.3,
            source=SOURCE,
            metadata={AssessmentMetadataKey.SOURCE_RUN_ID: run.info.run_id},
        )
    )
    run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)
    assert _count(store, SqlTraceRollupRebuild) == 0

    store.delete_run(run.info.run_id)

    # Only the assessment family is invalidated: run deletion does not remove the trace rows.
    assert _count(store, SqlTraceRollupRebuild, rollup_family=RollupFamily.TRACE_METRIC.value) == 0
    with store.ManagedSessionMaker() as session:
        queued = {
            (row.experiment_id, row.rollup_day, row.rollup_family)
            for row in session.query(SqlTraceRollupRebuild)
        }
    assert queued == {(int(exp_id), _day_of(DAY_A_MS), RollupFamily.ASSESSMENT.value)}


def test_null_denormalized_assessment_does_not_abort_run(store: SqlAlchemyStore):
    # The denormalized assessment columns are nullable (online prepopulation adds them before
    # backfill; orphaned assessments never get backfilled). A valid row with NULL columns must be
    # skipped by the day scan, not raise int(None) and abort maintenance for every experiment.
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    trace_id = _new_trace(store, exp_id, DAY_A_MS)
    assessment = _add_feedback(store, trace_id, value=0.5)
    with store.ManagedSessionMaker(read_only=False) as session:
        session.query(SqlAssessments).filter(
            SqlAssessments.assessment_id == assessment.assessment_id
        ).update(
            {"experiment_id": None, "trace_timestamp_ms": None},
            synchronize_session=False,
        )

    stats = run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)

    # The trace-metric partition still builds; the NULL assessment contributes to no rollup.
    assert stats.trace_metric.built == 1
    assert stats.assessment.built == 0
    assert _count(store, SqlAssessmentDailyRollup) == 0


def test_max_partitions_cap_limits_builds_across_runs(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    _new_trace(store, exp_id, DAY_A_MS)
    _new_trace(store, exp_id, DAY_B_MS)

    runs = [
        run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS, max_partitions_per_run=1)
        for _ in range(4)
    ]
    assert (
        sum(
            family.skipped_cap
            for family in (runs[0].trace_metric, runs[0].span_cost, runs[0].assessment)
        )
        >= 1
    )
    assert sum(stats.trace_metric.built for stats in runs) == 2
    for stats in runs:
        published = sum(
            family.built + family.emptied
            for family in (stats.trace_metric, stats.span_cost, stats.assessment)
        )
        assert published <= 1

    with store.ManagedSessionMaker() as session:
        built_days = (
            session
            .query(SqlTraceMetricDailyRollup.rollup_day)
            .filter_by(metric_name=TraceMetricKey.TRACE_COUNT, grouping_set="global")
            .distinct()
            .count()
        )
    assert built_days == 2


def test_new_discovery_builds_oldest_day_first_without_gaps(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add_all([
            SqlTraceInfo(
                request_id=f"tr-{uuid.uuid4()}",
                experiment_id=int(exp_id),
                timestamp_ms=DAY_A_MS,
                execution_time_ms=100,
                status=TraceStatus.OK.value,
            ),
            SqlTraceInfo(
                request_id=f"tr-{uuid.uuid4()}",
                experiment_id=int(exp_id),
                timestamp_ms=DAY_B_MS,
                execution_time_ms=100,
                status=TraceStatus.OK.value,
            ),
        ])

    run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS, max_partitions_per_run=1)

    with store.ManagedSessionMaker() as session:
        built_days = {
            rollup_day
            for (rollup_day,) in session.query(SqlTraceMetricDailyRollup.rollup_day).distinct()
        }
    assert built_days == {_day_of(DAY_A_MS)}


def test_new_discovery_stops_experiment_after_earlier_day_failure(
    store: SqlAlchemyStore, monkeypatch
):
    failed_exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    healthy_exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    _new_trace(store, failed_exp_id, DAY_A_MS)
    _new_trace(store, failed_exp_id, DAY_B_MS)
    _new_trace(store, healthy_exp_id, DAY_A_MS)
    attempted = []
    original_rebuild = trace_rollups._rebuild_partition

    def fail_first(session_factory, family, partition, *args):
        attempted.append((family, partition))
        if family == RollupFamily.TRACE_METRIC and partition[0] == int(failed_exp_id):
            raise RuntimeError("injected failure")
        return original_rebuild(session_factory, family, partition, *args)

    monkeypatch.setattr(trace_rollups, "_rebuild_partition", fail_first)

    stats = run_sql_trace_rollups(
        store.engine,
        now_ms=FUTURE_NOW_MS,
        max_partitions_per_run=3,
    )

    assert {candidate for candidate in attempted if candidate[0] == RollupFamily.TRACE_METRIC} == {
        (RollupFamily.TRACE_METRIC, (int(failed_exp_id), DAY_A_MS // MS_PER_DAY)),
        (RollupFamily.TRACE_METRIC, (int(healthy_exp_id), DAY_A_MS // MS_PER_DAY)),
    }
    assert stats.trace_metric.failed == 1
    assert stats.trace_metric.built == 1
    assert (
        _count(
            store,
            SqlTraceRollupRebuild,
            experiment_id=int(failed_exp_id),
            rollup_family=RollupFamily.TRACE_METRIC.value,
        )
        == 2
    )
    assert (
        _count(
            store,
            SqlTraceMetricDailyRollup,
            experiment_id=int(healthy_exp_id),
        )
        > 0
    )


def test_steady_state_does_not_run_per_experiment_day_discovery(
    store: SqlAlchemyStore, monkeypatch
):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    _new_trace(store, exp_id, DAY_A_MS)
    run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)
    day_discovery = Mock(side_effect=AssertionError("steady state must not scan source days"))
    monkeypatch.setattr(trace_rollups, "_new_candidates_for_experiment", day_discovery)

    run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)

    day_discovery.assert_not_called()


def test_steady_state_discovery_filters_on_raw_timestamp_boundaries(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    _new_trace(store, exp_id, DAY_A_MS)
    run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add(
            SqlTraceInfo(
                request_id=f"tr-{uuid.uuid4()}",
                experiment_id=int(exp_id),
                timestamp_ms=DAY_B_MS,
                execution_time_ms=100,
                status=TraceStatus.OK.value,
            )
        )

    statements = []

    def capture_select(_conn, _cursor, statement, _parameters, _context, _executemany):
        normalized = " ".join(statement.upper().split())
        if normalized.startswith("SELECT") and "FROM TRACE_INFO" in normalized:
            statements.append(normalized)

    frozen_day_bucket = (FUTURE_NOW_MS - ROLLUP_ELIGIBILITY_LAG_MS) // MS_PER_DAY
    event.listen(store.engine, "before_cursor_execute", capture_select)
    try:
        assert trace_rollups._candidate_experiment_ids(
            store.ManagedSessionMaker,
            RollupFamily.TRACE_METRIC,
            frozen_day_bucket,
        ) == [int(exp_id)]
        candidates, _ = trace_rollups._new_candidates_for_experiment(
            store.ManagedSessionMaker,
            RollupFamily.TRACE_METRIC,
            int(exp_id),
            frozen_day_bucket,
            10,
        )
    finally:
        event.remove(store.engine, "before_cursor_execute", capture_select)

    assert candidates == [(RollupFamily.TRACE_METRIC, (int(exp_id), DAY_B_MS // MS_PER_DAY))]
    assert len(statements) == 2
    for statement in statements:
        assert "TRACE_INFO.TIMESTAMP_MS <" in statement
        assert "TRACE_INFO.TIMESTAMP_MS >=" in statement


def test_queued_rebuild_precedes_new_partition_across_families(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    trace_id = _new_trace(store, exp_id, DAY_A_MS)
    run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)

    # This creates a queued assessment rebuild for day A.
    _add_feedback(store, trace_id, value=0.7)
    # Simulate raw data that predates queue invalidation support: day B is eligible but unbuilt and
    # has no queue entry.
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add(
            SqlTraceInfo(
                request_id=f"tr-{uuid.uuid4()}",
                experiment_id=int(exp_id),
                timestamp_ms=DAY_B_MS,
                execution_time_ms=100,
                status=TraceStatus.OK.value,
            )
        )

    stats = run_sql_trace_rollups(
        store.engine,
        now_ms=FUTURE_NOW_MS,
        max_partitions_per_run=1,
    )

    assert stats.assessment.built == 1
    assert stats.trace_metric.built == 0
    with store.ManagedSessionMaker() as session:
        assert (
            session
            .query(SqlTraceMetricDailyRollup)
            .filter_by(experiment_id=int(exp_id), rollup_day=_day_of(DAY_B_MS))
            .count()
            == 0
        )


def test_full_queue_batch_does_not_scan_for_new_source_candidates(
    store: SqlAlchemyStore, monkeypatch
):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    _new_trace(store, exp_id, DAY_A_MS)
    candidate_scan = Mock(side_effect=AssertionError("source candidates must not be scanned"))
    monkeypatch.setattr(trace_rollups, "_candidate_experiment_ids", candidate_scan)

    run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS, max_partitions_per_run=1)

    candidate_scan.assert_not_called()


def test_deferred_queue_entry_does_not_starve_later_eligible_work(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    trace_id = _new_trace(store, exp_id, DAY_A_MS)
    with store.ManagedSessionMaker(read_only=False) as session:
        session.query(SqlTraceRollupRebuild).delete(synchronize_session=False)
        session.add_all([
            SqlSpan(
                trace_id=trace_id,
                experiment_id=int(exp_id),
                span_id="span-open",
                status="UNSET",
                start_time_unix_nano=DAY_A_MS * 1_000_000,
                end_time_unix_nano=None,
                content="{}",
            ),
            SqlTraceRollupRebuild(
                experiment_id=int(exp_id),
                rollup_day=_day_of(DAY_A_MS),
                rollup_family=RollupFamily.TRACE_METRIC.value,
            ),
            SqlTraceInfo(
                request_id=f"tr-{uuid.uuid4()}",
                experiment_id=int(exp_id),
                timestamp_ms=DAY_B_MS,
                execution_time_ms=100,
                status=TraceStatus.OK.value,
            ),
        ])

    frozen_day_bucket = (FUTURE_NOW_MS - ROLLUP_ELIGIBILITY_LAG_MS) // MS_PER_DAY
    assert trace_rollups._candidate_experiment_ids(
        store.ManagedSessionMaker,
        RollupFamily.TRACE_METRIC,
        frozen_day_bucket,
    ) == [int(exp_id)]
    candidates, _ = trace_rollups._new_candidates_for_experiment(
        store.ManagedSessionMaker,
        RollupFamily.TRACE_METRIC,
        int(exp_id),
        frozen_day_bucket,
        1,
    )
    assert candidates == [(RollupFamily.TRACE_METRIC, (int(exp_id), DAY_B_MS // MS_PER_DAY))]

    stats = run_sql_trace_rollups(
        store.engine,
        now_ms=FUTURE_NOW_MS,
        max_partitions_per_run=1,
    )

    assert stats.trace_metric.deferred == 1
    assert stats.trace_metric.built == 1
    assert _count(store, SqlTraceRollupRebuild, rollup_family=RollupFamily.TRACE_METRIC.value) == 1


def test_percentile_columns_are_null_on_sqlite(store: SqlAlchemyStore):
    # Percentiles are Postgres-only (PERCENTILE_BACKENDS); on sqlite they must stay null.
    if store.engine.dialect.name != "sqlite":
        pytest.skip("sqlite-specific assertion")
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    _new_trace(store, exp_id, DAY_A_MS, duration_ms=250)

    run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)

    with store.ManagedSessionMaker() as session:
        latency = (
            session
            .query(SqlTraceMetricDailyRollup)
            .filter_by(metric_name=TraceMetricKey.LATENCY, grouping_set="global")
            .one()
        )
        percentiles = (latency.p50_value, latency.p90_value, latency.p99_value)
    assert percentiles == (None, None, None)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_partitions_per_run": 0}, "max_partitions_per_run must be positive"),
        ({"max_workers": 0}, "max_workers must be positive"),
    ],
)
def test_run_sql_trace_rollups_validates_arguments(
    store: SqlAlchemyStore, kwargs: dict[str, int], message: str
):
    with pytest.raises(ValueError, match=message):
        run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS, **kwargs)


def test_delete_sql_trace_rollups_preserves_authoritative_data(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    trace_id = _new_trace(store, exp_id, DAY_A_MS)
    rollup_day = _day_of(DAY_A_MS)
    with store.ManagedSessionMaker(read_only=False) as session:
        session.query(SqlTraceRollupRebuild).delete(synchronize_session=False)
        session.add_all([
            SqlTraceMetricDailyRollup(
                experiment_id=int(exp_id),
                rollup_day=rollup_day,
                metric_name=TraceMetricKey.TRACE_COUNT,
                grouping_set="global",
                sample_count=1,
            ),
            SqlSpanCostDailyRollup(
                experiment_id=int(exp_id),
                rollup_day=rollup_day,
                metric_name=SpanMetricKey.TOTAL_COST,
                grouping_set="global",
                sample_count=1,
                sum_value=0.5,
            ),
            SqlAssessmentDailyRollup(
                experiment_id=int(exp_id),
                rollup_day=rollup_day,
                metric_name=AssessmentMetricKey.ASSESSMENT_COUNT,
                grouping_set="global",
                sample_count=1,
            ),
            SqlTraceRollupRebuild(
                experiment_id=int(exp_id),
                rollup_day=rollup_day,
                rollup_family=RollupFamily.TRACE_METRIC.value,
            ),
        ])

    assert sql_trace_rollup_rows_exist(store.engine)

    stats = delete_sql_trace_rollups(store.engine)

    assert stats == RollupDeleteStats(
        trace_metric=1,
        span_cost=1,
        assessment=1,
        rebuild_queue=1,
    )
    assert not sql_trace_rollup_rows_exist(store.engine)
    assert _count(store, SqlTraceRollupRebuild) == 0
    with store.ManagedSessionMaker() as session:
        assert session.query(SqlTraceInfo).filter_by(request_id=trace_id).one()


def test_delete_trace_rollups_cli(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add(
            SqlTraceMetricDailyRollup(
                experiment_id=int(exp_id),
                rollup_day=_day_of(DAY_A_MS),
                metric_name=TraceMetricKey.TRACE_COUNT,
                grouping_set="global",
                sample_count=1,
            )
        )

    result = CliRunner().invoke(
        mlflow.db.commands,
        ["delete-trace-rollups", "--yes"],
        env={"MLFLOW_TRACKING_URI": store.engine.url.render_as_string(hide_password=False)},
    )

    assert result.exit_code == 0, result.output
    assert "trace_metric=1" in result.output
    assert "rebuild_queue=0" in result.output
    assert _count(store, SqlTraceMetricDailyRollup) == 0


def test_rebuild_lock_serializes_writer_after_publisher(store: SqlAlchemyStore, monkeypatch):
    if store.engine.dialect.name not in {"mssql", "sqlite"}:
        pytest.skip("covers the backend-specific SQL Server and SQLite rebuild locks")

    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    trace_id = _new_trace(store, exp_id, DAY_A_MS)
    with store.ManagedSessionMaker(read_only=False) as session:
        session.query(SqlTraceRollupRebuild).delete(synchronize_session=False)
        session.add(
            SqlTraceRollupRebuild(
                experiment_id=int(exp_id),
                rollup_day=_day_of(DAY_A_MS),
                rollup_family=RollupFamily.TRACE_METRIC.value,
            )
        )
    builder_locked = Event()
    release_builder = Event()
    writer_finished = Event()
    original_ensure = trace_rollups.ensure_locked_rebuild_entry

    def pause_after_builder_lock(session, family, experiment_id, rollup_day):
        entry = original_ensure(session, family, experiment_id, rollup_day)
        builder_locked.set()
        assert release_builder.wait(timeout=10)
        return entry

    monkeypatch.setattr(trace_rollups, "ensure_locked_rebuild_entry", pause_after_builder_lock)

    def publish():
        run_sql_trace_rollups(
            store.engine,
            now_ms=FUTURE_NOW_MS,
            max_partitions_per_run=1,
        )

    def write_source():
        assert builder_locked.wait(timeout=10)
        store.deprecated_end_trace_v2(
            trace_id,
            DAY_A_MS + 200,
            TraceStatus.ERROR,
            {},
            {},
        )
        writer_finished.set()

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-rebuild-lock") as executor:
        publisher = executor.submit(publish)
        writer = executor.submit(write_source)
        assert builder_locked.wait(timeout=10)
        assert not writer_finished.wait(timeout=0.5)
        with store.ManagedSessionMaker() as session:
            assert session.query(SqlTraceInfo).filter_by(request_id=trace_id).one().status == "OK"
        release_builder.set()
        publisher.result(timeout=10)
        writer.result(timeout=10)

    # The invalidation commits after publication and leaves the partition marked stale.
    assert _count(store, SqlTraceRollupRebuild, rollup_family=RollupFamily.TRACE_METRIC.value) == 1


def test_assessment_bulk_invalidation_materializes_only_distinct_days(
    store: SqlAlchemyStore, monkeypatch
):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    trace_a = _new_trace(store, exp_id, DAY_A_MS)
    trace_b = _new_trace(store, exp_id, DAY_B_MS)
    _add_feedback(store, trace_a, name="quality-a")
    _add_feedback(store, trace_a, name="quality-b")
    _add_feedback(store, trace_b, name="quality-c")
    captured = []

    def capture(_session, partitions):
        captured.extend(
            (family, experiment_id, list(timestamps))
            for family, experiment_id, timestamps in partitions
        )

    monkeypatch.setattr(sqlalchemy_store_module, "enqueue_rollup_rebuild_partitions", capture)
    with store.ManagedSessionMaker(read_only=False) as session:
        store._enqueue_assessment_rebuilds_for_filter(
            session,
            SqlAssessments.experiment_id == int(exp_id),
        )

    assert captured == [
        (RollupFamily.ASSESSMENT, int(exp_id), [DAY_A_MS // MS_PER_DAY * MS_PER_DAY]),
        (RollupFamily.ASSESSMENT, int(exp_id), [DAY_B_MS // MS_PER_DAY * MS_PER_DAY]),
    ]


def test_trace_delete_uses_single_locking_select_when_rollups_are_disabled(
    store: SqlAlchemyStore, monkeypatch
):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    trace_id = _new_trace(store, exp_id, DAY_A_MS)
    monkeypatch.setenv(MLFLOW_SQL_TRACE_ROLLUPS_ENABLED.name, "false")
    select_ids = Mock(wraps=store._select_trace_ids_for_delete)
    enqueue = Mock(wraps=store._enqueue_rollup_rebuilds_for_trace_delete)
    relock = Mock(wraps=store._lock_trace_ids_for_delete)
    monkeypatch.setattr(store, "_select_trace_ids_for_delete", select_ids)
    monkeypatch.setattr(store, "_enqueue_rollup_rebuilds_for_trace_delete", enqueue)
    monkeypatch.setattr(store, "_lock_trace_ids_for_delete", relock)

    assert store.delete_traces(exp_id, trace_ids=[trace_id]) == 1

    assert select_ids.call_count == 1
    assert select_ids.call_args.kwargs["lock"] is True
    enqueue.assert_not_called()
    relock.assert_not_called()


def test_start_trace_conflict_skips_rollup_snapshot_queries_when_disabled(
    store: SqlAlchemyStore, monkeypatch
):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    trace_id = f"tr-{uuid.uuid4()}"
    monkeypatch.setenv(MLFLOW_SQL_TRACE_ROLLUPS_ENABLED.name, "false")
    store.log_spans(
        exp_id,
        [create_test_span(trace_id, span_id=1, start_ns=DAY_A_MS * 1_000_000)],
    )
    selected_span_rows = []

    def capture_select(_conn, _cursor, statement, _parameters, _context, _executemany):
        normalized = " ".join(statement.upper().split())
        if normalized.startswith("SELECT") and "FROM SPANS" in normalized:
            selected_span_rows.append(normalized)

    event.listen(store.engine, "before_cursor_execute", capture_select)
    try:
        store.start_trace(
            TraceInfo(
                trace_id=trace_id,
                trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
                request_time=DAY_A_MS,
                execution_duration=100,
                state=TraceState.OK,
                tags={TraceTagKey.TRACE_NAME: "rollup-test"},
            )
        )
    finally:
        event.remove(store.engine, "before_cursor_execute", capture_select)

    assert selected_span_rows == []


def test_mssql_rebuild_lock_compiles_update_and_key_range_hints():
    session = Mock()
    session.get_bind.return_value.dialect.name = "mssql"
    query = Session().query(SqlTraceRollupRebuild)

    statement = _lock_rebuild_entry_query(session, query).statement.compile(dialect=mssql.dialect())

    assert "WITH (UPDLOCK, HOLDLOCK)" in str(statement)


def test_rebuild_keys_are_locked_in_global_order(monkeypatch):
    locked = []
    monkeypatch.setattr(
        sql_trace_rollup_utils,
        "ensure_locked_rebuild_entry",
        lambda _session, family, experiment_id, rollup_day: locked.append((
            experiment_id,
            rollup_day,
            family,
        )),
    )
    monkeypatch.setattr(
        sql_trace_rollup_utils,
        "get_current_time_millis",
        lambda: FUTURE_NOW_MS,
    )

    enqueue_rollup_rebuild_partitions(
        Mock(),
        [
            (RollupFamily.TRACE_METRIC, 2, [DAY_B_MS, DAY_A_MS]),
            (RollupFamily.SPAN_COST, 1, [DAY_B_MS]),
            (RollupFamily.ASSESSMENT, 1, [DAY_A_MS]),
        ],
    )

    assert locked == sorted(locked, key=lambda key: (key[0], key[1], key[2].value))


def test_non_sqlite_maintenance_uses_bounded_worker_pool(store: SqlAlchemyStore, monkeypatch):
    if store.engine.dialect.name == "sqlite":
        pytest.skip("requires a backend that supports concurrent writers")

    exp_a = store.create_experiment(f"exp-{uuid.uuid4()}")
    exp_b = store.create_experiment(f"exp-{uuid.uuid4()}")
    _new_trace(store, exp_a, DAY_A_MS)
    _new_trace(store, exp_b, DAY_B_MS)
    barrier = Barrier(2)
    worker_threads = set()

    def rebuild(*_args, **_kwargs):
        worker_threads.add(get_ident())
        barrier.wait(timeout=10)
        return "built"

    monkeypatch.setattr(trace_rollups, "_rebuild_partition", rebuild)

    stats = run_sql_trace_rollups(
        store.engine,
        now_ms=FUTURE_NOW_MS,
        max_partitions_per_run=2,
        max_workers=2,
    )

    assert stats.trace_metric.built == 2
    assert len(worker_threads) == 2
