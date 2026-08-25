import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest
import sqlalchemy as sa
from click.testing import CliRunner

import mlflow.db
import mlflow.store.db.utils
from mlflow.entities import AssessmentSource, AssessmentSourceType, Feedback, trace_location
from mlflow.entities.assessment import FeedbackValue
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_status import TraceStatus
from mlflow.environment_variables import MLFLOW_SQL_TRACE_ROLLUPS_ENABLED
from mlflow.store.db import trace_rollups
from mlflow.store.db.trace_rollups import (
    ROLLUP_ELIGIBILITY_LAG_MS,
    RollupBuildStats,
    run_sql_trace_rollups,
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
from mlflow.store.tracking.utils.sql_trace_rollups import RollupFamily, enqueue_rollup_rebuilds
from mlflow.tracing.constant import (
    AssessmentMetadataKey,
    AssessmentMetricKey,
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
    assert second_stats.assessment.built == 0
    assert _count(store, SqlTraceMetricDailyRollup) == first


def test_span_cost_rollups_are_never_built(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    _new_trace(store, exp_id, DAY_A_MS)

    run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)

    assert _count(store, SqlSpanCostDailyRollup) == 0


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
    _enqueue_entry(store, RollupFamily.TRACE_METRIC, exp_id, current_day_ms)

    stats = run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS + 1000)

    assert stats.trace_metric.deferred == 1
    assert _count(store, SqlTraceRollupRebuild, rollup_family=RollupFamily.TRACE_METRIC.value) == 1


def test_day_becomes_eligible_exactly_after_lag(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    _new_trace(store, exp_id, DAY_A_MS)

    # One millisecond before the lag elapses: still ineligible.
    before = run_sql_trace_rollups(store.engine, now_ms=DAY_A_MS + ROLLUP_ELIGIBILITY_LAG_MS - 1)
    assert before.trace_metric.built == 0

    # Exactly at the lag boundary: eligible.
    at = run_sql_trace_rollups(store.engine, now_ms=DAY_A_MS + ROLLUP_ELIGIBILITY_LAG_MS)
    assert at.trace_metric.built == 1


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

    stats = run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)

    assert stats.trace_metric.deferred == 1
    assert stats.trace_metric.built == 0
    # The queue entry survives so a later run retries once the span closes.
    assert _count(store, SqlTraceRollupRebuild, rollup_family=RollupFamily.TRACE_METRIC.value) == 1
    assert _count(store, SqlTraceMetricDailyRollup) == 0


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

    with pytest.raises(RuntimeError, match="aggregation failed"):
        run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS)

    assert _count(store, SqlTraceMetricDailyRollup) == previous_rollup_count
    assert _count(store, SqlTraceRollupRebuild, rollup_family=RollupFamily.TRACE_METRIC.value) == 1


def test_source_writes_enqueue_while_reads_are_disabled(
    store: SqlAlchemyStore, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(MLFLOW_SQL_TRACE_ROLLUPS_ENABLED.name, "false")
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")

    _new_trace(store, exp_id, DAY_A_MS)

    assert _count(store, SqlTraceRollupRebuild, rollup_family=RollupFamily.TRACE_METRIC.value) == 1
    assert _count(store, SqlTraceRollupRebuild, rollup_family=RollupFamily.ASSESSMENT.value) == 1


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
                start_ns=DAY_A_MS * 1_000_000,
                end_ns=(DAY_A_MS + 50) * 1_000_000,
            )
        ],
    )

    assert _count(store, SqlTraceRollupRebuild, rollup_family=RollupFamily.TRACE_METRIC.value) == 1


def test_trace_move_enqueues_old_and_new_partitions(store: SqlAlchemyStore):
    old_exp_id = store.create_experiment(f"old-{uuid.uuid4()}")
    new_exp_id = store.create_experiment(f"new-{uuid.uuid4()}")
    trace_id = _new_trace(store, old_exp_id, DAY_A_MS)
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
    }


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

    first = run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS, max_partitions_per_run=1)
    assert first.trace_metric.built == 1
    assert first.trace_metric.skipped_cap >= 1

    second = run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS, max_partitions_per_run=1)
    assert second.trace_metric.built == 1

    with store.ManagedSessionMaker() as session:
        built_days = (
            session
            .query(SqlTraceMetricDailyRollup.rollup_day)
            .filter_by(metric_name=TraceMetricKey.TRACE_COUNT, grouping_set="global")
            .distinct()
            .count()
        )
    assert built_days == 2


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
        ({"progress_every": 0}, "progress_every must be positive"),
        ({"max_partitions_per_run": 0}, "max_partitions_per_run must be positive"),
    ],
)
def test_run_sql_trace_rollups_validates_arguments(
    store: SqlAlchemyStore, kwargs: dict[str, int], message: str
):
    with pytest.raises(ValueError, match=message):
        run_sql_trace_rollups(store.engine, now_ms=FUTURE_NOW_MS, **kwargs)


def test_build_trace_rollups_cli(store: SqlAlchemyStore):
    exp_id = store.create_experiment(f"exp-{uuid.uuid4()}")
    _new_trace(store, exp_id, DAY_A_MS)
    _add_feedback(store, _new_trace(store, exp_id, DAY_A_MS + 1000), value=0.9)

    result = CliRunner().invoke(
        mlflow.db.commands,
        ["build-trace-rollups"],
        env={
            "MLFLOW_TRACKING_URI": store.engine.url.render_as_string(hide_password=False),
            "MLFLOW_SQL_TRACE_ROLLUPS_ENABLED": "true",
        },
    )

    assert result.exit_code == 0, result.output
    assert "Building SQL daily trace analytics rollups..." in result.output
    assert "trace_metric: built=" in result.output
    assert "assessment: built=" in result.output
    assert "Rollup build completed." in result.output


def test_build_trace_rollups_cli_noops_when_disabled(monkeypatch: pytest.MonkeyPatch):
    create_engine = Mock()
    monkeypatch.setattr(mlflow.store.db.utils, "create_sqlalchemy_engine_with_retry", create_engine)

    result = CliRunner().invoke(
        mlflow.db.commands,
        ["build-trace-rollups", "sqlite:///unused.db"],
        env={"MLFLOW_SQL_TRACE_ROLLUPS_ENABLED": "false"},
    )

    assert result.exit_code == 0
    assert "SQL trace rollups are disabled; no maintenance performed." in result.output
    create_engine.assert_not_called()


def test_build_trace_rollups_cli_reports_errors_and_disposes_engine(monkeypatch):
    engine = Mock()
    monkeypatch.setattr(
        mlflow.store.db.utils,
        "create_sqlalchemy_engine_with_retry",
        lambda _: engine,
    )
    monkeypatch.setattr(
        trace_rollups,
        "run_sql_trace_rollups",
        Mock(side_effect=RuntimeError("rollup build failed")),
    )

    result = CliRunner().invoke(
        mlflow.db.commands,
        ["build-trace-rollups", "sqlite:///unused.db"],
        env={"MLFLOW_SQL_TRACE_ROLLUPS_ENABLED": "true"},
    )

    assert result.exit_code == 1
    assert "Error: rollup build failed" in result.output
    engine.dispose.assert_called_once_with()


def test_build_trace_rollups_cli_does_not_expose_database_credentials(monkeypatch):
    database_url = "postgresql://trace_user:super-secret@database.example/mlflow"
    monkeypatch.setattr(
        mlflow.store.db.utils,
        "create_sqlalchemy_engine_with_retry",
        Mock(
            side_effect=sa.exc.OperationalError(
                f"connect to {database_url}",
                {},
                RuntimeError(database_url),
            )
        ),
    )

    result = CliRunner().invoke(
        mlflow.db.commands,
        ["build-trace-rollups"],
        env={
            "MLFLOW_TRACKING_URI": database_url,
            "MLFLOW_SQL_TRACE_ROLLUPS_ENABLED": "true",
        },
    )

    assert result.exit_code == 1
    assert "Database operation failed (OperationalError)" in result.output
    assert "super-secret" not in result.output


@pytest.mark.parametrize("max_partitions", ["0", "-1"])
def test_build_trace_rollups_cli_rejects_nonpositive_max_partitions(max_partitions: str):
    result = CliRunner().invoke(
        mlflow.db.commands,
        ["build-trace-rollups", "sqlite:///unused.db", "--max-partitions", max_partitions],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--max-partitions'" in result.output
