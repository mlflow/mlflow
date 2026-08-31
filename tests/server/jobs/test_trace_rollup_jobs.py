from unittest.mock import Mock

import pytest

from mlflow.environment_variables import (
    MLFLOW_SERVER_ENABLE_JOB_EXECUTION,
    MLFLOW_SQL_TRACE_ROLLUPS_ENABLED,
    MLFLOW_TRACE_ROLLUPS_MAX_PARTITIONS_PER_RUN,
    MLFLOW_TRACE_ROLLUPS_SCHEDULE,
)
from mlflow.exceptions import MlflowException
from mlflow.server.jobs.utils import register_periodic_tasks
from mlflow.store.db.trace_rollups import RollupBuildStats, RollupFamilyBuildStats
from mlflow.tracing import trace_rollup_service
from mlflow.tracing.trace_rollup_service import (
    get_sql_trace_rollup_schedule,
    run_sql_trace_rollup_scheduler,
)


class _RecordingHuey:
    def __init__(self):
        self.tasks = {}
        self.locks = {}

    def periodic_task(self, validation):
        def decorator(fn):
            self.tasks[fn.__name__] = (validation, fn)
            return fn

        return decorator

    def lock_task(self, lock_name):
        def decorator(fn):
            self.locks[fn.__name__] = lock_name
            return fn

        return decorator


def _stats():
    return RollupBuildStats(
        trace_metric=RollupFamilyBuildStats(built=1),
        assessment=RollupFamilyBuildStats(emptied=1),
    )


def test_rollup_schedule_defaults_to_daily_0200_utc(monkeypatch):
    monkeypatch.delenv(MLFLOW_TRACE_ROLLUPS_SCHEDULE.name, raising=False)

    schedule = get_sql_trace_rollup_schedule()

    assert (schedule.minute, schedule.hour, schedule.day, schedule.month, schedule.day_of_week) == (
        "0",
        "2",
        "*",
        "*",
        "*",
    )


def test_rollup_schedule_accepts_five_field_cron(monkeypatch):
    monkeypatch.setenv(MLFLOW_TRACE_ROLLUPS_SCHEDULE.name, "15 */6 * * 1-5")

    schedule = get_sql_trace_rollup_schedule()

    assert schedule.minute == "15"
    assert schedule.hour == "*/6"
    assert schedule.day_of_week == "1-5"


def test_rollup_schedule_rejects_wrong_field_count(monkeypatch):
    monkeypatch.setenv(MLFLOW_TRACE_ROLLUPS_SCHEDULE.name, "0 2 * *")

    with pytest.raises(MlflowException, match="five-field UTC cron expression"):
        get_sql_trace_rollup_schedule()


def test_register_periodic_tasks_includes_locked_rollup_scheduler(monkeypatch):
    monkeypatch.delenv(MLFLOW_TRACE_ROLLUPS_SCHEDULE.name, raising=False)
    huey = _RecordingHuey()

    register_periodic_tasks(huey)

    assert "sql_trace_rollup_scheduler" in huey.tasks
    assert huey.locks["sql_trace_rollup_scheduler"] == "sql-trace-rollup-scheduler-lock"


def test_invalid_schedule_skips_only_rollup_registration(monkeypatch):
    monkeypatch.setenv(MLFLOW_TRACE_ROLLUPS_SCHEDULE.name, "invalid")
    huey = _RecordingHuey()

    register_periodic_tasks(huey)

    assert "online_scoring_scheduler" in huey.tasks
    assert "trace_archival_scheduler" in huey.tasks
    assert "sql_trace_rollup_scheduler" not in huey.tasks


@pytest.mark.parametrize(
    ("job_execution", "rollups_enabled"),
    [(False, False), (False, True), (True, False)],
)
def test_scheduler_noops_when_prerequisites_disabled(
    monkeypatch, job_execution: bool, rollups_enabled: bool
):
    monkeypatch.setenv(
        MLFLOW_SERVER_ENABLE_JOB_EXECUTION.name, "true" if job_execution else "false"
    )
    monkeypatch.setenv(
        MLFLOW_SQL_TRACE_ROLLUPS_ENABLED.name, "true" if rollups_enabled else "false"
    )
    maintenance = Mock()
    monkeypatch.setattr(trace_rollup_service, "run_sql_trace_rollups", maintenance)

    assert run_sql_trace_rollup_scheduler() is None
    maintenance.assert_not_called()


def test_scheduler_noops_for_non_sql_tracking_store(monkeypatch):
    monkeypatch.setenv(MLFLOW_SERVER_ENABLE_JOB_EXECUTION.name, "true")
    monkeypatch.setenv(MLFLOW_SQL_TRACE_ROLLUPS_ENABLED.name, "true")
    monkeypatch.setattr("mlflow.server.handlers._get_tracking_store", lambda: object())
    maintenance = Mock()
    monkeypatch.setattr(trace_rollup_service, "run_sql_trace_rollups", maintenance)

    assert run_sql_trace_rollup_scheduler() is None
    maintenance.assert_not_called()


def test_scheduler_delegates_to_shared_maintenance_path(monkeypatch):
    monkeypatch.setenv(MLFLOW_SERVER_ENABLE_JOB_EXECUTION.name, "true")
    monkeypatch.setenv(MLFLOW_SQL_TRACE_ROLLUPS_ENABLED.name, "true")
    monkeypatch.setenv(MLFLOW_TRACE_ROLLUPS_MAX_PARTITIONS_PER_RUN.name, "7")
    engine = object()
    monkeypatch.setattr("mlflow.server.handlers._get_tracking_store", lambda: Mock(engine=engine))
    expected = _stats()
    maintenance = Mock(return_value=expected)
    monkeypatch.setattr(trace_rollup_service, "run_sql_trace_rollups", maintenance)

    assert run_sql_trace_rollup_scheduler() == expected
    maintenance.assert_called_once_with(engine, max_partitions_per_run=7)
