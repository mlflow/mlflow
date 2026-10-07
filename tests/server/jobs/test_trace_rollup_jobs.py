from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mlflow.environment_variables import (
    MLFLOW_SERVER_ENABLE_JOB_EXECUTION,
    MLFLOW_SQL_TRACE_ROLLUPS_ENABLED,
    MLFLOW_TRACE_ROLLUPS_MAX_PARTITIONS_PER_RUN,
    MLFLOW_TRACE_ROLLUPS_MAX_WORKERS,
    MLFLOW_TRACE_ROLLUPS_SCHEDULE,
)
from mlflow.exceptions import MlflowException
from mlflow.server.jobs.utils import (
    initialize_periodic_tasks_tracking_store,
    register_periodic_tasks,
)
from mlflow.store.db.trace_rollups import RollupBuildStats, RollupFamilyBuildStats
from mlflow.tracing import trace_rollup_service
from mlflow.tracing.trace_rollup_service import (
    run_sql_trace_rollup_scheduler,
    validate_and_resolve_sql_trace_rollup_schedule,
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
        span_cost=RollupFamilyBuildStats(),
        assessment=RollupFamilyBuildStats(emptied=1),
    )


def test_rollup_schedule_defaults_to_daily_0200_utc(monkeypatch):
    monkeypatch.delenv(MLFLOW_TRACE_ROLLUPS_SCHEDULE.name, raising=False)

    schedule = validate_and_resolve_sql_trace_rollup_schedule()

    assert (schedule.minute, schedule.hour, schedule.day, schedule.month, schedule.day_of_week) == (
        "0",
        "2",
        "*",
        "*",
        "*",
    )


def test_rollup_schedule_accepts_five_field_cron(monkeypatch):
    monkeypatch.setenv(MLFLOW_TRACE_ROLLUPS_SCHEDULE.name, "15 */6 * * 1-5")

    schedule = validate_and_resolve_sql_trace_rollup_schedule()

    assert schedule.minute == "15"
    assert schedule.hour == "*/6"
    assert schedule.day_of_week == "1-5"


@pytest.mark.parametrize("schedule", ["0 2 * *", "foo 2 * * *", "61 2 * * *", "0 24 * * *"])
def test_rollup_schedule_rejects_invalid_cron(monkeypatch, schedule):
    monkeypatch.setenv(MLFLOW_TRACE_ROLLUPS_SCHEDULE.name, schedule)

    with pytest.raises(MlflowException, match="five-field UTC cron"):
        validate_and_resolve_sql_trace_rollup_schedule()


def test_register_periodic_tasks_includes_locked_rollup_scheduler(monkeypatch):
    monkeypatch.setenv(MLFLOW_SQL_TRACE_ROLLUPS_ENABLED.name, "true")
    monkeypatch.delenv(MLFLOW_TRACE_ROLLUPS_SCHEDULE.name, raising=False)
    huey = _RecordingHuey()

    register_periodic_tasks(huey)

    assert "sql_trace_rollup_scheduler" in huey.tasks
    assert huey.locks["sql_trace_rollup_scheduler"] == "sql-trace-rollup-scheduler-lock"


def test_periodic_worker_initializes_tracking_store_from_public_server_config(monkeypatch):
    monkeypatch.setenv("MLFLOW_BACKEND_STORE_URI", "sqlite:///primary.db")
    monkeypatch.setenv("MLFLOW_DEFAULT_ARTIFACT_ROOT", "file:///artifacts")
    expected = Mock(engine=object())
    get_store = Mock(return_value=expected)
    monkeypatch.setattr("mlflow.tracking._tracking_service.utils._get_store", get_store)

    assert initialize_periodic_tasks_tracking_store() is expected
    get_store.assert_called_once_with(
        store_uri="sqlite:///primary.db",
        artifact_uri="file:///artifacts",
    )


def test_periodic_worker_requires_explicit_backend_store(monkeypatch):
    monkeypatch.delenv("MLFLOW_BACKEND_STORE_URI", raising=False)

    with pytest.raises(MlflowException, match="MLFLOW_BACKEND_STORE_URI"):
        initialize_periodic_tasks_tracking_store()


def test_registered_periodic_services_share_tracking_store(monkeypatch):
    monkeypatch.setenv(MLFLOW_SQL_TRACE_ROLLUPS_ENABLED.name, "true")
    tracking_store = object()
    archival = Mock()
    rollup = Mock()
    monkeypatch.setattr("mlflow.server.jobs.utils._run_trace_archival_scheduler", archival)
    settings = SimpleNamespace(interval_seconds=60)
    monkeypatch.setattr(
        "mlflow.server.jobs.utils._get_trace_archival_scheduler_settings",
        Mock(return_value=settings),
    )
    monkeypatch.setattr(
        "mlflow.server.jobs.utils._should_run_trace_archival_scheduler", Mock(return_value=True)
    )
    monkeypatch.setattr(trace_rollup_service, "run_sql_trace_rollup_scheduler", rollup)
    monkeypatch.setattr(
        "mlflow.server.jobs.utils.initialize_periodic_tasks_tracking_store",
        Mock(return_value=tracking_store),
    )
    huey = _RecordingHuey()

    register_periodic_tasks(huey)
    huey.tasks["trace_archival_scheduler"][1]()
    huey.tasks["sql_trace_rollup_scheduler"][1]()

    archival.assert_called_once_with(tracking_store, settings=settings)
    rollup.assert_called_once_with(tracking_store)


def test_periodic_tasks_initialize_store_lazily_and_retry_after_failure(monkeypatch):
    monkeypatch.setenv(MLFLOW_SQL_TRACE_ROLLUPS_ENABLED.name, "true")
    store = object()
    initialize = Mock(side_effect=[RuntimeError("not ready"), store])
    archival = Mock()
    rollup = Mock()
    monkeypatch.setattr(
        "mlflow.server.jobs.utils.initialize_periodic_tasks_tracking_store", initialize
    )
    monkeypatch.setattr("mlflow.server.jobs.utils._run_trace_archival_scheduler", archival)
    monkeypatch.setattr(trace_rollup_service, "run_sql_trace_rollup_scheduler", rollup)
    monkeypatch.setattr(
        "mlflow.server.jobs.utils._get_trace_archival_scheduler_settings", Mock(return_value=None)
    )
    huey = _RecordingHuey()

    register_periodic_tasks(huey)

    # Registration and the store-independent scorer do not require a tracking store.
    huey.tasks["online_scoring_scheduler"][1]()
    initialize.assert_not_called()

    # A failed first store initialization is not cached; the next poll can recover.
    huey.tasks["sql_trace_rollup_scheduler"][1]()
    huey.tasks["sql_trace_rollup_scheduler"][1]()
    assert initialize.call_count == 2
    rollup.assert_called_once_with(store)


def test_invalid_schedule_is_ignored_when_rollups_are_disabled(monkeypatch):
    monkeypatch.setenv(MLFLOW_SQL_TRACE_ROLLUPS_ENABLED.name, "false")
    monkeypatch.setenv(MLFLOW_TRACE_ROLLUPS_SCHEDULE.name, "invalid")
    huey = _RecordingHuey()

    register_periodic_tasks(huey)

    assert "online_scoring_scheduler" in huey.tasks
    assert "trace_archival_scheduler" in huey.tasks
    assert "sql_trace_rollup_scheduler" not in huey.tasks


def test_invalid_schedule_fails_registration_when_rollups_are_enabled(monkeypatch):
    monkeypatch.setenv(MLFLOW_SQL_TRACE_ROLLUPS_ENABLED.name, "true")
    monkeypatch.setenv(MLFLOW_TRACE_ROLLUPS_SCHEDULE.name, "invalid")

    with pytest.raises(MlflowException, match="five-field UTC cron"):
        register_periodic_tasks(_RecordingHuey())


@pytest.mark.parametrize(
    "variable",
    [MLFLOW_TRACE_ROLLUPS_MAX_PARTITIONS_PER_RUN, MLFLOW_TRACE_ROLLUPS_MAX_WORKERS],
)
@pytest.mark.parametrize("value", ["abc", "0", "-1"])
def test_rollup_positive_integer_settings_are_validated_at_registration(
    monkeypatch, variable, value
):
    monkeypatch.setenv(MLFLOW_SQL_TRACE_ROLLUPS_ENABLED.name, "true")
    monkeypatch.setenv(variable.name, value)

    with pytest.raises(MlflowException, match=variable.name):
        register_periodic_tasks(_RecordingHuey())


def test_scheduler_noops_when_rollups_are_disabled(monkeypatch):
    monkeypatch.setenv(MLFLOW_SERVER_ENABLE_JOB_EXECUTION.name, "true")
    monkeypatch.setenv(MLFLOW_SQL_TRACE_ROLLUPS_ENABLED.name, "false")
    maintenance = Mock()
    monkeypatch.setattr(trace_rollup_service, "run_sql_trace_rollups", maintenance)

    assert run_sql_trace_rollup_scheduler(object()) is None
    maintenance.assert_not_called()


def test_service_entrypoint_logs_historical_bootstrap(monkeypatch, caplog):
    monkeypatch.setenv(MLFLOW_SERVER_ENABLE_JOB_EXECUTION.name, "false")
    monkeypatch.setenv(MLFLOW_SQL_TRACE_ROLLUPS_ENABLED.name, "true")
    engine = object()
    tracking_store = Mock(engine=engine)
    expected = _stats()
    maintenance = Mock(return_value=expected)
    monkeypatch.setattr(trace_rollup_service, "run_sql_trace_rollups", maintenance)
    monkeypatch.setattr(
        trace_rollup_service, "sql_trace_rollup_rows_exist", Mock(return_value=False)
    )

    assert run_sql_trace_rollup_scheduler(tracking_store) == expected
    assert "MLFLOW_TRACE_ROLLUPS_MAX_PARTITIONS_PER_RUN" in caplog.text
    maintenance.assert_called_once_with(
        engine,
        max_partitions_per_run=MLFLOW_TRACE_ROLLUPS_MAX_PARTITIONS_PER_RUN.get(),
        max_workers=MLFLOW_TRACE_ROLLUPS_MAX_WORKERS.get(),
    )


def test_scheduler_noops_for_non_sql_tracking_store(monkeypatch):
    monkeypatch.setenv(MLFLOW_SERVER_ENABLE_JOB_EXECUTION.name, "true")
    monkeypatch.setenv(MLFLOW_SQL_TRACE_ROLLUPS_ENABLED.name, "true")
    maintenance = Mock()
    monkeypatch.setattr(trace_rollup_service, "run_sql_trace_rollups", maintenance)

    assert run_sql_trace_rollup_scheduler(object()) is None
    maintenance.assert_not_called()


def test_scheduler_delegates_to_shared_maintenance_path(monkeypatch):
    monkeypatch.setenv(MLFLOW_SERVER_ENABLE_JOB_EXECUTION.name, "true")
    monkeypatch.setenv(MLFLOW_SQL_TRACE_ROLLUPS_ENABLED.name, "true")
    monkeypatch.setenv(MLFLOW_TRACE_ROLLUPS_MAX_PARTITIONS_PER_RUN.name, "7")
    monkeypatch.setenv(MLFLOW_TRACE_ROLLUPS_MAX_WORKERS.name, "3")
    engine = object()
    tracking_store = Mock(engine=engine)
    expected = _stats()
    maintenance = Mock(return_value=expected)
    monkeypatch.setattr(trace_rollup_service, "run_sql_trace_rollups", maintenance)
    monkeypatch.setattr(
        trace_rollup_service, "sql_trace_rollup_rows_exist", Mock(return_value=True)
    )

    assert run_sql_trace_rollup_scheduler(tracking_store) == expected
    maintenance.assert_called_once_with(engine, max_partitions_per_run=7, max_workers=3)
