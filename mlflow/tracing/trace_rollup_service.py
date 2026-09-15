from __future__ import annotations

import logging
from dataclasses import dataclass

from mlflow.environment_variables import (
    MLFLOW_SERVER_ENABLE_JOB_EXECUTION,
    MLFLOW_SQL_TRACE_ROLLUPS_ENABLED,
    MLFLOW_TRACE_ROLLUPS_MAX_PARTITIONS_PER_RUN,
    MLFLOW_TRACE_ROLLUPS_MAX_WORKERS,
    MLFLOW_TRACE_ROLLUPS_SCHEDULE,
)
from mlflow.exceptions import MlflowException
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.store.db.trace_rollups import (
    RollupBuildStats,
    run_sql_trace_rollups,
    sql_trace_rollup_rows_exist,
)
from mlflow.store.db.utils import create_sqlalchemy_engine_with_retry
from mlflow.utils.uri import get_uri_scheme

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SqlTraceRollupSchedule:
    minute: str
    hour: str
    day: str
    month: str
    day_of_week: str


def validate_sql_trace_rollup_startup(backend_store_uri: str | None) -> None:
    """Reject disabling rollups while previously materialized rows remain."""
    if MLFLOW_SQL_TRACE_ROLLUPS_ENABLED.get() or not backend_store_uri:
        return
    if get_uri_scheme(backend_store_uri) not in DATABASE_ENGINES:
        return

    engine = create_sqlalchemy_engine_with_retry(backend_store_uri)
    try:
        has_rollups = sql_trace_rollup_rows_exist(engine)
    finally:
        engine.dispose()
    if has_rollups:
        raise MlflowException.invalid_parameter_value(
            "SQL trace rollups cannot be disabled while materialized rollup rows exist. "
            "Stop all MLflow servers using this database, then run "
            "`mlflow db delete-trace-rollups <database-url>` before restarting."
        )


def get_sql_trace_rollup_schedule() -> SqlTraceRollupSchedule:
    """Resolve the configured five-field UTC cron expression."""
    schedule = MLFLOW_TRACE_ROLLUPS_SCHEDULE.get().strip()
    fields = schedule.split()
    if len(fields) != 5:
        raise MlflowException.invalid_parameter_value(
            f"{MLFLOW_TRACE_ROLLUPS_SCHEDULE.name} must be a five-field UTC cron expression, "
            f"got {schedule!r}."
        )
    resolved = SqlTraceRollupSchedule(*fields)

    # Huey's default parser accepts invalid tokens as expressions that never match. Validate in
    # strict mode here so an enabled scheduler cannot silently stop running because of a typo.
    from huey import crontab

    try:
        crontab(
            minute=resolved.minute,
            hour=resolved.hour,
            day=resolved.day,
            month=resolved.month,
            day_of_week=resolved.day_of_week,
            strict=True,
        )
    except ValueError as e:
        raise MlflowException.invalid_parameter_value(
            f"{MLFLOW_TRACE_ROLLUPS_SCHEDULE.name} must be a valid five-field UTC cron "
            f"expression, got {schedule!r}."
        ) from e
    return resolved


def run_sql_trace_rollup_scheduler(tracking_store) -> RollupBuildStats | None:
    """Run one server-owned rollup maintenance pass when scheduler prerequisites are enabled."""
    if not MLFLOW_SERVER_ENABLE_JOB_EXECUTION.get() or not MLFLOW_SQL_TRACE_ROLLUPS_ENABLED.get():
        return None

    engine = getattr(tracking_store, "engine", None)
    if engine is None:
        _logger.info("SQL trace rollup scheduler skipped because the tracking store is not SQL.")
        return None

    stats = run_sql_trace_rollups(
        engine,
        max_partitions_per_run=MLFLOW_TRACE_ROLLUPS_MAX_PARTITIONS_PER_RUN.get(),
        max_workers=MLFLOW_TRACE_ROLLUPS_MAX_WORKERS.get(),
    )
    _logger.info(
        "SQL trace rollup maintenance completed: trace_metric=%s, span_cost=%s, assessment=%s",
        stats.trace_metric,
        stats.span_cost,
        stats.assessment,
    )
    return stats
