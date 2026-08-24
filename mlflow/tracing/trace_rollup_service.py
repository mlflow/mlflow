from __future__ import annotations

import logging
from dataclasses import dataclass

from mlflow.environment_variables import (
    MLFLOW_SERVER_ENABLE_JOB_EXECUTION,
    MLFLOW_SQL_TRACE_ROLLUPS_ENABLED,
    MLFLOW_TRACE_ROLLUPS_MAX_PARTITIONS_PER_RUN,
    MLFLOW_TRACE_ROLLUPS_SCHEDULE,
)
from mlflow.exceptions import MlflowException
from mlflow.store.db.trace_rollups import RollupBuildStats, run_sql_trace_rollups

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SqlTraceRollupSchedule:
    minute: str
    hour: str
    day: str
    month: str
    day_of_week: str


def get_sql_trace_rollup_schedule() -> SqlTraceRollupSchedule:
    """Resolve the configured five-field UTC cron expression."""
    schedule = MLFLOW_TRACE_ROLLUPS_SCHEDULE.get().strip()
    fields = schedule.split()
    if len(fields) != 5:
        raise MlflowException.invalid_parameter_value(
            f"{MLFLOW_TRACE_ROLLUPS_SCHEDULE.name} must be a five-field UTC cron expression, "
            f"got {schedule!r}."
        )
    return SqlTraceRollupSchedule(*fields)


def run_sql_trace_rollup_scheduler() -> RollupBuildStats | None:
    """Run one server-owned rollup maintenance pass when scheduler prerequisites are enabled."""
    if not MLFLOW_SERVER_ENABLE_JOB_EXECUTION.get() or not MLFLOW_SQL_TRACE_ROLLUPS_ENABLED.get():
        return None

    from mlflow.server.handlers import _get_tracking_store

    tracking_store = _get_tracking_store()
    engine = getattr(tracking_store, "engine", None)
    if engine is None:
        _logger.info("SQL trace rollup scheduler skipped because the tracking store is not SQL.")
        return None

    stats = run_sql_trace_rollups(
        engine,
        max_partitions_per_run=MLFLOW_TRACE_ROLLUPS_MAX_PARTITIONS_PER_RUN.get(),
    )
    _logger.info(
        "SQL trace rollup maintenance completed: trace_metric=%s, assessment=%s",
        stats.trace_metric,
        stats.assessment,
    )
    return stats
