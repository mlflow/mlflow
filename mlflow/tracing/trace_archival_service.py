from __future__ import annotations

import logging
import random
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass

from mlflow.entities.workspace import TraceArchivalConfig
from mlflow.environment_variables import (
    MLFLOW_ENABLE_WORKSPACES,
)
from mlflow.exceptions import MlflowException
from mlflow.tracing.trace_archival_config import get_trace_archival_server_config
from mlflow.utils.uri import append_to_uri_path
from mlflow.utils.validation import (
    _validate_trace_archival_repository_support,
    _validate_trace_archival_retention_string,
)
from mlflow.utils.workspace_context import ServerWorkspaceContext, get_request_workspace
from mlflow.utils.workspace_utils import WORKSPACES_DIR_NAME

_logger = logging.getLogger(__name__)

_TRACE_ARCHIVAL_SCHEDULER_STATE_LOCK = threading.Lock()
_TRACE_ARCHIVAL_SCHEDULER_LAST_RUN_MONOTONIC = 0.0


@dataclass(frozen=True)
class _TraceArchivalSchedulerSettings:
    location: str
    retention: str
    long_retention_allowlist: set[str]
    interval_seconds: int
    max_traces_per_pass: int | None


def _resolve_scheduler_trace_archival_config(
    tracking_store,
    *,
    default_trace_archival_location: str,
    default_retention: str,
) -> TraceArchivalConfig:
    trace_archival_config = tracking_store.resolve_trace_archival_config(
        default_trace_archival_location=default_trace_archival_location,
        default_retention=default_retention,
    ).with_broader_defaults(
        default_location=default_trace_archival_location,
        default_retention=default_retention,
    )

    resolved_trace_archival_location = _validate_trace_archival_repository_support(
        trace_archival_config.config.location,
        parameter_name="resolved_trace_archival_location",
    )
    if trace_archival_config.append_workspace_prefix and (
        workspace_name := get_request_workspace()
    ):
        resolved_trace_archival_location = append_to_uri_path(
            resolved_trace_archival_location,
            WORKSPACES_DIR_NAME,
            workspace_name,
        )

    return TraceArchivalConfig(
        location=resolved_trace_archival_location,
        retention=_validate_trace_archival_retention_string(trace_archival_config.config.retention),
    )


def run_trace_archival_scheduler() -> int:
    """
    Run one scheduler poll for server-owned trace archival.

    This entrypoint is invoked by the periodic Huey task. It resolves scheduler settings,
    enforces the configured in-process polling interval, iterates the applicable workspace
    contexts, resolves broader-scope archival config for each scope, and calls the tracking
    store archival API.

    Returns:
        The total number of traces archived across all processed scopes for this poll. Returns
        ``0`` when the scheduler is disabled, archival is not configured, or the configured
        interval has not elapsed yet.
    """
    settings = _get_trace_archival_scheduler_settings()
    if settings is None or not _should_run_trace_archival_scheduler(settings.interval_seconds):
        return 0

    from mlflow.server.handlers import _get_tracking_store

    tracking_store = _get_tracking_store()
    archived_total = 0
    remaining_traces_per_pass = settings.max_traces_per_pass
    # Count processed scheduler scopes (workspace contexts, or the single default scope).
    scope_count = 0
    start_time = time.monotonic()

    for workspace_ctx in _get_trace_archival_workspace_contexts():
        if remaining_traces_per_pass is not None and remaining_traces_per_pass <= 0:
            break
        with workspace_ctx as workspace:
            scope_count += 1
            workspace_label = f" in workspace '{workspace}'" if workspace else ""
            try:
                resolved_trace_archival_config = _resolve_scheduler_trace_archival_config(
                    tracking_store,
                    default_trace_archival_location=settings.location,
                    default_retention=settings.retention,
                )
            except MlflowException:
                _logger.warning(
                    "Trace archival scheduler skipped%s because the resolved archival "
                    "configuration is invalid or unsupported.",
                    workspace_label,
                    exc_info=True,
                )
                continue

            try:
                archived_in_scope = tracking_store.archive_traces(
                    resolved_trace_archival_location=resolved_trace_archival_config.location,
                    broader_retention=resolved_trace_archival_config.retention,
                    long_retention_allowlist=settings.long_retention_allowlist,
                    max_traces_per_pass=remaining_traces_per_pass,
                )
                archived_total += archived_in_scope
                if remaining_traces_per_pass is not None:
                    remaining_traces_per_pass = max(
                        remaining_traces_per_pass - archived_in_scope,
                        0,
                    )
            except Exception as e:
                _logger.exception(
                    "Trace archival scheduler failed%s: %r",
                    workspace_label,
                    e,
                )

    elapsed_seconds = time.monotonic() - start_time
    _logger.info(
        "Trace archival scheduler pass archived %s trace(s) across %s scope(s) in %.2f second(s).",
        archived_total,
        scope_count,
        elapsed_seconds,
    )
    return archived_total


def _get_trace_archival_scheduler_settings() -> _TraceArchivalSchedulerSettings | None:
    try:
        trace_archival_config = get_trace_archival_server_config()
    except MlflowException:
        _logger.warning(
            "Ignoring invalid trace archival scheduler configuration.",
            exc_info=True,
        )
        return None

    if trace_archival_config is None or not trace_archival_config.enabled:
        return None

    return _TraceArchivalSchedulerSettings(
        location=trace_archival_config.location,
        retention=trace_archival_config.retention,
        long_retention_allowlist=set(trace_archival_config.long_retention_allowlist),
        interval_seconds=trace_archival_config.interval_seconds,
        max_traces_per_pass=trace_archival_config.max_traces_per_pass,
    )


def _should_run_trace_archival_scheduler(interval_seconds: int) -> bool:
    """
    Decide whether the current scheduler poll should run an archival pass.

    Huey polls every minute, but the effective archival cadence may be longer. This helper uses a
    process-local monotonic timestamp and lock to skip polls until the configured interval has
    elapsed, then records the current time when a pass is admitted.

    Args:
        interval_seconds: Minimum number of seconds between admitted archival passes in the
            current process.

    Returns:
        ``True`` if this poll should proceed with archival work, otherwise ``False``.
    """
    global _TRACE_ARCHIVAL_SCHEDULER_LAST_RUN_MONOTONIC

    now = time.monotonic()
    with _TRACE_ARCHIVAL_SCHEDULER_STATE_LOCK:
        if now - _TRACE_ARCHIVAL_SCHEDULER_LAST_RUN_MONOTONIC < interval_seconds:
            return False
        _TRACE_ARCHIVAL_SCHEDULER_LAST_RUN_MONOTONIC = now
        return True


def _get_trace_archival_workspace_contexts():
    if not MLFLOW_ENABLE_WORKSPACES.get():
        return [nullcontext()]

    from mlflow.server.workspace_helpers import _get_workspace_store  # avoid circular import

    store = _get_workspace_store()
    workspaces = list(store.list_workspaces())
    if not workspaces:
        _logger.info("Trace archival scheduler found no workspaces; skipping.")
        return []

    # Shuffle workspace order each pass so a shared pass budget does not always favor the
    # same tenants first. More advanced fairness controls can be layered on later if needed.
    random.shuffle(workspaces)
    return [ServerWorkspaceContext(workspace.name) for workspace in workspaces]
