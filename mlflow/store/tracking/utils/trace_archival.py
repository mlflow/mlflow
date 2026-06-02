from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from mlflow.entities import TraceInfo
from mlflow.exceptions import MlflowException
from mlflow.tracing.constant import TraceExperimentTagKey
from mlflow.utils.validation import (
    _parse_trace_archival_duration_config,
    _validate_experiment_id,
    _validate_trace_archival_retention_string,
)

_logger = logging.getLogger(__name__)

_TRACE_ARCHIVAL_DURATION_MULTIPLIER_MILLIS = {
    "m": 60 * 1000,
    "h": 60 * 60 * 1000,
    "d": 24 * 60 * 60 * 1000,
}
# Keep grouped experiment scans well below backend parameter limits (notably MSSQL's 2100).
_TRACE_ARCHIVAL_EXPERIMENT_ID_CHUNK_SIZE = 1000


class _ArchiveNowRemainingState(str, Enum):
    DONE = "done"
    ARCHIVABLE = "archivable"
    TRANSIENT = "transient"
    BLOCKED_UNMARKED = "blocked_unmarked"
    TERMINAL_FAILURES_ONLY = "terminal_failures_only"


@dataclass(frozen=True)
class _ArchiveNowRequest:
    older_than_millis: int | None

    @classmethod
    def from_tag_value(cls, value: str | None) -> _ArchiveNowRequest | None:
        if value is None:
            return None

        try:
            older_than = _parse_trace_archival_duration_config(
                value,
                duration_key="older_than",
                allow_missing_duration=True,
            )
            return cls(older_than_millis=_parse_trace_archival_duration_millis(older_than))
        except MlflowException:
            _logger.warning(
                "Ignoring malformed trace archive-now tag value: %r",
                value,
            )
            return None


@dataclass(frozen=True)
class _ArchiveNowCleanupRequest:
    experiment_id: str
    raw_value: str
    parsed_request: _ArchiveNowRequest


@dataclass(frozen=True)
class _TraceArchiveCandidate:
    trace_id: str
    experiment_id: str
    timestamp_ms: int


@dataclass(frozen=True)
class _TraceDeleteSelection:
    trace_id: str
    archived_artifact_uri: str | None = None


@dataclass(frozen=True)
class _TraceSpanSnapshot:
    """Minimal persisted span fields captured while rebuilding trace payloads."""

    content: str
    parent_span_id: int | None
    start_time_unix_nano: int


@dataclass(frozen=True)
class _TraceReadSnapshot:
    """Trace-level metadata plus the DB-backed spans used for a single read/export pass."""

    trace_info: TraceInfo
    spans: list[_TraceSpanSnapshot]


def _parse_trace_archival_duration_millis(value: str | None) -> int | None:
    if value is None:
        return None

    trimmed = _validate_trace_archival_retention_string(value)
    amount = trimmed[:-1]
    unit = trimmed[-1]
    return int(amount) * _TRACE_ARCHIVAL_DURATION_MULTIPLIER_MILLIS[unit]


def _format_trace_archival_duration_millis(value: int | None) -> str | None:
    if value is None:
        return None

    for unit in ("d", "h", "m"):
        multiplier = _TRACE_ARCHIVAL_DURATION_MULTIPLIER_MILLIS[unit]
        if value % multiplier == 0:
            return f"{value // multiplier}{unit}"

    return f"{value // _TRACE_ARCHIVAL_DURATION_MULTIPLIER_MILLIS['m']}m"


def _parse_trace_archival_long_retention_allowlist(value: str | None) -> list[str]:
    if value is None:
        return []

    allowlist = []
    seen = set()
    for raw_experiment_id in value.split(","):
        experiment_id = raw_experiment_id.strip()
        if not experiment_id:
            continue

        _validate_experiment_id(experiment_id)
        if experiment_id not in seen:
            allowlist.append(experiment_id)
            seen.add(experiment_id)

    return allowlist


def _parse_experiment_trace_archival_retention(value: str | None) -> str | None:
    try:
        return _parse_trace_archival_duration_config(
            value,
            duration_key="value",
            expected_type="duration",
        )
    except MlflowException:
        _logger.warning("Ignoring invalid trace archival retention tag value: %r", value)
        return None


def _resolve_effective_trace_archival_retention(
    *,
    experiment_id: str,
    experiment_tags: dict[str, str],
    broader_retention: str,
    long_retention_allowlist: set[str],
) -> str:
    experiment_retention = _parse_experiment_trace_archival_retention(
        experiment_tags.get(TraceExperimentTagKey.ARCHIVAL_RETENTION)
    )
    if experiment_retention is None:
        return broader_retention

    broader_retention_millis = _parse_trace_archival_duration_millis(broader_retention)
    experiment_retention_millis = _parse_trace_archival_duration_millis(experiment_retention)

    if experiment_retention_millis is None or broader_retention_millis is None:
        return broader_retention

    if experiment_retention_millis <= broader_retention_millis:
        return experiment_retention

    if experiment_id in long_retention_allowlist:
        return experiment_retention

    return broader_retention


def _parse_experiment_trace_archival_retention_millis(value: str | None) -> int | None:
    return _parse_trace_archival_duration_millis(_parse_experiment_trace_archival_retention(value))
