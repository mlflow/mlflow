from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from mlflow.exceptions import MlflowException
from mlflow.utils.validation import (
    _parse_trace_archival_duration_config,
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


def _parse_trace_archival_duration_millis(value: str | None) -> int | None:
    if value is None:
        return None

    trimmed = _validate_trace_archival_retention_string(value)
    amount = trimmed[:-1]
    unit = trimmed[-1]
    return int(amount) * _TRACE_ARCHIVAL_DURATION_MULTIPLIER_MILLIS[unit]


def _parse_experiment_trace_archival_retention_millis(value: str | None) -> int | None:
    try:
        duration = _parse_trace_archival_duration_config(
            value,
            duration_key="value",
            expected_type="duration",
        )
        return _parse_trace_archival_duration_millis(duration)
    except MlflowException:
        _logger.warning("Ignoring invalid trace archival retention tag value: %r", value)
        return None
