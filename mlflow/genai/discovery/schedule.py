"""Storage and timing of an experiment's recurring issue-detection schedule.

A schedule lives in experiment tags, the way the experiment's detection priorities already do:
``mlflow.issueDetection.schedule`` holds the cadence, timezone and paused flag the owner chose,
``mlflow.issueDetection.scheduleTraceFilter`` holds the optional clauses narrowing which traces a
run analyzes, and ``mlflow.issueDetection.scheduleState`` holds what the server has dispatched so
far. The first two are written by whoever edits the schedule; the last is written only by the
server, so an owner saving a new cadence can never clobber the dispatch bookkeeping, and the
server stamping an occurrence can never clobber their edit.

This module owns the storage contract and the timing arithmetic. Dispatching the due occurrence
and running detection are separate.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mlflow.entities.experiment import Experiment
from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.utils.mlflow_tags import (
    MLFLOW_ISSUE_DETECTION_SCHEDULE,
    MLFLOW_ISSUE_DETECTION_SCHEDULE_STATE,
    MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER,
)

_logger = logging.getLogger(__name__)

# Schema version of each envelope below. Versioned in-band (the tag keys are unversioned), so a
# parser reads only an envelope whose ``version`` it knows and ignores any other. Bump on a
# breaking value-shape change and teach the parser the new shape.
SCHEDULE_VERSION = 1

DEFAULT_TIMEZONE = "UTC"

# Upper bound on how many clauses one schedule may carry. The tag is written by a client, so the
# dispatcher reads it defensively rather than trusting its size; 20 is far more than the modal
# offers and still bounds the filter string a run builds.
MAX_TRACE_FILTERS = 20

# A clause is spliced into a ``search_traces`` filter string, where a backtick or quote would
# either break the expression or change which traces it matches. Such a clause is dropped rather
# than escaped: the filter grammar has no documented escape, and a run that silently analyzed the
# wrong traces would be worse than one that ignored a clause nobody can express.
_FORBIDDEN_FILTER_CHARS = "`'\"\\"


class DetectionCadence(str, Enum):
    """How often a schedule runs. ``off`` is not a cadence -- it is the absence of a schedule."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"


# When each cadence fires, in the schedule's own timezone: hourly on the hour, daily at 09:00, and
# weekly on Monday at 09:00.
_FIRE_HOUR = 9
_MONDAY = 0


@dataclass(frozen=True)
class ScheduleTraceFilter:
    """
    One equality clause narrowing which traces a scheduled run analyzes.

    A schedule may carry several, ANDed: a run only analyzes traces matching every clause.
    ``field`` says which namespace the key lives in; ``operator`` is fixed to ``=`` today but
    stored so the shape can carry other comparisons later without a migration.
    """

    field: Literal["metadata", "tag"]
    key: str
    value: str
    operator: Literal["="] = "="

    def to_filter_clause(self) -> str:
        namespace = "request_metadata" if self.field == "metadata" else "tags"
        return f"{namespace}.`{self.key}` {self.operator} '{self.value}'"

    def to_dict(self) -> dict[str, str]:
        return {
            "field": self.field,
            "operator": self.operator,
            "key": self.key,
            "value": self.value,
        }


@dataclass(frozen=True)
class DetectionSchedule:
    """
    An experiment's recurring detection schedule, as stored.

    A paused schedule keeps its cadence and filters so resuming restores them; it simply does not
    fire. "Off" in the UI is a paused schedule, not a deleted one.
    """

    cadence: DetectionCadence
    timezone: str = DEFAULT_TIMEZONE
    paused: bool = False
    trace_filters: tuple[ScheduleTraceFilter, ...] = ()

    def to_json(self) -> str:
        """Serialize the cadence tag. The filters live in their own tag."""
        return json.dumps({
            "version": SCHEDULE_VERSION,
            "cadence": self.cadence.value,
            "timezone": self.timezone,
            "paused": self.paused,
        })

    def trace_filters_to_json(self) -> str:
        return json.dumps([f.to_dict() for f in self.trace_filters])

    def to_filter_string(self) -> str:
        """The clauses as one ANDed ``search_traces`` expression, or ``""`` when there are none."""
        return " AND ".join(f.to_filter_clause() for f in self.trace_filters)


@dataclass(frozen=True)
class DetectionScheduleState:
    """
    What the server has dispatched for a schedule. Written only by the server.

    Args:
        last_fire_time_ms: Fire time (epoch ms) of the occurrence last dispatched. The next
            occurrence is the first fire time strictly after it, so this both prevents the same
            occurrence running twice and bounds catch-up after downtime to one run.
        last_dispatched_at_ms: Wall-clock time the dispatch happened (audit only). Differs from
            ``last_fire_time_ms`` when the server was busy or down at the fire time.
        last_run_id: The detection run the last dispatch created (audit only).
    """

    last_fire_time_ms: int
    last_dispatched_at_ms: int | None = None
    last_run_id: str | None = None

    def to_json(self) -> str:
        payload: dict[str, object] = {
            "version": SCHEDULE_VERSION,
            "lastFireTimeMs": self.last_fire_time_ms,
        }
        if self.last_dispatched_at_ms is not None:
            payload["lastDispatchedAtMs"] = self.last_dispatched_at_ms
        if self.last_run_id is not None:
            payload["lastRunId"] = self.last_run_id
        return json.dumps(payload)


def _parse_envelope(raw_value: str | None) -> dict[str, Any] | None:
    if not raw_value:
        return None
    try:
        parsed = json.loads(raw_value)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict) or parsed.get("version") != SCHEDULE_VERSION:
        return None
    return parsed


def _parse_trace_filter(entry: object) -> ScheduleTraceFilter | None:
    if not isinstance(entry, dict):
        return None
    field = entry.get("field")
    key = entry.get("key")
    value = entry.get("value")
    if field not in ("metadata", "tag"):
        return None
    if not isinstance(key, str) or not isinstance(value, str) or not key:
        return None
    # Only equality exists today. An unknown operator is dropped rather than coerced to "=", which
    # would analyze a wider set of traces than the clause asked for.
    if entry.get("operator", "=") != "=":
        return None
    if any(char in key or char in value for char in _FORBIDDEN_FILTER_CHARS):
        return None
    return ScheduleTraceFilter(field=field, key=key, value=value)


def parse_trace_filters(raw_value: str | None) -> tuple[ScheduleTraceFilter, ...]:
    """
    Parse the trace-filter tag, dropping any clause this reader cannot use.

    Dropping one clause widens what a run analyzes, so each is reported. Returning the
    clauses that did parse keeps a schedule running on the rest rather than silently
    analyzing the whole experiment or stopping.
    """
    if not raw_value:
        return ()
    try:
        parsed = json.loads(raw_value)
    except (TypeError, ValueError):
        _logger.warning("Ignoring an unreadable issue-detection schedule trace filter tag")
        return ()
    if not isinstance(parsed, list):
        return ()
    filters = []
    for entry in parsed[:MAX_TRACE_FILTERS]:
        if (parsed_filter := _parse_trace_filter(entry)) is None:
            _logger.warning(
                "Ignoring an issue-detection schedule trace filter clause this server cannot "
                "apply; the scheduled run will analyze traces the clause meant to exclude"
            )
        else:
            filters.append(parsed_filter)
    if len(parsed) > MAX_TRACE_FILTERS:
        _logger.warning(
            "An issue-detection schedule carries %d trace filter clauses; applying the first %d",
            len(parsed),
            MAX_TRACE_FILTERS,
        )
    return tuple(filters)


def _resolve_timezone(name: object) -> str:
    """
    The schedule's timezone, falling back to UTC when it names a zone this server cannot load.

    A zone the server's tzdata does not carry must not stop the schedule: UTC still fires it on
    cadence, only at a different local hour, which is a better outcome than silence.
    """
    if not isinstance(name, str) or not name:
        return DEFAULT_TIMEZONE
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        _logger.warning(
            "Issue-detection schedule names timezone %r, which this server cannot load; "
            "firing it in %s instead",
            name,
            DEFAULT_TIMEZONE,
        )
        return DEFAULT_TIMEZONE
    return name


def parse_detection_schedule(
    raw_value: str | None, raw_trace_filters: str | None = None
) -> DetectionSchedule | None:
    """
    Parse the schedule tags.

    Returns ``None`` when there is no schedule, when the envelope carries an unrecognized
    ``version``, or when it names a cadence this server does not run -- all of which mean "do not
    fire" rather than "fire with a guess".
    """
    if (parsed := _parse_envelope(raw_value)) is None:
        return None
    try:
        cadence = DetectionCadence(parsed.get("cadence"))
    except ValueError:
        _logger.warning(
            "Ignoring an issue-detection schedule with unrecognized cadence %r",
            parsed.get("cadence"),
        )
        return None
    return DetectionSchedule(
        cadence=cadence,
        timezone=_resolve_timezone(parsed.get("timezone")),
        paused=parsed.get("paused") is True,
        trace_filters=parse_trace_filters(raw_trace_filters),
    )


def parse_detection_schedule_state(raw_value: str | None) -> DetectionScheduleState | None:
    if (parsed := _parse_envelope(raw_value)) is None:
        return None
    last_fire_time_ms = parsed.get("lastFireTimeMs")
    if isinstance(last_fire_time_ms, bool) or not isinstance(last_fire_time_ms, int):
        return None
    last_dispatched_at_ms = parsed.get("lastDispatchedAtMs")
    return DetectionScheduleState(
        last_fire_time_ms=last_fire_time_ms,
        last_dispatched_at_ms=last_dispatched_at_ms
        if isinstance(last_dispatched_at_ms, int) and not isinstance(last_dispatched_at_ms, bool)
        else None,
        last_run_id=parsed.get("lastRunId") if isinstance(parsed.get("lastRunId"), str) else None,
    )


def read_detection_schedule(experiment: Experiment) -> DetectionSchedule | None:
    return parse_detection_schedule(
        experiment.tags.get(MLFLOW_ISSUE_DETECTION_SCHEDULE),
        experiment.tags.get(MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER),
    )


def read_detection_schedule_state(experiment: Experiment) -> DetectionScheduleState | None:
    return parse_detection_schedule_state(
        experiment.tags.get(MLFLOW_ISSUE_DETECTION_SCHEDULE_STATE)
    )


def write_detection_schedule(
    store: AbstractStore, experiment_id: str, schedule: DetectionSchedule
) -> None:
    """Persist the cadence and filter tags. Both are rewritten so they stay consistent."""
    store.set_experiment_tag(
        experiment_id, ExperimentTag(MLFLOW_ISSUE_DETECTION_SCHEDULE, schedule.to_json())
    )
    store.set_experiment_tag(
        experiment_id,
        ExperimentTag(
            MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER, schedule.trace_filters_to_json()
        ),
    )


def write_detection_schedule_state(
    store: AbstractStore, experiment_id: str, state: DetectionScheduleState
) -> None:
    store.set_experiment_tag(
        experiment_id, ExperimentTag(MLFLOW_ISSUE_DETECTION_SCHEDULE_STATE, state.to_json())
    )


def next_fire_time_ms(schedule: DetectionSchedule, after_ms: int) -> int:
    """
    The first time this schedule's cadence fires, strictly after ``after_ms`` (epoch ms).

    Candidates are built as local wall-clock times and then resolved to an instant, so the hour a
    schedule fires at stays the same local hour across a DST change rather than drifting by an
    hour. ``zoneinfo`` resolves an ambiguous wall clock (the repeated hour when clocks go back) to
    its first occurrence, and a nonexistent one (the skipped hour when they go forward) to the
    instant the gap moves it to, so every candidate maps to exactly one instant.
    """
    zone = ZoneInfo(schedule.timezone)
    local = datetime.fromtimestamp(after_ms / 1000, tz=zone)
    naive = local.replace(tzinfo=None)

    if schedule.cadence is DetectionCadence.HOURLY:
        base = naive.replace(minute=0, second=0, microsecond=0)
        step = timedelta(hours=1)
    elif schedule.cadence is DetectionCadence.DAILY:
        base = naive.replace(hour=_FIRE_HOUR, minute=0, second=0, microsecond=0)
        step = timedelta(days=1)
    else:
        base = naive.replace(hour=_FIRE_HOUR, minute=0, second=0, microsecond=0) - timedelta(
            days=(naive.weekday() - _MONDAY) % 7
        )
        step = timedelta(weeks=1)

    # Walking candidates in wall-clock order, the first one past ``after_ms`` wins. A handful of
    # steps covers it: the base candidate is at most one interval behind, and a DST shift can only
    # move an instant by an hour or two, never past the next interval.
    for index in range(4):
        candidate = (base + step * index).replace(tzinfo=zone)
        candidate_ms = int(candidate.timestamp() * 1000)
        if candidate_ms > after_ms:
            return candidate_ms
    raise ValueError(
        f"Could not find the next {schedule.cadence.value} fire time after {after_ms} "
        f"in {schedule.timezone}"
    )


def due_occurrence_ms(
    schedule: DetectionSchedule,
    state: DetectionScheduleState | None,
    now_ms: int,
) -> int | None:
    """
    The fire time of the occurrence that should run now, or ``None`` when none is due.

    A schedule the server has not seen before gets no immediate run: its first occurrence is the
    next fire time after the server noticed it, the same as a cron would do. That is why a
    first pass records state without dispatching -- see ``claim_first_occurrence``.

    Only one occurrence is ever returned. After downtime that spans several fire times, the
    schedule resumes with a single run rather than a backlog, and how much history that run may
    reach back over is bounded separately by the sampling window's stale-watermark cap.
    """
    if schedule.paused or state is None:
        return None
    fire_time_ms = next_fire_time_ms(schedule, state.last_fire_time_ms)
    return fire_time_ms if fire_time_ms <= now_ms else None


def claim_first_occurrence(now_ms: int) -> DetectionScheduleState:
    """
    The state to record for a schedule the server has not dispatched before.

    Anchoring at ``now_ms`` makes the first run happen at the schedule's next fire time rather
    than immediately, so saving a daily schedule in the afternoon does not run it on the spot.
    """
    return DetectionScheduleState(last_fire_time_ms=now_ms, last_dispatched_at_ms=None)


def claim_occurrence(
    state: DetectionScheduleState, fire_time_ms: int, now_ms: int | None = None
) -> DetectionScheduleState:
    """The state to record when dispatching the occurrence that fired at ``fire_time_ms``."""
    return replace(
        state,
        last_fire_time_ms=fire_time_ms,
        last_dispatched_at_ms=now_ms if now_ms is not None else int(time.time() * 1000),
    )


def scheduled_detection_available() -> bool:
    """
    Whether this server can run a schedule at all.

    Recurring detection needs the periodic-task dispatcher and the job system behind it, which in
    turn need a SQL backend store. Without them a saved schedule would sit there looking active
    and never run, so the capability is reported rather than inferred by a client.
    """
    from mlflow.environment_variables import (
        MLFLOW_SERVER_ENABLE_JOB_EXECUTION,
        MLFLOW_SERVER_JOB_ENABLE_PERIODIC_TASKS,
    )
    from mlflow.server.jobs.utils import _check_requirements

    if not MLFLOW_SERVER_ENABLE_JOB_EXECUTION.get():
        return False
    if not MLFLOW_SERVER_JOB_ENABLE_PERIODIC_TASKS.get():
        return False
    try:
        _check_requirements()
    except Exception:
        _logger.debug(
            "Scheduled issue detection is unavailable because the job requirements are not met",
            exc_info=True,
        )
        return False
    return True
