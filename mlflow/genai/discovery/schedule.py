"""Storage and timing of an experiment's recurring issue-detection schedule.

A schedule lives in experiment tags, the way the experiment's detection priorities already do:
``mlflow.issueDetection.schedule`` holds the cadence, timezone and paused flag,
``mlflow.issueDetection.scheduleTraceFilter`` the optional clauses narrowing which traces a run
analyzes, and ``mlflow.issueDetection.scheduleState`` what the server has dispatched. The first
two are written by whoever edits the schedule, the last only by the server, so neither can clobber
the other.

This module owns the storage contract and the timing arithmetic; dispatching and running are
separate.
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

# Upper bound on how many clauses one schedule may carry. The tag is client-written, so its size
# is read defensively; 20 is far more than the modal offers.
MAX_TRACE_FILTERS = 20

# A clause's key and value are spliced into a ``search_traces`` filter string, so a character that
# would break the expression or change which traces it matches makes the clause unusable. Such a
# clause is dropped rather than escaped or rewritten: the filter grammar has no escape (doubling a
# quote yields both characters, not one), and a run that silently analyzed the wrong traces would
# be worse than one that ignored a clause nobody can express.
_FORBIDDEN_FILTER_CHARS = "`'\"\\"


def _has_control_char(text: str) -> bool:
    """
    Whether ``text`` carries a character that cannot appear in a filter clause.

    Control characters and DEL build a malformed clause. Ordinary whitespace does not, unlike in a
    model identifier -- a tag value like "north america" is legitimate.
    """
    return any(ord(char) < 0x20 or ord(char) == 0x7F for char in text)


# Upper bound on a stored provider, model or endpoint name. The tag is client-written, so the
# dispatcher bounds what it will splice into a model URI rather than trusting its length.
MAX_MODEL_IDENTIFIER_LENGTH = 200


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

    A schedule may carry several, ANDed. ``field`` says which namespace the key lives in;
    ``operator`` is fixed to ``=`` today but stored so other comparisons need no migration.
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
class ScheduleModel:
    """
    The model a scheduled run analyzes with, as chosen in the schedule modal.

    Either an AI Gateway endpoint (``endpoint_name``) or a provider's model directly (``provider``
    + ``model``). A schedule saved before this was offered carries none, reads back as ``None``,
    and stays on the server's default -- the behaviour it had when it was saved.

    A direct selection also records ``secret_id``: *which* stored credential the owner chose, not
    the credential itself. The alternative -- resolving a key from the provider at each fire time
    -- survives key rotation but picks whatever secret happens to be newest for that provider,
    which on a multi-user server is a key somebody else saved for their own use. Naming the secret
    is what keeps an occurrence on the credential its owner actually designated; an occurrence
    whose secret has since been deleted fails and says so, rather than silently substituting one.

    Args:
        provider: Provider id for a direct selection, e.g. ``openai``.
        model: Model name for a direct selection.
        endpoint_name: AI Gateway endpoint name, which carries its own credential server-side.
        secret_id: The stored credential a direct selection runs with. Empty on an endpoint
            selection, which needs none, and on a schedule saved before this was recorded -- that
            one falls back to the server's own environment.
    """

    provider: str = ""
    model: str = ""
    endpoint_name: str = ""
    secret_id: str = ""

    def __post_init__(self) -> None:
        # Every field defaulting to "" keeps the two selections independent, but it also makes
        # ``ScheduleModel()`` constructable into the URI ":/", which names no model. The parser
        # already returns None rather than building one, so this guards a future caller.
        if not self.endpoint_name and not (self.provider and self.model):
            raise ValueError(
                "A ScheduleModel names either an AI Gateway endpoint or a provider and model"
            )

    def to_model_uri(self) -> str:
        """The model identifier the job takes, in the same form the manual path builds."""
        if self.endpoint_name:
            return f"gateway:/{self.endpoint_name}"
        return f"{self.provider.lower()}:/{self.model}"

    def to_dict(self) -> dict[str, str]:
        if self.endpoint_name:
            return {"endpointName": self.endpoint_name}
        stored = {"provider": self.provider, "model": self.model}
        if self.secret_id:
            stored["secretId"] = self.secret_id
        return stored


@dataclass(frozen=True)
class DetectionSchedule:
    """
    An experiment's recurring detection schedule, as stored.

    "Off" in the UI is a paused schedule rather than a deleted one: the cadence and filters stay
    stored so resuming restores them.
    """

    cadence: DetectionCadence
    timezone: str = DEFAULT_TIMEZONE
    paused: bool = False
    trace_filters: tuple[ScheduleTraceFilter, ...] = ()
    model: ScheduleModel | None = None

    def to_json(self) -> str:
        """Serialize the cadence tag. The filters live in their own tag."""
        payload: dict[str, Any] = {
            "version": SCHEDULE_VERSION,
            "cadence": self.cadence.value,
            "timezone": self.timezone,
            "paused": self.paused,
        }
        if self.model is not None:
            payload["model"] = self.model.to_dict()
        return json.dumps(payload)

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
            occurrence is the first fire time strictly after it -- see ``due_occurrence_ms``.
        last_dispatched_at_ms: Wall-clock dispatch time (audit only). Differs from
            ``last_fire_time_ms`` when the server was busy or down at the fire time.
        last_run_id: The run that dispatch created (audit only).
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
    # Anything that is not "tag" reads as request metadata, as it does in the tab, so a clause
    # written before this field existed is applied rather than dropped.
    field: Literal["metadata", "tag"] = "tag" if entry.get("field") == "tag" else "metadata"
    key = entry.get("key")
    if not isinstance(key, str):
        return None
    # Trimmed, since a padded key would build a filter matching nothing; blank once trimmed is
    # no clause at all.
    key = key.strip()
    if not key:
        return None
    # A missing or non-text value reads as the empty string, matching the tab, rather than
    # dropping the clause.
    raw_value = entry.get("value")
    value = raw_value if isinstance(raw_value, str) else ""
    # An unknown operator is dropped rather than coerced to "=": coercing "!=" would analyze a
    # set disjoint from the one the clause asked for, hiding the traces it meant to select.
    if entry.get("operator", "=") != "=":
        return None
    if any(char in key or char in value for char in _FORBIDDEN_FILTER_CHARS):
        return None
    if _has_control_char(key) or _has_control_char(value):
        return None
    return ScheduleTraceFilter(field=field, key=key, value=value)


def _parse_model_identifier(value: object) -> str:
    """
    One stored provider / model / endpoint name, or ``""`` when it is not usable.

    Interior whitespace and control characters are rejected rather than stripped: the value
    becomes a model URI, and rewriting it could point a run at a different model.
    """
    if not isinstance(value, str):
        return ""
    stripped = value.strip()
    if not stripped or len(stripped) > MAX_MODEL_IDENTIFIER_LENGTH:
        return ""
    if any(char.isspace() or ord(char) < 0x20 for char in stripped):
        return ""
    return stripped


def _parse_model(entry: object) -> ScheduleModel | None:
    """
    Parse the model half of the schedule envelope.

    Returns ``None`` when there is no usable selection, leaving the run on the server's default
    rather than failing the occurrence -- firing with the default beats stopping.
    """
    if not isinstance(entry, dict):
        return None
    if endpoint_name := _parse_model_identifier(entry.get("endpointName")):
        return ScheduleModel(endpoint_name=endpoint_name)
    provider = _parse_model_identifier(entry.get("provider"))
    model = _parse_model_identifier(entry.get("model"))
    if not provider or not model:
        _logger.warning(
            "Ignoring an unusable model selection on an issue-detection schedule; the scheduled "
            "run will use this server's default detection model"
        )
        return None
    # A secret id is a stored reference, not a name spliced into a URI, so it is checked for
    # usability but not length-bounded the way an identifier is.
    secret_id = entry.get("secretId")
    return ScheduleModel(
        provider=provider,
        model=model,
        secret_id=secret_id if isinstance(secret_id, str) and secret_id.strip() else "",
    )


def parse_trace_filters(raw_value: str | None) -> tuple[ScheduleTraceFilter, ...]:
    """
    Parse the trace-filter tag, dropping any clause this reader cannot use.

    Dropping a clause widens what a run analyzes, so each is logged. The clauses that did parse
    are still returned, which keeps the schedule narrowed by the rest rather than not at all.
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

    A zone missing from the server's tzdata must not stop the schedule: UTC still fires it on
    cadence, only at a different local hour.
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

    Returns ``None`` when there is no schedule, the envelope carries an unrecognized ``version``,
    or it names a cadence this server does not run -- each meaning "do not fire", not "guess".
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
        model=_parse_model(parsed.get("model")),
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
    """
    Persist the cadence and filter tags. Both are rewritten so they stay consistent.

    The filters are written **before** the cadence, matching the tab's writer, and the order is
    the safety property rather than a detail: the cadence tag is what makes the dispatcher pick
    the schedule up, so writing it first leaves a window -- a failed second write, or a poll
    landing between the two -- where a run could analyze traces the new clauses meant to exclude.
    This way a narrowed scope is in force before anything runs under it, and a failure before the
    cadence write leaves the schedule as it was.

    The two writes are not atomic, so a failure between them can leave the stored filters ahead of
    the cadence. Both are rewritten on every save, and that is the safe direction to fail in.
    """
    store.set_experiment_tag(
        experiment_id,
        ExperimentTag(
            MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER, schedule.trace_filters_to_json()
        ),
    )
    store.set_experiment_tag(
        experiment_id, ExperimentTag(MLFLOW_ISSUE_DETECTION_SCHEDULE, schedule.to_json())
    )


def write_detection_schedule_state(
    store: AbstractStore, experiment_id: str, state: DetectionScheduleState
) -> None:
    store.set_experiment_tag(
        experiment_id, ExperimentTag(MLFLOW_ISSUE_DETECTION_SCHEDULE_STATE, state.to_json())
    )


def _fire_time_anchor(
    schedule: DetectionSchedule, reference_ms: int
) -> tuple[ZoneInfo, datetime, timedelta]:
    """
    The candidate anchor and interval for ``schedule``, as a naive wall clock in its timezone.

    Kept naive on purpose: candidates are built in wall-clock terms and only then resolved to an
    instant, so the local hour a schedule fires at survives a DST change instead of drifting.
    """
    zone = ZoneInfo(schedule.timezone)
    naive = datetime.fromtimestamp(reference_ms / 1000, tz=zone).replace(tzinfo=None)
    if schedule.cadence is DetectionCadence.HOURLY:
        return zone, naive.replace(minute=0, second=0, microsecond=0), timedelta(hours=1)
    if schedule.cadence is DetectionCadence.DAILY:
        return (
            zone,
            naive.replace(hour=_FIRE_HOUR, minute=0, second=0, microsecond=0),
            timedelta(days=1),
        )
    monday = naive.replace(hour=_FIRE_HOUR, minute=0, second=0, microsecond=0) - timedelta(
        days=(naive.weekday() - _MONDAY) % 7
    )
    return zone, monday, timedelta(weeks=1)


def next_fire_time_ms(schedule: DetectionSchedule, after_ms: int) -> int:
    """
    The first time this schedule's cadence fires, strictly after ``after_ms`` (epoch ms).

    ``zoneinfo`` resolves each wall-clock candidate (see ``_fire_time_anchor``) to exactly one
    instant: an ambiguous one, the repeated hour when clocks go back, to its first occurrence, and
    a nonexistent one, the hour skipped when they go forward, to the instant the gap moves it to.
    """
    zone, base, step = _fire_time_anchor(schedule, after_ms)
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


def _latest_fire_time_ms(schedule: DetectionSchedule, at_or_before_ms: int) -> int | None:
    """
    The most recent time this schedule fires at or before ``at_or_before_ms``, if there is one.

    Candidates are filtered rather than assumed to be in increasing instant order, since a DST
    shift can move one by an hour either way.
    """
    zone, base, step = _fire_time_anchor(schedule, at_or_before_ms)
    due = [
        candidate_ms
        for index in range(1, -3, -1)
        if (candidate_ms := int((base + step * index).replace(tzinfo=zone).timestamp() * 1000))
        <= at_or_before_ms
    ]
    return max(due) if due else None


def due_occurrence_ms(
    schedule: DetectionSchedule,
    state: DetectionScheduleState | None,
    now_ms: int,
) -> int | None:
    """
    The fire time of the occurrence that should run now, or ``None`` when none is due. A
    schedule with no state yet is never due -- see ``claim_first_occurrence``.

    Only one occurrence is ever returned, and it is the most recent one due rather than the oldest
    missed. That is what keeps downtime to a single catch-up run: claiming the oldest would
    advance the state only to that historical fire time, leaving the next poll to claim the one
    after it, so a month-long pause would dispatch a month of hourly runs. How much history that
    one run reads is bounded separately by ``STALE_WATERMARK_CAP_MS``, measured from this fire
    time -- near now, not near the start of the gap.
    """
    if schedule.paused or state is None:
        return None
    fire_time_ms = next_fire_time_ms(schedule, state.last_fire_time_ms)
    if fire_time_ms > now_ms:
        return None
    latest_ms = _latest_fire_time_ms(schedule, now_ms)
    return fire_time_ms if latest_ms is None else max(fire_time_ms, latest_ms)


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

    Recurring detection needs the periodic-task dispatcher, the job system behind it, and the SQL
    backend store both need. Reported rather than left for a client to infer, since a saved
    schedule would otherwise look active and never run.

    The ``executor`` job-execution engine is reported unavailable: it rejects the environment a
    submission carries, which is how a scheduled run receives the credential for its model, so a
    schedule naming a provider could never authenticate there. This is broader than it has to be
    -- an AI Gateway endpoint needs no credential delivered and would run fine -- but the answer
    is per-server and given before anyone has chosen a model. Offering the control and failing
    every occurrence afterwards is the worse of the two.
    """
    from mlflow.environment_variables import (
        MLFLOW_SERVER_ENABLE_JOB_EXECUTION,
        MLFLOW_SERVER_JOB_ENABLE_PERIODIC_TASKS,
    )
    from mlflow.server.jobs.utils import _check_requirements, get_job_execution_engine

    if not MLFLOW_SERVER_ENABLE_JOB_EXECUTION.get():
        return False
    if not MLFLOW_SERVER_JOB_ENABLE_PERIODIC_TASKS.get():
        return False
    if get_job_execution_engine() == "executor":
        _logger.debug(
            "Scheduled issue detection is unavailable because the executor job-execution engine "
            "cannot carry the credential a scheduled run needs"
        )
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
