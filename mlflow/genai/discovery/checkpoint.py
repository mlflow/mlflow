"""Watermark of the traces scheduled issue detection has already covered."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

from mlflow.entities.experiment import Experiment
from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.utils.mlflow_tags import MLFLOW_ISSUE_DETECTION_CHECKPOINT

# Schema version of the envelope stored in the ``mlflow.issueDetection.checkpoint`` experiment
# tag. The envelope is versioned in-band (the tag key is unversioned), so the parser reads only
# an envelope whose ``version`` equals this and ignores any it cannot interpret. Bump this on a
# breaking value-shape change and teach the parser to recognize the new shape.
CHECKPOINT_VERSION = 1

# Cap on how many traces one scheduled run samples from its window.
MAX_SCHEDULED_SAMPLE_SIZE = 100

# Cold-start lookback per cadence, used only when there is no usable checkpoint (first run, or an
# envelope this reader ignored): the window's lower bound is ``now - lookback``, one cadence
# interval, so a fresh schedule starts from "the last interval" rather than the whole history.
FIRST_RUN_LOOKBACK_MS = {
    "hourly": 60 * 60 * 1000,
    "daily": 24 * 60 * 60 * 1000,
    "weekly": 7 * 24 * 60 * 60 * 1000,
}

# Cap on how far back a run may reach when its watermark is stale -- the schedule was paused for a
# long time, say. The lower bound is ``max(watermark, now - cap)``, so a resumed schedule never
# rescans a huge, mostly-stale backlog. Fixed rather than cadence-derived and independent of the
# cold-start lookback: it bounds the between-run window, so a single run covers at most this much
# history even after a long gap. Traces older than the cap that no run covered are skipped.
#
# 8 days = the 7-day weekly interval plus ~1 day of DST and scheduler-jitter headroom, so a healthy
# weekly run (watermark ~7 days old) is never clamped at the boundary; the cap only clips a
# genuinely long-paused schedule. Comfortably larger than the hourly and daily intervals too, so
# their behavior is unchanged.
STALE_WATERMARK_CAP_MS = 8 * 24 * 60 * 60 * 1000


@dataclass
class DetectionCheckpoint:
    """
    Watermark of the last trace scheduled issue detection has covered for an experiment.

    Only the scheduled sampling path reads or writes it; manual detection runs never touch it.

    Args:
        last_scanned_trace_time_ms: The watermark -- the newest trace request time (epoch ms) in
            the last run's sampling window. The next run's window is ``(this, now]``, so this
            marks the boundary already covered. A timestamp, since a trace ID alone has no order.
        last_scanned_trace_id: ID of the newest trace at ``last_scanned_trace_time_ms``. Audit
            only; the window's lower bound is the timestamp, not this ID.
        last_run_id: Detection run that advanced the watermark (audit only).
        updated_at_ms: Wall-clock time (epoch ms) the tag was last written (audit only).
    """

    last_scanned_trace_time_ms: int
    last_scanned_trace_id: str | None = None
    last_run_id: str | None = None
    updated_at_ms: int | None = None

    def to_json(self) -> str:
        payload = {
            "version": CHECKPOINT_VERSION,
            "lastScannedTraceTimeMs": self.last_scanned_trace_time_ms,
        }
        if self.last_scanned_trace_id is not None:
            payload["lastScannedTraceId"] = self.last_scanned_trace_id
        if self.last_run_id is not None:
            payload["lastRunUuid"] = self.last_run_id
        if self.updated_at_ms is not None:
            payload["updatedAtMs"] = self.updated_at_ms
        return json.dumps(payload)


@dataclass
class CheckpointTrace:
    """The minimal trace shape the checkpoint logic needs: an ID and an epoch-ms time."""

    trace_id: str
    time_ms: int


def _finite_int(value: object) -> int | None:
    # ``bool`` is an ``int`` subclass, and a JSON ``true`` is not a timestamp. ``json.loads``
    # accepts the ``NaN``/``Infinity`` literals, so a non-finite number has to be rejected too.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if math.isfinite(value) else None


def parse_detection_checkpoint(raw_value: str | None) -> DetectionCheckpoint | None:
    """
    Parse a raw tag value into a checkpoint.

    Returns ``None`` when the tag is absent, carries an unrecognized ``version``, or lacks a
    usable watermark. Strict on the two fields that must be trustworthy -- the in-band
    ``version``, so a foreign or future envelope is never misread as the current shape, and
    ``lastScannedTraceTimeMs``, which drives trace selection -- and lenient on the optional
    audit fields.
    """
    if not raw_value:
        return None
    try:
        parsed = json.loads(raw_value)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    if parsed.get("version") != CHECKPOINT_VERSION:
        return None
    if (watermark := _finite_int(parsed.get("lastScannedTraceTimeMs"))) is None:
        return None
    return DetectionCheckpoint(
        last_scanned_trace_time_ms=watermark,
        last_scanned_trace_id=parsed.get("lastScannedTraceId")
        if isinstance(parsed.get("lastScannedTraceId"), str)
        else None,
        last_run_id=parsed.get("lastRunUuid")
        if isinstance(parsed.get("lastRunUuid"), str)
        else None,
        updated_at_ms=_finite_int(parsed.get("updatedAtMs")),
    )


def read_detection_checkpoint(experiment: Experiment) -> DetectionCheckpoint | None:
    return parse_detection_checkpoint(experiment.tags.get(MLFLOW_ISSUE_DETECTION_CHECKPOINT))


def write_detection_checkpoint(
    store: AbstractStore, experiment_id: str, checkpoint: DetectionCheckpoint
) -> None:
    """Persist the checkpoint tag. Only the scheduled sampling path calls this."""
    store.set_experiment_tag(
        experiment_id,
        ExperimentTag(MLFLOW_ISSUE_DETECTION_CHECKPOINT, checkpoint.to_json()),
    )


@dataclass
class SampleWindow:
    """
    The window a scheduled run samples from: ``(lower_bound_ms, now_ms]``.

    Lower bound exclusive, upper bound inclusive.

    Args:
        lower_bound_ms: Exclusive lower bound. With a checkpoint this is the watermark, clamped
            so it never reaches further back than ``now_ms - STALE_WATERMARK_CAP_MS``; with no
            checkpoint it is the cold-start bound ``now_ms - first_run_lookback_ms``.
        now_ms: Inclusive upper bound -- the time the window was computed.
    """

    lower_bound_ms: int
    now_ms: int

    def contains(self, trace: CheckpointTrace) -> bool:
        return self.lower_bound_ms < trace.time_ms <= self.now_ms

    def to_filter_clause(self) -> str:
        """
        The ``timestamp_ms`` clause bounding this window for ``search_traces``.

        The run then randomly samples within the window, so this clause bounds what it draws
        from rather than selecting the sample itself.
        """
        return f"timestamp_ms > {self.lower_bound_ms} AND timestamp_ms <= {self.now_ms}"


def sample_window(
    checkpoint: DetectionCheckpoint | None,
    now_ms: int,
    first_run_lookback_ms: int,
    stale_watermark_cap_ms: int = STALE_WATERMARK_CAP_MS,
) -> SampleWindow:
    """
    The window a run draws its random sample from.

    The two lower bounds are independent: the cold-start lookback tracks the cadence, while the
    stale cap is fixed and only applies once a watermark exists.
    """
    lower_bound_ms = (
        max(checkpoint.last_scanned_trace_time_ms, now_ms - stale_watermark_cap_ms)
        if checkpoint
        else now_ms - first_run_lookback_ms
    )
    return SampleWindow(lower_bound_ms=lower_bound_ms, now_ms=now_ms)


def advance_checkpoint(
    window_newest: CheckpointTrace | None,
    *,
    run_id: str | None = None,
    now_ms: int | None = None,
) -> DetectionCheckpoint | None:
    """
    The checkpoint to persist after a run.

    The watermark advances to the window's *newest* trace, so the next run's window starts
    strictly after it. That is the newest trace in the window, not the newest trace the run
    sampled: traces the run did not sample are intentionally not carried forward, since the
    window is what the run is entitled to call covered.

    Returns ``None`` when the window held no trace, signalling "do not advance".
    """
    if window_newest is None:
        return None
    return DetectionCheckpoint(
        last_scanned_trace_time_ms=window_newest.time_ms,
        last_scanned_trace_id=window_newest.trace_id,
        last_run_id=run_id,
        updated_at_ms=now_ms if now_ms is not None else int(time.time() * 1000),
    )
