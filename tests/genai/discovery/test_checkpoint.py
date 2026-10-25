import json
from unittest import mock

import pytest

from mlflow.entities.experiment import Experiment
from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.genai.discovery.checkpoint import (
    CHECKPOINT_VERSION,
    FIRST_RUN_LOOKBACK_MS,
    STALE_WATERMARK_CAP_MS,
    CheckpointTrace,
    DetectionCheckpoint,
    advance_checkpoint,
    parse_detection_checkpoint,
    read_detection_checkpoint,
    sample_window,
    write_detection_checkpoint,
)
from mlflow.utils.mlflow_tags import MLFLOW_ISSUE_DETECTION_CHECKPOINT

NOW_MS = 1_800_000_000_000
HOUR_MS = 60 * 60 * 1000


def _experiment(tags: dict[str, str]) -> Experiment:
    return Experiment(
        experiment_id="1",
        name="exp",
        artifact_location="loc",
        lifecycle_stage="active",
        tags=[ExperimentTag(key, value) for key, value in tags.items()],
    )


def test_checkpoint_round_trips():
    checkpoint = DetectionCheckpoint(
        last_scanned_trace_time_ms=NOW_MS,
        last_scanned_trace_id="tr-1",
        last_run_id="run-1",
        updated_at_ms=NOW_MS + 5,
    )
    assert parse_detection_checkpoint(checkpoint.to_json()) == checkpoint


def test_serialized_envelope_matches_the_stored_shape():
    serialized = DetectionCheckpoint(
        last_scanned_trace_time_ms=NOW_MS, last_scanned_trace_id="tr-1"
    ).to_json()
    assert json.loads(serialized) == {
        "version": CHECKPOINT_VERSION,
        "lastScannedTraceTimeMs": NOW_MS,
        "lastScannedTraceId": "tr-1",
    }


@pytest.mark.parametrize(
    "raw_value",
    [
        None,
        "",
        "not json",
        "[]",
        '"a string"',
        json.dumps({"lastScannedTraceTimeMs": NOW_MS}),
        json.dumps({"version": CHECKPOINT_VERSION + 1, "lastScannedTraceTimeMs": NOW_MS}),
        json.dumps({"version": str(CHECKPOINT_VERSION), "lastScannedTraceTimeMs": NOW_MS}),
        json.dumps({"version": CHECKPOINT_VERSION}),
        json.dumps({"version": CHECKPOINT_VERSION, "lastScannedTraceTimeMs": "soon"}),
        json.dumps({"version": CHECKPOINT_VERSION, "lastScannedTraceTimeMs": True}),
        '{"version": 1, "lastScannedTraceTimeMs": NaN}',
    ],
)
def test_unusable_envelopes_read_as_no_checkpoint(raw_value: str | None):
    assert parse_detection_checkpoint(raw_value) is None


def test_audit_fields_of_the_wrong_type_do_not_discard_the_watermark():
    parsed = parse_detection_checkpoint(
        json.dumps({
            "version": CHECKPOINT_VERSION,
            "lastScannedTraceTimeMs": NOW_MS,
            "lastScannedTraceId": 7,
            "lastRunUuid": [],
            "updatedAtMs": "yesterday",
        })
    )
    assert parsed == DetectionCheckpoint(last_scanned_trace_time_ms=NOW_MS)


def test_unknown_fields_are_ignored():
    parsed = parse_detection_checkpoint(
        json.dumps({
            "version": CHECKPOINT_VERSION,
            "lastScannedTraceTimeMs": NOW_MS,
            "somethingNewer": {"nested": True},
        })
    )
    assert parsed == DetectionCheckpoint(last_scanned_trace_time_ms=NOW_MS)


def test_read_and_write_use_the_experiment_tag():
    checkpoint = DetectionCheckpoint(last_scanned_trace_time_ms=NOW_MS)
    experiment = _experiment({MLFLOW_ISSUE_DETECTION_CHECKPOINT: checkpoint.to_json()})
    assert read_detection_checkpoint(experiment) == checkpoint
    assert read_detection_checkpoint(_experiment({})) is None

    store = mock.Mock()
    write_detection_checkpoint(store, "1", checkpoint)
    store.set_experiment_tag.assert_called_once()
    experiment_id, tag = store.set_experiment_tag.call_args[0]
    assert experiment_id == "1"
    assert tag.key == MLFLOW_ISSUE_DETECTION_CHECKPOINT
    assert parse_detection_checkpoint(tag.value) == checkpoint


def test_first_run_window_looks_back_one_cadence_interval():
    window = sample_window(None, NOW_MS, FIRST_RUN_LOOKBACK_MS["daily"])
    assert window.lower_bound_ms == NOW_MS - FIRST_RUN_LOOKBACK_MS["daily"]
    assert window.now_ms == NOW_MS


def test_window_starts_at_the_watermark_when_one_exists():
    checkpoint = DetectionCheckpoint(last_scanned_trace_time_ms=NOW_MS - HOUR_MS)
    window = sample_window(checkpoint, NOW_MS, FIRST_RUN_LOOKBACK_MS["weekly"])
    assert window.lower_bound_ms == NOW_MS - HOUR_MS


def test_a_stale_watermark_does_not_widen_the_window():
    checkpoint = DetectionCheckpoint(
        last_scanned_trace_time_ms=NOW_MS - 90 * 24 * HOUR_MS,
    )
    window = sample_window(checkpoint, NOW_MS, FIRST_RUN_LOOKBACK_MS["daily"])
    assert window.lower_bound_ms == NOW_MS - STALE_WATERMARK_CAP_MS


def test_a_healthy_weekly_watermark_is_not_clamped():
    checkpoint = DetectionCheckpoint(last_scanned_trace_time_ms=NOW_MS - 7 * 24 * HOUR_MS)
    window = sample_window(checkpoint, NOW_MS, FIRST_RUN_LOOKBACK_MS["weekly"])
    assert window.lower_bound_ms == checkpoint.last_scanned_trace_time_ms


@pytest.mark.parametrize(
    ("time_ms", "expected"),
    [
        (NOW_MS - HOUR_MS - 1, False),
        (NOW_MS - HOUR_MS, False),
        (NOW_MS - HOUR_MS + 1, True),
        (NOW_MS, True),
        (NOW_MS + 1, False),
    ],
)
def test_window_bounds_are_exclusive_below_and_inclusive_above(time_ms: int, expected: bool):
    window = sample_window(
        DetectionCheckpoint(last_scanned_trace_time_ms=NOW_MS - HOUR_MS),
        NOW_MS,
        FIRST_RUN_LOOKBACK_MS["hourly"],
    )
    assert window.contains(CheckpointTrace(trace_id="tr", time_ms=time_ms)) is expected


def test_window_filter_clause_matches_its_bounds():
    window = sample_window(None, NOW_MS, HOUR_MS)
    assert window.to_filter_clause() == (
        f"timestamp_ms > {NOW_MS - HOUR_MS} AND timestamp_ms <= {NOW_MS}"
    )


def test_checkpoint_advances_to_the_window_newest_trace():
    advanced = advance_checkpoint(
        CheckpointTrace(trace_id="tr-newest", time_ms=NOW_MS - 5),
        run_id="run-1",
        now_ms=NOW_MS,
    )
    assert advanced == DetectionCheckpoint(
        last_scanned_trace_time_ms=NOW_MS - 5,
        last_scanned_trace_id="tr-newest",
        last_run_id="run-1",
        updated_at_ms=NOW_MS,
    )


def test_an_empty_window_does_not_advance_the_checkpoint():
    assert advance_checkpoint(None, run_id="run-1", now_ms=NOW_MS) is None
