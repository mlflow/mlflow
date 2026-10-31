from pathlib import Path

import pytest

from mlflow.entities import (
    Assessment,
    AssessmentSource,
    AssessmentSourceType,
    Feedback,
    TraceInfo,
    TraceLocation,
    TraceState,
)
from mlflow.entities.trace_status import TraceStatus
from mlflow.environment_variables import MLFLOW_ALLOW_FILE_STORE
from mlflow.store.tracking.file_store import FileStore
from mlflow.utils.time import get_current_time_millis


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FileStore:
    monkeypatch.setenv(MLFLOW_ALLOW_FILE_STORE.name, "true")
    return FileStore(str(tmp_path / "mlruns"))


def _trace_info(
    experiment_id: str, trace_id: str, assessments: list[Assessment] | None = None
) -> TraceInfo:
    return TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation.from_experiment_id(experiment_id),
        request_time=get_current_time_millis(),
        state=TraceState.OK,
        assessments=assessments or [],
    )


def _feedback(trace_id: str | None) -> Feedback:
    return Feedback(
        trace_id=trace_id,
        name="correctness",
        value=True,
        source=AssessmentSource(source_type=AssessmentSourceType.CODE),
    )


@pytest.mark.parametrize("carried_trace_id", [None, "tr-missing", "tr-other"])
def test_start_trace_assigns_nested_assessments_to_the_created_trace(
    store: FileStore, carried_trace_id: str | None
):
    exp_id = store.create_experiment("exp")
    store.start_trace(_trace_info(exp_id, "tr-other"))

    store.start_trace(_trace_info(exp_id, "tr-new", [_feedback(carried_trace_id)]))

    stored = store.get_trace_info("tr-new").assessments
    assert [(a.name, a.trace_id) for a in stored] == [("correctness", "tr-new")]
    assert store.get_trace_info("tr-other").assessments == []


def test_deprecated_end_trace_v2_keeps_existing_assessments(store: FileStore):
    exp_id = store.create_experiment("exp")
    trace_info = store.deprecated_start_trace_v2(
        experiment_id=exp_id, timestamp_ms=0, request_metadata={}, tags={}
    )
    created = store.create_assessment(_feedback(trace_info.request_id))

    store.deprecated_end_trace_v2(
        request_id=trace_info.request_id,
        timestamp_ms=1,
        status=TraceStatus.OK,
        request_metadata={},
        tags={},
    )

    assert store.get_trace_info(trace_info.request_id).assessments == [created]
