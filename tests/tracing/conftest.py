from typing import Dict
from unittest import mock

import pytest

import mlflow
from mlflow.entities import TraceInfo
from mlflow.entities.trace_status import TraceStatus
from mlflow.tracing.display import IPythonTraceDisplayHandler
from mlflow.tracing.export.inference_table import _TRACE_BUFFER
from mlflow.tracing.fluent import TRACE_BUFFER
from mlflow.tracing.provider import _TRACER_PROVIDER_INITIALIZED
from mlflow.tracing.trace_manager import InMemoryTraceManager

from tests.tracing.helper import create_test_trace_info


@pytest.fixture(autouse=True)
def clear_singleton():
    """
    Clear the singleton instances after each tests to avoid side effects.
    """
    InMemoryTraceManager._instance = None
    IPythonTraceDisplayHandler._instance = None
    TRACE_BUFFER.clear()
    _TRACE_BUFFER.clear()

    # Tracer provider also needs to be reset as it may hold reference to the singleton
    with _TRACER_PROVIDER_INITIALIZED._lock:
        _TRACER_PROVIDER_INITIALIZED._done = False


@pytest.fixture(autouse=True)
def reset_active_experiment():
    yield
    mlflow.tracking.fluent._active_experiment_id = None


@pytest.fixture
def mock_upload_trace_data():
    with mock.patch(
        "mlflow.tracking._tracking_service.client.TrackingServiceClient.end_trace",
        return_value=TraceInfo(
            request_id="tr-1234",
            experiment_id="0",
            timestamp_ms=0,
            execution_time_ms=0,
            status=TraceStatus.OK,
            request_metadata={},
            tags={"mlflow.artifactLocation": "test"},
        ),
    ), mock.patch(
        "mlflow.tracking._tracking_service.client.TrackingServiceClient._upload_trace_data",
        return_value=None,
    ) as mock_upload_trace_data:
        yield mock_upload_trace_data


@pytest.fixture
def mock_store(monkeypatch):
    """
    Mocking the StartTrace and EndTrace API call to the tracking backend. We only mock those two
    API calls, so the rest of the tracking API calls the actual tracking store e.g. create_run().
    """
    store = mlflow.tracking._tracking_service.utils._get_store()
    with mock.patch("mlflow.tracking._tracking_service.utils._get_store") as mock_get_store:
        mock_get_store.return_value = store

        _traces: Dict[str, TraceInfo] = {}

        def _mock_start_trace(experiment_id, timestamp_ms, request_metadata, tags):
            trace_info = create_test_trace_info(
                request_id="tr-12345",
                experiment_id=experiment_id,
                timestamp_ms=timestamp_ms,
                execution_time_ms=None,
                status=TraceStatus.IN_PROGRESS,
                request_metadata=request_metadata,
                tags={
                    "mlflow.user": "bob",
                    "mlflow.artifactLocation": "test",
                    **tags,
                },
            )
            _traces[trace_info.request_id] = trace_info
            return trace_info

        def _mock_end_trace(request_id, timestamp_ms, status, request_metadata, tags):
            trace_info = _traces[request_id]
            trace_info.execution_time_ms = timestamp_ms - trace_info.timestamp_ms
            trace_info.status = status
            trace_info.request_metadata.update(request_metadata)
            trace_info.tags.update(tags)
            return trace_info

        monkeypatch.setattr(store, "start_trace", mock.MagicMock(side_effect=_mock_start_trace))
        monkeypatch.setattr(store, "end_trace", mock.MagicMock(side_effect=_mock_end_trace))
        yield store


@pytest.fixture
def databricks_tracking_uri():
    with mock.patch(
        "mlflow.tracking._tracking_service.utils.get_tracking_uri", return_value="databricks"
    ):
        yield
