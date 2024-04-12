from unittest import mock

import pytest

from mlflow.entities import SpanStatus, Trace, TraceData, TraceInfo, TraceStatus
from mlflow.tracing.clients import InMemoryTraceClientWithTracking


@pytest.fixture
def mock_tracking_serving_client():
    with mock.patch(
        "mlflow.tracking._tracking_service.client.TrackingServiceClient.create_trace_info",
        return_value=TraceInfo(
            request_id="tr-1234",
            experiment_id="0",
            timestamp_ms=0,
            execution_time_ms=0,
            status=SpanStatus(TraceStatus.OK),
            request_metadata={},
            tags={"mlflow.artifactLocation": "test"},
        ),
    ) as mock_create_trace_info, mock.patch(
        "mlflow.tracking._tracking_service.client.TrackingServiceClient._upload_trace_data",
        return_value=None,
    ) as mock_upload_trace_data:
        yield

        mock_create_trace_info.assert_called()
        mock_upload_trace_data.assert_called()


def test_log_and_get_trace(monkeypatch, create_trace, mock_tracking_serving_client):
    monkeypatch.setenv("MLFLOW_TRACING_CLIENT_BUFFER_SIZE", "3")

    def _create_trace(request_id: str):
        return Trace(
            info=TraceInfo(
                request_id=request_id,
                experiment_id="test",
                timestamp_ms=0,
                execution_time_ms=1,
                status=SpanStatus(TraceStatus.OK),
                request_metadata={},
                tags={},
            ),
            data=TraceData(),
        )

    client = InMemoryTraceClientWithTracking.get_instance()
    traces = client.get_traces()
    assert len(traces) == 0

    client.log_trace(create_trace("a"))
    client.log_trace(create_trace("b"))
    client.log_trace(create_trace("c"))

    traces = client.get_traces()
    assert len(traces) == 3
    assert traces[0].info.request_id == "a"

    traces = client.get_traces(1)
    assert len(traces) == 1
    assert traces[0].info.request_id == "c"

    client.log_trace(create_trace("d"))
    traces = client.get_traces()
    assert len(traces) == 3
    assert traces[0].info.request_id == "b"
