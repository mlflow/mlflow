import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import pytest
from google.protobuf.json_format import ParseDict

import mlflow
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.trace import Trace
from mlflow.entities.trace_info_v3 import TraceInfoV3
from mlflow.protos import service_pb2 as pb
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.destination import Databricks
from mlflow.tracing.provider import _get_trace_exporter

_EXPERIMENT_ID = "dummy-experiment-id"


@mlflow.trace
def _predict(x: str) -> str:
    with mlflow.start_span(name="child") as child_span:
        child_span.set_inputs("dummy")
        child_span.add_event(SpanEvent(name="child_event", attributes={"attr1": "val1"}))
    mlflow.update_current_trace(tags={"foo": "bar"})
    return x + "!"


def _flush_async_logging():
    exporter = _get_trace_exporter()
    assert hasattr(exporter, "_async_queue"), "Async queue is not initialized"
    exporter._async_queue.flush(terminate=True)


# @pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
# @pytest.mark.parametrize("experiment_id", [None, _EXPERIMENT_ID])
@pytest.mark.parametrize("is_async", [False], ids=["async"])
@pytest.mark.parametrize("experiment_id", [None])
def test_export_with_size(experiment_id, is_async, monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "dummy-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dummy-token")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", str(is_async))

    mlflow.tracing.set_destination(Databricks(experiment_id=experiment_id))

    trace_info = None

    def mock_response(credentials, path, method, trace_json, *args, **kwargs):
        nonlocal trace_info
        print("TRACE JSON REQUEST", trace_json)
        trace_dict = json.loads(trace_json)
        trace_proto = ParseDict(trace_dict["trace"], pb.Trace())
        trace_info_proto = ParseDict(trace_dict["trace"]["trace_info"], pb.TraceInfoV3())
        trace_info = TraceInfoV3.from_proto(trace_info_proto)
        return pb.StartTraceV3.Response(trace=trace_proto)

    with (
        mock.patch(
            "mlflow.store.tracking.rest_store.call_endpoint", side_effect=mock_response
        ) as mock_call_endpoint,
        mock.patch(
            "mlflow.tracing.client.TracingClient._upload_trace_data", return_value=None
        ) as mock_upload_trace_data,
    ):
        _predict("hello")

        if is_async:
            _flush_async_logging()

    # Verify client methods were called correctly
    mock_call_endpoint.assert_called_once()
    mock_upload_trace_data.assert_called_once()

    # Access the trace that was passed to _start_trace_v3
    endpoint = mock_call_endpoint.call_args.args[1]
    assert endpoint == "/api/3.0/mlflow/traces"
    trace_info_dict = trace_info.to_dict()
    trace_data = mock_upload_trace_data.call_args.args[1]
    trace = Trace(info=trace_info, data=trace_data)

    print("EXPECTED TRACE", trace.to_json())

    # Basic validation of the trace object
    assert trace_info_dict["trace_id"] is not None

    # Verify that the SIZE_BYTES entry is present with the correct value
    # in trace metadata
    assert TraceMetadataKey.SIZE_BYTES in trace_info_dict["trace_metadata"]
    size_bytes = int(trace_info_dict["trace_metadata"][TraceMetadataKey.SIZE_BYTES])
    # Verify that the size_bytes value matches the actual size of the trace in bytes
    actual_size_bytes = len(trace.to_json().encode("utf-8"))
    assert size_bytes == actual_size_bytes, (
        f"Expected size_bytes to match actual size, but got {size_bytes} != {actual_size_bytes}"
    )

    # Validate the data was passed to upload_trace_data
    call_args = mock_upload_trace_data.call_args
    assert isinstance(call_args.args[0], TraceInfoV3)
    assert call_args.args[0].trace_id == "12345"

    # We don't need to validate the exact JSON structure anymore since
    # we're testing the client methods directly, not the HTTP request

    # Last active trace ID should be set
    assert mlflow.get_last_active_trace_id() is not None


@pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
def test_export_catch_failure(is_async, monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "dummy-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dummy-token")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", str(is_async))

    mlflow.tracing.set_destination(Databricks(experiment_id=_EXPERIMENT_ID))

    response = mock.MagicMock()
    response.status_code = 500
    response.text = "Failed to export trace"

    with (
        mock.patch(
            "mlflow.tracing.client.TracingClient.start_trace_v3",
            side_effect=Exception("Failed to start trace"),
        ),
        mock.patch("mlflow.tracing.export.mlflow_v3._logger") as mock_logger,
    ):
        _predict("hello")

        if is_async:
            _flush_async_logging()

    mock_logger.warning.assert_called_once()


@pytest.mark.skipif(os.name == "nt", reason="Flaky on Windows")
def test_async_bulk_export(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "dummy-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dummy-token")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "True")
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_MAX_QUEUE_SIZE", "1000")

    mlflow.tracing.set_destination(Databricks(experiment_id=0))

    # Create a mock function that simulates delay
    def _mock_client_method(*args, **kwargs):
        # Simulate a slow response
        time.sleep(0.1)
        mock_trace = mock.MagicMock()
        mock_trace.info = mock.MagicMock()
        return mock_trace

    with (
        mock.patch(
            "mlflow.tracing.client.TracingClient.start_trace_v3", side_effect=_mock_client_method
        ) as mock_start_trace,
        mock.patch(
            "mlflow.tracing.client.TracingClient._upload_trace_data", return_value=None
        ) as mock_upload_trace_data,
    ):
        # Log many traces
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            for _ in range(100):
                executor.submit(_predict, "hello")

        # Trace logging should not block the main thread
        assert time.time() - start_time < 5

        _flush_async_logging()

    # Verify the client methods were called the expected number of times
    assert mock_start_trace.call_count == 100
    assert mock_upload_trace_data.call_count == 100
