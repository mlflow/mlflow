import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import pytest
from google.protobuf.json_format import ParseDict

import mlflow


def join_thread_by_name_prefix(prefix: str, timeout: float = 5.0):
    """Join thread by name prefix to avoid time.sleep in tests."""
    for thread in threading.enumerate():
        if thread != threading.main_thread() and thread.name.startswith(prefix):
            thread.join(timeout=timeout)


from mlflow.entities.model_registry import PromptVersion
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.protos import service_pb2 as pb
from mlflow.tracing.constant import TraceMetadataKey, TraceSizeStatsKey
from mlflow.tracing.destination import Databricks
from mlflow.tracing.export.mlflow_v3 import MlflowV3SpanExporter
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


@pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
def test_export(is_async, monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "dummy-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dummy-token")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", str(is_async))

    mlflow.tracing.set_destination(Databricks(experiment_id=_EXPERIMENT_ID))

    trace_info = None

    def mock_response(credentials, path, method, trace_json, *args, **kwargs):
        nonlocal trace_info
        trace_dict = json.loads(trace_json)
        trace_proto = ParseDict(trace_dict["trace"], pb.Trace())
        trace_info_proto = ParseDict(trace_dict["trace"]["trace_info"], pb.TraceInfoV3())
        trace_info = TraceInfo.from_proto(trace_info_proto)
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

    # Access the trace that was passed to _start_trace
    endpoint = mock_call_endpoint.call_args.args[1]
    assert endpoint == "/api/3.0/mlflow/traces"
    trace_data = mock_upload_trace_data.call_args.args[1]

    # Basic validation of the trace object
    assert trace_info.trace_id is not None

    # Validate the size stats metadata
    # Using pop() to exclude the size of these fields when computing the expected size
    size_stats = json.loads(trace_info.trace_metadata.pop(TraceMetadataKey.SIZE_STATS))
    size_bytes = int(trace_info.trace_metadata.pop(TraceMetadataKey.SIZE_BYTES))

    # The total size of the trace should much with the size of the trace object
    expected_size_bytes = len(Trace(info=trace_info, data=trace_data).to_json().encode("utf-8"))

    assert size_bytes == expected_size_bytes
    assert size_stats[TraceSizeStatsKey.TOTAL_SIZE_BYTES] == expected_size_bytes
    assert size_stats[TraceSizeStatsKey.NUM_SPANS] == 2
    assert size_stats[TraceSizeStatsKey.MAX_SPAN_SIZE_BYTES] > 0

    # Verify percentile stats are included
    assert TraceSizeStatsKey.P25_SPAN_SIZE_BYTES in size_stats
    assert TraceSizeStatsKey.P50_SPAN_SIZE_BYTES in size_stats
    assert TraceSizeStatsKey.P75_SPAN_SIZE_BYTES in size_stats

    # Verify percentiles are valid integers
    assert isinstance(size_stats[TraceSizeStatsKey.P25_SPAN_SIZE_BYTES], int)
    assert isinstance(size_stats[TraceSizeStatsKey.P50_SPAN_SIZE_BYTES], int)
    assert isinstance(size_stats[TraceSizeStatsKey.P75_SPAN_SIZE_BYTES], int)

    # Verify percentile ordering: P25 <= P50 <= P75 <= max
    assert (
        size_stats[TraceSizeStatsKey.P25_SPAN_SIZE_BYTES]
        <= size_stats[TraceSizeStatsKey.P50_SPAN_SIZE_BYTES]
    )
    assert (
        size_stats[TraceSizeStatsKey.P50_SPAN_SIZE_BYTES]
        <= size_stats[TraceSizeStatsKey.P75_SPAN_SIZE_BYTES]
    )
    assert (
        size_stats[TraceSizeStatsKey.P75_SPAN_SIZE_BYTES]
        <= size_stats[TraceSizeStatsKey.MAX_SPAN_SIZE_BYTES]
    )

    # Validate the data was passed to upload_trace_data
    call_args = mock_upload_trace_data.call_args
    assert isinstance(call_args.args[0], TraceInfo)
    assert call_args.args[0].trace_id == trace_info.trace_id

    # We don't need to validate the exact JSON structure anymore since
    # we're testing the client methods directly, not the HTTP request

    # Last active trace ID should be set
    assert mlflow.get_last_active_trace_id() is not None


@mock.patch("mlflow.tracing.export.mlflow_v3.is_in_databricks_notebook", return_value=True)
def test_async_logging_disabled_in_databricks_notebook(mock_is_in_db_notebook, monkeypatch):
    exporter = MlflowV3SpanExporter()
    assert not exporter._is_async_enabled

    # If the env var is set explicitly, we should respect that
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "True")
    exporter = MlflowV3SpanExporter()
    assert exporter._is_async_enabled


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
            "mlflow.tracing.client.TracingClient.start_trace",
            side_effect=Exception("Failed to start trace"),
        ),
        mock.patch("mlflow.tracing.export.mlflow_v3._logger") as mock_logger,
    ):
        _predict("hello")

        if is_async:
            _flush_async_logging()

    mock_logger.warning.assert_called()
    warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
    assert any("Failed to start trace" in msg for msg in warning_calls)


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
            "mlflow.tracing.client.TracingClient.start_trace", side_effect=_mock_client_method
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


@pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
def test_prompt_linking_in_mlflow_v3_exporter(is_async, monkeypatch):
    """Test that prompts are correctly linked when using MLflow v3 exporter."""
    monkeypatch.setenv("DATABRICKS_HOST", "dummy-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dummy-token")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", str(is_async))

    mlflow.tracing.set_destination(Databricks(experiment_id=_EXPERIMENT_ID))

    # Capture prompt linking calls
    captured_prompts = None
    captured_trace_id = None

    def mock_link_prompt_versions_to_trace(trace_id, prompts):
        nonlocal captured_prompts, captured_trace_id
        captured_prompts = prompts
        captured_trace_id = trace_id

    # Mock the prompt linking method and other client methods
    with (
        mock.patch(
            "mlflow.tracing.client.TracingClient.start_trace",
        ) as mock_start_trace,
        mock.patch(
            "mlflow.tracing.client.TracingClient._upload_trace_data", return_value=None
        ) as mock_upload_trace_data,
        mock.patch(
            "mlflow.tracing.client.TracingClient.link_prompt_versions_to_trace",
            side_effect=mock_link_prompt_versions_to_trace,
        ) as mock_link_prompts,
    ):
        # Create test prompt versions
        prompt1 = PromptVersion(
            name="test_prompt_1",
            version=1,
            template="Hello, {{name}}!",
            commit_message="Test prompt 1",
            creation_timestamp=123456789,
        )
        prompt2 = PromptVersion(
            name="test_prompt_2",
            version=2,
            template="Goodbye, {{name}}!",
            commit_message="Test prompt 2",
            creation_timestamp=123456790,
        )

        # Use the MLflow v3 exporter directly to test prompt linking
        from mlflow.entities import LiveSpan
        from mlflow.tracing.trace_manager import InMemoryTraceManager
        from mlflow.tracing.utils import generate_trace_id_v3

        from tests.tracing.helper import create_mock_otel_span, create_test_trace_info

        # Create a mock OTEL span and trace
        otel_span = create_mock_otel_span(
            name="root",
            trace_id=12345,
            span_id=1,
            parent_id=None,
        )
        trace_id = generate_trace_id_v3(otel_span)
        span = LiveSpan(otel_span, trace_id)

        # Register the trace and spans
        trace_manager = InMemoryTraceManager.get_instance()
        trace_info = create_test_trace_info(trace_id, _EXPERIMENT_ID)
        trace_manager.register_trace(otel_span.context.trace_id, trace_info)
        trace_manager.register_span(span)

        # Register prompts to the trace
        trace_manager.register_prompt(trace_id, prompt1)
        trace_manager.register_prompt(trace_id, prompt2)

        # Create and use the exporter
        exporter = MlflowV3SpanExporter()
        exporter.export([otel_span])

        if is_async:
            # For async tests, we need to flush the specific exporter's queue
            exporter._async_queue.flush(terminate=True)

        # Wait for any prompt linking threads to complete
        join_thread_by_name_prefix("link_prompts_from_exporter")

    # Verify that prompt linking was called
    mock_link_prompts.assert_called_once()
    assert captured_prompts is not None, "Prompts were not passed to link method"
    assert len(captured_prompts) == 2, f"Expected 2 prompts, got {len(captured_prompts)}"

    # Verify prompt details
    prompt_names = {p.name for p in captured_prompts}
    assert prompt_names == {"test_prompt_1", "test_prompt_2"}

    # Verify the trace ID matches
    assert captured_trace_id == trace_id

    # Verify other client methods were also called
    mock_start_trace.assert_called_once()
    mock_upload_trace_data.assert_called_once()


@pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
def test_prompt_linking_with_empty_prompts_mlflow_v3(is_async, monkeypatch):
    """Test that empty prompts list doesn't cause issues with MLflow v3 exporter."""
    monkeypatch.setenv("DATABRICKS_HOST", "dummy-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dummy-token")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", str(is_async))

    mlflow.tracing.set_destination(Databricks(experiment_id=_EXPERIMENT_ID))

    # Capture prompt linking calls
    captured_prompts = None
    captured_trace_id = None

    def mock_link_prompt_versions_to_trace(trace_id, prompts):
        nonlocal captured_prompts, captured_trace_id
        captured_prompts = prompts
        captured_trace_id = trace_id

    # Mock the client methods
    with (
        mock.patch(
            "mlflow.tracing.client.TracingClient.start_trace",
            return_value=mock.MagicMock(trace_id="test-trace-id"),
        ) as mock_start_trace,
        mock.patch(
            "mlflow.tracing.client.TracingClient._upload_trace_data", return_value=None
        ) as mock_upload_trace_data,
        mock.patch(
            "mlflow.tracing.client.TracingClient.link_prompt_versions_to_trace",
            side_effect=mock_link_prompt_versions_to_trace,
        ) as mock_link_prompts,
    ):
        # Use the MLflow v3 exporter directly with no prompts
        from mlflow.entities import LiveSpan
        from mlflow.tracing.trace_manager import InMemoryTraceManager
        from mlflow.tracing.utils import generate_trace_id_v3

        from tests.tracing.helper import create_mock_otel_span, create_test_trace_info

        # Create a mock OTEL span and trace (no prompts added)
        otel_span = create_mock_otel_span(
            name="root",
            trace_id=12345,
            span_id=1,
            parent_id=None,
        )
        trace_id = generate_trace_id_v3(otel_span)
        span = LiveSpan(otel_span, trace_id)

        # Register the trace and spans (but no prompts)
        trace_manager = InMemoryTraceManager.get_instance()
        trace_info = create_test_trace_info(trace_id, _EXPERIMENT_ID)
        trace_manager.register_trace(otel_span.context.trace_id, trace_info)
        trace_manager.register_span(span)

        # Create and use the exporter
        exporter = MlflowV3SpanExporter()
        exporter.export([otel_span])

        if is_async:
            # For async tests, we need to flush the specific exporter's queue
            exporter._async_queue.flush(terminate=True)

        # Wait for any prompt linking threads to complete
        join_thread_by_name_prefix("link_prompts_from_exporter")

    # Verify that prompt linking was NOT called for empty prompts (this is correct behavior)
    mock_link_prompts.assert_not_called()
    # Since no prompts were passed, no thread was started and no call was made
    assert captured_trace_id is None  # No linking occurred, so trace_id was never captured

    # Verify other client methods were also called
    mock_start_trace.assert_called_once()
    mock_upload_trace_data.assert_called_once()


def test_prompt_linking_error_handling_mlflow_v3(monkeypatch):
    """Test that MLflow v3 exporter handles prompt linking errors gracefully."""
    monkeypatch.setenv("DATABRICKS_HOST", "dummy-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dummy-token")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "False")  # Use sync for easier testing

    mlflow.tracing.set_destination(Databricks(experiment_id=_EXPERIMENT_ID))

    # Mock the client methods with prompt linking failing
    with (
        mock.patch(
            "mlflow.tracing.client.TracingClient.start_trace",
            return_value=mock.MagicMock(trace_id="test-trace-id"),
        ) as mock_start_trace,
        mock.patch(
            "mlflow.tracing.client.TracingClient._upload_trace_data", return_value=None
        ) as mock_upload_trace_data,
        mock.patch(
            "mlflow.tracing.client.TracingClient.link_prompt_versions_to_trace",
            side_effect=Exception("Prompt linking failed"),
        ) as mock_link_prompts,
        mock.patch("mlflow.tracing.export.utils._logger") as mock_logger,
    ):
        # Use the MLflow v3 exporter directly
        from mlflow.entities import LiveSpan
        from mlflow.tracing.trace_manager import InMemoryTraceManager
        from mlflow.tracing.utils import generate_trace_id_v3

        from tests.tracing.helper import create_mock_otel_span, create_test_trace_info

        # Create a mock OTEL span and trace with a prompt
        otel_span = create_mock_otel_span(
            name="root",
            trace_id=12345,
            span_id=1,
            parent_id=None,
        )
        trace_id = generate_trace_id_v3(otel_span)
        span = LiveSpan(otel_span, trace_id)

        # Create a test prompt
        prompt = PromptVersion(
            name="test_prompt",
            version=1,
            template="Hello, {{name}}!",
            commit_message="Test prompt",
            creation_timestamp=123456789,
        )

        # Register the trace, span, and prompt
        trace_manager = InMemoryTraceManager.get_instance()
        trace_info = create_test_trace_info(trace_id, _EXPERIMENT_ID)
        trace_manager.register_trace(otel_span.context.trace_id, trace_info)
        trace_manager.register_span(span)
        trace_manager.register_prompt(trace_id, prompt)

        # Create and use the exporter
        exporter = MlflowV3SpanExporter()
        exporter.export([otel_span])

        # Wait for any prompt linking threads to complete so the error can be caught
        join_thread_by_name_prefix("link_prompts_from_exporter")

    # Verify that prompt linking was attempted but failed
    mock_link_prompts.assert_called_once()

    # Verify other client methods were still called
    # (trace export should succeed despite prompt linking failure)
    mock_start_trace.assert_called_once()
    mock_upload_trace_data.assert_called_once()

    # Verify that the error was logged but didn't crash the export
    mock_logger.warning.assert_called()
    warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
    assert any("Prompt linking failed" in msg for msg in warning_calls)


def test_export_log_spans():
    """Test that log_spans is called during trace export."""

    from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
    from opentelemetry.trace import SpanContext, TraceFlags

    from mlflow.entities.span import SpanStatus, SpanStatusCode, create_mlflow_span
    from mlflow.entities.trace import Trace
    from mlflow.entities.trace_data import TraceData
    from mlflow.tracing.export.mlflow_v3 import MlflowV3SpanExporter

    # Create test span using the same pattern as other tests
    trace_id = "tr-123"
    trace_id_int = int(trace_id.replace("tr-", ""), 16)
    span_id_int = int("456", 16)

    readable_span = OTelReadableSpan(
        name="test_span",
        context=SpanContext(
            trace_id=trace_id_int,
            span_id=span_id_int,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        ),
        parent=None,
        start_time=1000000000,
        end_time=2000000000,
        attributes={
            "mlflow.traceRequestId": '{"tr-123"}',
            "mlflow.spanInputs": '{"input": "test"}',
            "mlflow.spanOutputs": '{"output": "result"}',
            "mlflow.spanType": '"UNKNOWN"',
        },
        status=SpanStatus(SpanStatusCode.OK).to_otel_status(),
        resource=None,
        events=[],
    )

    span = create_mlflow_span(readable_span, trace_id)

    # Create trace with spans
    from tests.tracing.helper import create_test_trace_info

    trace_info = create_test_trace_info(trace_id, "0")
    trace_data = TraceData(spans=[span])
    trace = Trace(trace_info, trace_data)

    # Create exporter
    exporter = MlflowV3SpanExporter()

    with (
        mock.patch.object(
            exporter._client, "start_trace", return_value=trace_info
        ) as mock_start_trace,
        mock.patch.object(exporter._client, "log_spans", return_value=[span]) as mock_log_spans,
        mock.patch.object(exporter._client, "_upload_trace_data", return_value=None) as mock_upload,
    ):
        # Call _log_trace directly
        exporter._log_trace(trace, prompts=[])

        # Verify all methods were called
        mock_start_trace.assert_called_once_with(trace_info)
        mock_log_spans.assert_called_once_with([span])
        mock_upload.assert_called_once_with(trace_info, trace_data)


def test_export_log_spans_integration():
    """
    Integration test that verifies log_spans is called with proper formatting.
    This would have caught both the 422 error and spanType warning.
    """
    from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
    from opentelemetry.trace import SpanContext, TraceFlags

    from mlflow.entities.span import SpanStatus, SpanStatusCode, create_mlflow_span
    from mlflow.entities.trace import Trace
    from mlflow.entities.trace_data import TraceData
    from mlflow.tracing.export.mlflow_v3 import MlflowV3SpanExporter

    from tests.tracing.helper import create_test_trace_info

    # Create test span with attributes that would trigger the issues
    trace_id = "tr-1234567890abcdef1234567890abcdef"
    trace_id_int = int("1234567890abcdef1234567890abcdef", 16)
    span_id_int = int("1234567890abcdef", 16)

    readable_span = OTelReadableSpan(
        name="test_span",
        context=SpanContext(
            trace_id=trace_id_int,
            span_id=span_id_int,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        ),
        parent=None,
        start_time=1000000000,
        end_time=2000000000,
        attributes={
            "mlflow.traceRequestId": json.dumps(trace_id),
            "mlflow.spanType": json.dumps("LLM"),  # This would trigger the warning
            "mlflow.spanInputs": json.dumps({"prompt": "test"}),
            "mlflow.spanOutputs": json.dumps({"response": "result"}),
        },
        status=SpanStatus(SpanStatusCode.OK).to_otel_status(),
    )

    span = create_mlflow_span(readable_span, trace_id)
    trace_info = create_test_trace_info(trace_id, "0")
    trace_data = TraceData(spans=[span])
    trace = Trace(trace_info, trace_data)

    # Create exporter
    exporter = MlflowV3SpanExporter()

    # Capture what's sent to log_spans
    captured_spans = None

    async def mock_log_spans(spans):
        nonlocal captured_spans
        captured_spans = spans
        # Just verify we can access span attributes correctly
        for span in spans:
            # This would have failed with the spanType warning
            span_type = span.span_type
            assert span_type == "LLM"
        return spans

    with (
        mock.patch.object(
            exporter._client, "start_trace", return_value=trace_info
        ) as mock_start_trace,
        mock.patch.object(
            exporter._client, "log_spans", side_effect=mock_log_spans
        ) as mock_log_spans_method,
        mock.patch.object(exporter._client, "_upload_trace_data", return_value=None) as mock_upload,
    ):
        # Call _log_trace directly
        exporter._log_trace(trace, prompts=[])

        # Verify all methods were called
        mock_start_trace.assert_called_once_with(trace_info)
        mock_log_spans_method.assert_called_once()
        mock_upload.assert_called_once()

        # Verify spans were passed correctly
        assert captured_spans is not None
        assert len(captured_spans) == 1
        assert captured_spans[0].name == "test_span"
        assert captured_spans[0].span_type == "LLM"
