from typing import Optional
from unittest import mock

import pytest

import mlflow
from mlflow.entities import LiveSpan, Trace
from mlflow.entities.model_registry import PromptVersion
from mlflow.entities.trace_info import TraceInfo
from mlflow.protos import service_pb2 as pb
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.export.inference_table import (
    _TRACE_BUFFER,
    InferenceTableSpanExporter,
    _initialize_trace_buffer,
    pop_trace,
)
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import generate_trace_id_v3

from tests.tracing.helper import create_mock_otel_span, create_test_trace_info

_OTEL_TRACE_ID = 12345
_DATABRICKS_REQUEST_ID_1 = "databricks-request-id-1"
_DATABRICKS_REQUEST_ID_2 = "databricks-request-id-2"


@pytest.mark.parametrize("dual_write_enabled", [True, False])
def test_export(dual_write_enabled, monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_TRACE_DUAL_WRITE_IN_MODEL_SERVING", str(dual_write_enabled))

    otel_span = create_mock_otel_span(
        name="root",
        trace_id=_OTEL_TRACE_ID,
        span_id=1,
        parent_id=None,
        start_time=0,
        end_time=1_000_000,  # 1 millisecond
    )
    trace_id = generate_trace_id_v3(otel_span)
    span = LiveSpan(otel_span, trace_id)
    span.set_inputs({"input1": "very long input" * 100})
    span.set_outputs("very long output" * 100)
    _register_span_and_trace(span, client_request_id=_DATABRICKS_REQUEST_ID_1)

    child_otel_span = create_mock_otel_span(
        name="child", trace_id=_OTEL_TRACE_ID, span_id=2, parent_id=1
    )
    child_span = LiveSpan(child_otel_span, trace_id)
    _register_span_and_trace(child_span, client_request_id=_DATABRICKS_REQUEST_ID_1)

    # Invalid span should be also ignored
    invalid_otel_span = create_mock_otel_span(trace_id=23456, span_id=1)

    mock_tracing_client = mock.MagicMock()
    with mock.patch(
        "mlflow.tracing.export.inference_table.TracingClient", return_value=mock_tracing_client
    ):
        exporter = InferenceTableSpanExporter()

    exporter.export([otel_span, invalid_otel_span])

    # Spans should be cleared from the trace manager
    assert len(exporter._trace_manager._traces) == 0

    # Trace should be added to the in-memory buffer and can be extracted
    assert len(_TRACE_BUFFER) == 1
    trace_dict = pop_trace(_DATABRICKS_REQUEST_ID_1)
    trace_info = trace_dict["info"]
    assert trace_info["trace_id"] == trace_id
    assert trace_info["client_request_id"] == _DATABRICKS_REQUEST_ID_1
    assert trace_info["request_time"] == "1970-01-01T00:00:00Z"
    assert trace_info["execution_duration_ms"] == 1

    spans = trace_dict["data"]["spans"]
    assert len(spans) == 2
    assert spans[0]["name"] == "root"
    assert isinstance(spans[0]["attributes"], dict)

    # Last active trace ID should be set
    assert mlflow.get_last_active_trace_id() == trace_id

    if dual_write_enabled:
        exporter._async_queue.flush(terminate=True)

        assert mock_tracing_client.start_trace_v3.call_count == 1
        trace = mock_tracing_client.start_trace_v3.call_args[0][0]
        assert isinstance(trace.info, TraceInfo)
        # The trace ID should be updated to the format that MLflow backend accept
        assert trace.info.trace_id == trace_id
        # The databricks request ID should be set to the client request ID
        assert trace.info.client_request_id == _DATABRICKS_REQUEST_ID_1
    else:
        assert mock_tracing_client.log_trace.call_count == 0


def test_export_warn_invalid_attributes():
    otel_span = create_mock_otel_span(trace_id=_OTEL_TRACE_ID, span_id=1)
    trace_id = generate_trace_id_v3(otel_span)
    span = LiveSpan(otel_span, trace_id)
    span.set_attribute("valid", "value")
    # # Users may set attribute directly to the OpenTelemetry span
    # otel_span.set_attribute("int", 1)
    span.set_attribute("str", "a")
    _register_span_and_trace(span, client_request_id=_DATABRICKS_REQUEST_ID_1)

    exporter = InferenceTableSpanExporter()
    exporter.export([otel_span])

    trace_dict = pop_trace(_DATABRICKS_REQUEST_ID_1)
    trace = Trace.from_dict(trace_dict)
    stored_span = trace.data.spans[0]
    assert stored_span.attributes == {
        "mlflow.traceRequestId": trace_id,
        "mlflow.spanType": "UNKNOWN",
        "valid": "value",
        "str": "a",
    }

    # Users shouldn't set attribute directly to the OTel span
    otel_span.set_attribute("int", 1)
    exporter.export([otel_span])
    with mock.patch("mlflow.entities.span._logger.warning") as mock_warning:
        span.attributes
        mock_warning.assert_called_once()
        msg = mock_warning.call_args[0][0]
        assert msg.startswith("Failed to get value for key int")


def test_export_trace_buffer_not_exceeds_max_size(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACE_BUFFER_MAX_SIZE", "1")
    monkeypatch.setattr(
        mlflow.tracing.export.inference_table, "_TRACE_BUFFER", _initialize_trace_buffer()
    )

    exporter = InferenceTableSpanExporter()

    otel_span_1 = create_mock_otel_span(name="1", trace_id=_OTEL_TRACE_ID, span_id=1)
    trace_id = generate_trace_id_v3(otel_span_1)
    _register_span_and_trace(
        LiveSpan(otel_span_1, trace_id), client_request_id=_DATABRICKS_REQUEST_ID_1
    )

    exporter.export([otel_span_1])

    assert pop_trace(_DATABRICKS_REQUEST_ID_1) is not None

    otel_span_2 = create_mock_otel_span(name="2", trace_id=_OTEL_TRACE_ID + 1, span_id=1)
    _register_span_and_trace(
        LiveSpan(otel_span_2, trace_id), client_request_id=_DATABRICKS_REQUEST_ID_2
    )

    exporter.export([otel_span_2])

    assert pop_trace(_DATABRICKS_REQUEST_ID_1) is None
    assert pop_trace(_DATABRICKS_REQUEST_ID_2) is not None


def test_size_bytes_in_trace_sent_to_mlflow_backend(monkeypatch):
    """Test that SIZE_BYTES is correctly set in the trace sent to MLflow backend via dual write."""
    # Enable dual write
    monkeypatch.setenv("MLFLOW_ENABLE_TRACE_DUAL_WRITE_IN_MODEL_SERVING", "True")

    # Create spans and trace
    otel_span = create_mock_otel_span(
        name="root",
        trace_id=_OTEL_TRACE_ID,
        span_id=1,
        parent_id=None,
        start_time=0,
        end_time=1_000_000,  # 1 millisecond
    )
    trace_id = generate_trace_id_v3(otel_span)
    span = LiveSpan(otel_span, trace_id)
    span.set_inputs({"input1": "very long input" * 100})
    span.set_outputs("very long output" * 100)

    # Register with experiment_id to ensure dual write happens (the code checks for this)
    _register_span_and_trace(span, client_request_id=_DATABRICKS_REQUEST_ID_1, experiment_id="123")

    # Set up mocks to capture the trace sent to MLflow backend
    captured_trace = None

    def mock_log_trace_to_mlflow_backend(trace):
        nonlocal captured_trace
        # Store a copy of the trace before it's modified by the log_trace method
        captured_trace = Trace(
            info=TraceInfo.from_proto(TraceInfo.to_proto(trace.info)), data=trace.data
        )
        # Mock implementation continues
        return pb.StartTraceV3.Response()

    # Create mock client that captures the trace
    mock_tracing_client = mock.MagicMock()
    mock_tracing_client.start_trace_v3.side_effect = mock_log_trace_to_mlflow_backend

    with mock.patch(
        "mlflow.tracing.export.inference_table.TracingClient", return_value=mock_tracing_client
    ):
        exporter = InferenceTableSpanExporter()
        exporter.export([otel_span])
        # Ensure async queue is processed
        exporter._async_queue.flush(terminate=True)

    # Verify the trace sent to MLflow backend has SIZE_BYTES
    assert captured_trace is not None, "Trace was not sent to MLflow backend"
    assert TraceMetadataKey.SIZE_BYTES in captured_trace.info.trace_metadata, (
        "SIZE_BYTES missing in trace metadata"
    )

    # Get the size bytes that were set
    size_bytes = int(captured_trace.info.trace_metadata[TraceMetadataKey.SIZE_BYTES])

    # Remove the size metadata to calculate the expected size
    if TraceMetadataKey.SIZE_BYTES in captured_trace.info.trace_metadata:
        del captured_trace.info.trace_metadata[TraceMetadataKey.SIZE_BYTES]

    # Calculate size exactly the same way the function does
    expected_size_bytes = len(captured_trace.to_json().encode("utf-8"))

    # Verify that size_bytes matches the expected calculation
    assert size_bytes == expected_size_bytes, (
        f"Size bytes mismatch: got {size_bytes}, expected {expected_size_bytes}"
    )


def test_prompt_linking_with_dual_write(monkeypatch):
    """Test that prompts are correctly linked when dual write to MLflow backend is enabled."""
    # Enable dual write
    monkeypatch.setenv("MLFLOW_ENABLE_TRACE_DUAL_WRITE_IN_MODEL_SERVING", "True")

    # Create span and trace
    otel_span = create_mock_otel_span(
        name="root",
        trace_id=_OTEL_TRACE_ID,
        span_id=1,
        parent_id=None,
    )
    trace_id = generate_trace_id_v3(otel_span)
    span = LiveSpan(otel_span, trace_id)

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

    # Register span and trace with experiment_id to enable dual write
    trace_manager = InMemoryTraceManager.get_instance()
    trace_info = create_test_trace_info(trace_id, "123")
    trace_info.client_request_id = _DATABRICKS_REQUEST_ID_1
    trace_manager.register_trace(otel_span.context.trace_id, trace_info)
    trace_manager.register_span(span)

    # Register prompts to the trace
    trace_manager.register_prompt(trace_id, prompt1)
    trace_manager.register_prompt(trace_id, prompt2)

    # Mock the tracing client
    mock_tracing_client = mock.MagicMock()
    captured_prompts = None
    captured_trace_id = None

    def mock_link_prompt_versions_to_trace(trace_id, prompts):
        nonlocal captured_prompts, captured_trace_id
        captured_prompts = prompts
        captured_trace_id = trace_id

    mock_tracing_client.link_prompt_versions_to_trace.side_effect = (
        mock_link_prompt_versions_to_trace
    )

    # Mock start_trace_v3 to return a mock trace info with the correct trace_id
    mock_trace_info = mock.MagicMock()
    mock_trace_info.trace_id = trace_id
    mock_tracing_client.start_trace_v3.return_value = mock_trace_info

    with mock.patch(
        "mlflow.tracing.export.inference_table.TracingClient", return_value=mock_tracing_client
    ):
        exporter = InferenceTableSpanExporter()
        exporter.export([otel_span])
        # Ensure async queue is processed
        exporter._async_queue.flush(terminate=True)

    # Verify that prompts were passed to the linking method
    assert captured_prompts is not None, "Prompts were not passed to link method"
    assert len(captured_prompts) == 2, f"Expected 2 prompts, got {len(captured_prompts)}"

    # Verify prompt details
    prompt_names = {p.name for p in captured_prompts}
    assert prompt_names == {"test_prompt_1", "test_prompt_2"}

    # Verify the link method was called with correct trace ID and prompts
    mock_tracing_client.link_prompt_versions_to_trace.assert_called_once_with(
        trace_id=trace_id, prompts=captured_prompts
    )
    assert captured_trace_id == trace_id


def test_prompt_linking_disabled_without_dual_write(monkeypatch):
    """Test that prompt linking is not attempted when dual write is disabled."""
    # Disable dual write
    monkeypatch.setenv("MLFLOW_ENABLE_TRACE_DUAL_WRITE_IN_MODEL_SERVING", "False")

    # Create span and trace
    otel_span = create_mock_otel_span(
        name="root",
        trace_id=_OTEL_TRACE_ID,
        span_id=1,
        parent_id=None,
    )
    trace_id = generate_trace_id_v3(otel_span)
    span = LiveSpan(otel_span, trace_id)

    # Create test prompt version
    prompt = PromptVersion(
        name="test_prompt",
        version=1,
        template="Hello, {{name}}!",
        commit_message="Test prompt",
        creation_timestamp=123456789,
    )

    # Register span and trace
    trace_manager = InMemoryTraceManager.get_instance()
    trace_info = create_test_trace_info(trace_id, "0")
    trace_info.client_request_id = _DATABRICKS_REQUEST_ID_1
    trace_manager.register_trace(otel_span.context.trace_id, trace_info)
    trace_manager.register_span(span)

    # Register prompt to the trace
    trace_manager.register_prompt(trace_id, prompt)

    # Mock the tracing client (shouldn't be called when dual write is disabled)
    mock_tracing_client = mock.MagicMock()

    with mock.patch(
        "mlflow.tracing.export.inference_table.TracingClient", return_value=mock_tracing_client
    ):
        exporter = InferenceTableSpanExporter()
        exporter.export([otel_span])

    # Verify that no dual write methods were called
    mock_tracing_client.start_trace_v3.assert_not_called()
    mock_tracing_client.link_prompt_versions_to_trace.assert_not_called()

    # But the trace should still be in the inference table buffer
    assert len(_TRACE_BUFFER) == 1
    trace_dict = pop_trace(_DATABRICKS_REQUEST_ID_1)
    assert trace_dict is not None


def test_prompt_linking_with_empty_prompts(monkeypatch):
    """Test that empty prompts list doesn't cause issues with dual write."""
    # Enable dual write
    monkeypatch.setenv("MLFLOW_ENABLE_TRACE_DUAL_WRITE_IN_MODEL_SERVING", "True")

    # Create span and trace
    otel_span = create_mock_otel_span(
        name="root",
        trace_id=_OTEL_TRACE_ID,
        span_id=1,
        parent_id=None,
    )
    trace_id = generate_trace_id_v3(otel_span)
    span = LiveSpan(otel_span, trace_id)

    # Register span and trace with experiment_id to enable dual write (no prompts added)
    trace_manager = InMemoryTraceManager.get_instance()
    trace_info = create_test_trace_info(trace_id, "123")
    trace_info.client_request_id = _DATABRICKS_REQUEST_ID_1
    trace_manager.register_trace(otel_span.context.trace_id, trace_info)
    trace_manager.register_span(span)

    # Mock the tracing client
    mock_tracing_client = mock.MagicMock()
    captured_prompts = None
    captured_trace_id = None

    def mock_link_prompt_versions_to_trace(trace_id, prompts):
        nonlocal captured_prompts, captured_trace_id
        captured_prompts = prompts
        captured_trace_id = trace_id

    mock_tracing_client.link_prompt_versions_to_trace.side_effect = (
        mock_link_prompt_versions_to_trace
    )

    # Mock start_trace_v3 to return a mock trace info with the correct trace_id
    mock_trace_info = mock.MagicMock()
    mock_trace_info.trace_id = trace_id
    mock_tracing_client.start_trace_v3.return_value = mock_trace_info

    with mock.patch(
        "mlflow.tracing.export.inference_table.TracingClient", return_value=mock_tracing_client
    ):
        exporter = InferenceTableSpanExporter()
        exporter.export([otel_span])
        # Ensure async queue is processed
        exporter._async_queue.flush(terminate=True)

    # Verify that prompt linking was NOT called for empty prompts (this is optimized behavior)
    mock_tracing_client.link_prompt_versions_to_trace.assert_not_called()
    # Since no prompts were passed, no linking occurred, so trace_id was never captured
    assert captured_trace_id is None  # No linking occurred, so trace_id was never captured


def test_prompt_linking_error_handling_with_dual_write(monkeypatch):
    """Test that prompt linking errors are handled gracefully with dual write enabled."""
    # Enable dual write
    monkeypatch.setenv("MLFLOW_ENABLE_TRACE_DUAL_WRITE_IN_MODEL_SERVING", "True")

    # Create span and trace
    otel_span = create_mock_otel_span(
        name="root",
        trace_id=_OTEL_TRACE_ID,
        span_id=1,
        parent_id=None,
    )
    trace_id = generate_trace_id_v3(otel_span)
    span = LiveSpan(otel_span, trace_id)

    # Create test prompt version
    prompt = PromptVersion(
        name="test_prompt",
        version=1,
        template="Hello, {{name}}!",
        commit_message="Test prompt",
        creation_timestamp=123456789,
    )

    # Register span and trace with experiment_id to enable dual write
    trace_manager = InMemoryTraceManager.get_instance()
    trace_info = create_test_trace_info(trace_id, "123")
    trace_info.client_request_id = _DATABRICKS_REQUEST_ID_1
    trace_manager.register_trace(otel_span.context.trace_id, trace_info)
    trace_manager.register_span(span)

    # Register prompt to the trace
    trace_manager.register_prompt(trace_id, prompt)

    # Mock the tracing client with prompt linking failing
    mock_tracing_client = mock.MagicMock()
    mock_tracing_client.link_prompt_versions_to_trace.side_effect = Exception(
        "Prompt linking failed"
    )

    # Mock start_trace_v3 to return a mock trace info with the correct trace_id
    mock_trace_info = mock.MagicMock()
    mock_trace_info.trace_id = trace_id
    mock_tracing_client.start_trace_v3.return_value = mock_trace_info

    with (
        mock.patch(
            "mlflow.tracing.export.inference_table.TracingClient",
            return_value=mock_tracing_client,
        ),
        mock.patch("mlflow.tracing.export.utils._logger") as mock_logger,
    ):
        exporter = InferenceTableSpanExporter()
        exporter.export([otel_span])
        # Ensure async queue is processed
        exporter._async_queue.flush(terminate=True)

    # Verify that the prompt linking method was called with expected arguments but failed
    expected_prompts = [prompt]  # Should have the one prompt we registered
    mock_tracing_client.link_prompt_versions_to_trace.assert_called_once_with(
        trace_id=trace_id, prompts=expected_prompts
    )

    # Verify other client methods were still called (trace export should succeed)
    mock_tracing_client.start_trace_v3.assert_called_once()
    mock_tracing_client._upload_trace_data.assert_called_once()

    # Verify that the error was logged but didn't crash the export
    mock_logger.warning.assert_called()
    warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
    assert any("Prompt linking failed" in msg for msg in warning_calls)

    # Verify the trace is still in the inference table buffer
    assert len(_TRACE_BUFFER) == 1
    trace_dict = pop_trace(_DATABRICKS_REQUEST_ID_1)
    assert trace_dict is not None


def _register_span_and_trace(
    span: LiveSpan, client_request_id: str, experiment_id: Optional[str] = None
):
    trace_manager = InMemoryTraceManager.get_instance()
    if span.parent_id is None:
        trace_info = create_test_trace_info(span.trace_id, experiment_id or "0")
        trace_info.client_request_id = client_request_id
        trace_manager.register_trace(span._span.context.trace_id, trace_info)
    trace_manager.register_span(span)
