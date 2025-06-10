"""
Enhanced tests for inference table exporter with specific argument verification.
"""

from unittest import mock

from mlflow.entities import LiveSpan
from mlflow.entities.model_registry import PromptVersion
from mlflow.tracing.export.inference_table import (
    _TRACE_BUFFER,
    InferenceTableSpanExporter,
    pop_trace,
)
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import generate_trace_id_v3

from tests.tracing.helper import create_mock_otel_span, create_test_trace_info

_OTEL_TRACE_ID = 12345
_DATABRICKS_REQUEST_ID_1 = "databricks-request-id-1"


def test_prompt_linking_with_dual_write_enhanced_assertions(monkeypatch):
    """Test that prompts are correctly linked with enhanced assertion specificity."""
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
        version_metadata={"test": "prompt1"},
        creation_timestamp=123456789,
    )
    prompt2 = PromptVersion(
        name="test_prompt_2",
        version=2,
        template="Goodbye, {{name}}!",
        commit_message="Test prompt 2",
        version_metadata={"test": "prompt2"},
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

    # Verify that the link method was called with correct trace ID and prompts
    # This uses specific argument verification instead of just assert_called_once()
    mock_tracing_client.link_prompt_versions_to_trace.assert_called_once_with(
        trace_id=trace_id, prompts=captured_prompts
    )

    # Verify prompt details
    assert captured_prompts is not None, "Prompts were not passed to link method"
    assert len(captured_prompts) == 2, f"Expected 2 prompts, got {len(captured_prompts)}"

    prompt_names = {p.name for p in captured_prompts}
    assert prompt_names == {"test_prompt_1", "test_prompt_2"}
    assert captured_trace_id == trace_id


def test_prompt_linking_error_handling_with_dual_write_enhanced(monkeypatch):
    """Test that prompt linking errors are handled gracefully with enhanced verification."""
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
        version_metadata={"test": "prompt"},
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
        mock.patch("mlflow.tracing.export.inference_table._logger") as mock_logger,
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
    mock_logger.warning.assert_called_once()
    warning_message = mock_logger.warning.call_args[0][0]
    assert "Failed to link prompts to trace" in warning_message
    assert "Prompt linking failed" in warning_message

    # Verify the trace is still in the inference table buffer
    assert len(_TRACE_BUFFER) == 1
    trace_dict = pop_trace(_DATABRICKS_REQUEST_ID_1)
    assert trace_dict is not None


def test_empty_prompts_list_linking_enhanced(monkeypatch):
    """Test that empty prompts list doesn't cause issues with enhanced verification."""
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

    # Verify that an empty prompts list was passed with specific argument verification
    mock_tracing_client.link_prompt_versions_to_trace.assert_called_once_with(
        trace_id=trace_id, prompts=[]
    )

    assert captured_prompts is not None, "Prompts parameter was not passed"
    assert len(captured_prompts) == 0, f"Expected 0 prompts, got {len(captured_prompts)}"
    assert captured_trace_id == trace_id
