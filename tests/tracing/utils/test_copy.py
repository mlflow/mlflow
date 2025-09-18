from unittest.mock import MagicMock, patch

from mlflow.tracing.utils.copy import copy_trace_to_experiment


def test_copy_trace_same_experiment_returns_early():
    """Test that copy_trace_to_experiment returns early when trace is already in target
    experiment.
    """
    # Create a trace_dict with MLFLOW_EXPERIMENT location
    trace_dict = {
        "info": {
            "trace_id": "existing-trace-123",
            "trace_location": {
                "type": "MLFLOW_EXPERIMENT",
                "mlflow_experiment": {"experiment_id": "exp-123"},
            },
        },
        "data": {
            "spans": [
                {
                    "span_id": "span-1",
                    "trace_id": "existing-trace-123",
                    "parent_id": None,
                    "name": "root_span",
                    "start_time_ns": 1000000000,
                    "end_time_ns": 2000000000,
                    "span_type": "UNKNOWN",
                    "inputs": {},
                    "outputs": {},
                    "attributes": {},
                    "events": [],
                }
            ]
        },
    }

    # Call with same experiment_id
    result = copy_trace_to_experiment(trace_dict, experiment_id="exp-123")

    # Should return the existing trace ID without creating a new trace
    assert result == "existing-trace-123"


def test_copy_trace_v2_same_experiment_returns_early():
    """Test that copy_trace_to_experiment returns early for V2 trace already in target
    experiment.
    """
    # Create a V2 trace_dict (has request_id field, direct experiment_id)
    trace_dict = {
        "info": {
            "request_id": "existing-trace-v2-123",
            "experiment_id": "exp-123",
            "timestamp_ms": 1000000000,
            "execution_time_ms": 1000,
            "status": "OK",
            "request_metadata": {},
            "tags": {},
        },
        "data": {
            "spans": [
                {
                    "span_id": "span-1",
                    "trace_id": "existing-trace-v2-123",
                    "parent_id": None,
                    "name": "root_span",
                    "start_time_ns": 1000000000,
                    "end_time_ns": 2000000000,
                    "span_type": "UNKNOWN",
                    "inputs": {},
                    "outputs": {},
                    "attributes": {},
                    "events": [],
                }
            ]
        },
    }

    # Call with same experiment_id
    result = copy_trace_to_experiment(trace_dict, experiment_id="exp-123")

    # Should return the existing request_id (trace ID for V2)
    assert result == "existing-trace-v2-123"


def test_copy_trace_different_experiment_creates_new_trace():
    """Test that copy_trace_to_experiment creates new trace when experiment IDs differ."""
    trace_dict = {
        "info": {
            "trace_id": "existing-trace-123",
            "trace_location": {
                "type": "MLFLOW_EXPERIMENT",
                "mlflow_experiment": {"experiment_id": "exp-123"},
            },
        },
        "data": {
            "spans": [
                {
                    "span_id": "span-1",
                    "trace_id": "existing-trace-123",
                    "parent_id": None,
                    "name": "root_span",
                    "start_time_ns": 1000000000,
                    "end_time_ns": 2000000000,
                    "span_type": "UNKNOWN",
                    "inputs": {},
                    "outputs": {},
                    "attributes": {},
                    "events": [],
                }
            ]
        },
    }

    # Mock the trace manager and LiveSpan creation
    with (
        patch("mlflow.tracing.utils.copy.InMemoryTraceManager") as mock_manager,
        patch("mlflow.tracing.utils.copy.LiveSpan") as mock_live_span,
        patch("mlflow.tracing.utils.copy.Span") as mock_span,
    ):
        # Setup mocks
        mock_manager.get_instance.return_value = MagicMock()

        mock_span_instance = MagicMock()
        mock_span_instance.parent_id = None
        mock_span.from_dict.return_value = mock_span_instance

        mock_new_span = MagicMock()
        mock_new_span.trace_id = "new-trace-456"
        mock_new_span.parent_id = None
        mock_live_span.from_immutable_span.return_value = mock_new_span

        # Call with different experiment_id
        result = copy_trace_to_experiment(trace_dict, experiment_id="exp-456")

        # Should create and return new trace ID
        assert result == "new-trace-456"

        # Verify that LiveSpan.from_immutable_span was called with correct experiment_id
        mock_live_span.from_immutable_span.assert_called_once()
        call_kwargs = mock_live_span.from_immutable_span.call_args[1]
        assert call_kwargs["experiment_id"] == "exp-456"


def test_copy_trace_non_mlflow_experiment_location_creates_new_trace():
    """Test that copy_trace_to_experiment creates new trace when location is not
    MLFLOW_EXPERIMENT.
    """
    trace_dict = {
        "info": {
            "trace_id": "existing-trace-123",
            "trace_location": {
                "type": "INFERENCE_TABLE",
                "inference_table": {"full_table_name": "catalog.schema.table"},
            },
        },
        "data": {
            "spans": [
                {
                    "span_id": "span-1",
                    "trace_id": "existing-trace-123",
                    "parent_id": None,
                    "name": "root_span",
                    "start_time_ns": 1000000000,
                    "end_time_ns": 2000000000,
                    "span_type": "UNKNOWN",
                    "inputs": {},
                    "outputs": {},
                    "attributes": {},
                    "events": [],
                }
            ]
        },
    }

    # Mock the trace manager and LiveSpan creation
    with (
        patch("mlflow.tracing.utils.copy.InMemoryTraceManager") as mock_manager,
        patch("mlflow.tracing.utils.copy.LiveSpan") as mock_live_span,
        patch("mlflow.tracing.utils.copy.Span") as mock_span,
    ):
        # Setup mocks
        mock_manager.get_instance.return_value = MagicMock()

        mock_span_instance = MagicMock()
        mock_span_instance.parent_id = None
        mock_span.from_dict.return_value = mock_span_instance

        mock_new_span = MagicMock()
        mock_new_span.trace_id = "new-trace-456"
        mock_new_span.parent_id = None
        mock_live_span.from_immutable_span.return_value = mock_new_span

        # Call with any experiment_id
        result = copy_trace_to_experiment(trace_dict, experiment_id="exp-456")

        # Should create and return new trace ID (doesn't match, so creates new)
        assert result == "new-trace-456"
