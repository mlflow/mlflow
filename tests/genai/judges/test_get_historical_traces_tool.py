"""
Tests for GetHistoricalTracesTool.

This module tests the functionality of the GetHistoricalTracesTool class,
including successful trace retrieval, error handling, and parameter validation.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.tools.get_historical_traces import GetHistoricalTracesTool
from mlflow.genai.judges.tools.types import HistoricalTrace
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.types.llm import ToolDefinition


def test_get_historical_traces_tool_name():
    """Test that the tool returns the correct name."""
    tool = GetHistoricalTracesTool()
    assert tool.name == "get_historical_traces"


def test_get_historical_traces_tool_get_definition():
    """Test that the tool returns a valid tool definition."""
    tool = GetHistoricalTracesTool()
    definition = tool.get_definition()

    assert isinstance(definition, ToolDefinition)
    assert definition.function.name == "get_historical_traces"
    assert "historical traces" in definition.function.description.lower()
    assert "multi-turn evaluation" in definition.function.description.lower()
    assert definition.function.parameters.type == "object"
    assert len(definition.function.parameters.required) == 0
    assert definition.type == "function"

    # Check parameter definitions
    params = definition.function.parameters.properties
    assert "max_results" in params
    assert "order_by" in params

    assert params["max_results"].type == "integer"
    assert params["order_by"].type == "array"


def create_mock_trace(trace_id: str, experiment_id: str = "exp-123", session_id: str | None = None):
    """Helper function to create mock trace objects."""
    tags = {}
    if session_id:
        tags["session.id"] = session_id

    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation.from_experiment_id(experiment_id),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=250,
        tags=tags,
    )
    return Trace(info=trace_info, data=None)


def create_mock_search_result():
    """Helper function to create mock search_traces result."""
    # Create mock trace data for DataFrame
    trace_data = [
        {
            "trace_id": "trace-1",
            "trace": (
                '{"info": {"trace_id": "trace-1", "experiment_id": "exp-123", '
                '"request_time": 1000, "state": "OK", "execution_duration": 100, '
                '"tags": {"session.id": "session-123"}}}'
            ),
            "request": "What is machine learning?",
            "response": "Machine learning is a subset of AI...",
        },
        {
            "trace_id": "trace-2",
            "trace": (
                '{"info": {"trace_id": "trace-2", "experiment_id": "exp-123", '
                '"request_time": 2000, "state": "OK", "execution_duration": 150, '
                '"tags": {"session.id": "session-123"}}}'
            ),
            "request": "Can you give an example?",
            "response": "Sure! A common example is...",
        },
    ]
    return pd.DataFrame(trace_data)


@patch("mlflow.search_traces")
def test_get_historical_traces_tool_invoke_success(mock_search_traces):
    """Test successful retrieval of historical traces."""
    tool = GetHistoricalTracesTool()

    # Create test trace with session ID
    current_trace = create_mock_trace("current-trace", "exp-123", "session-123")

    # Mock search_traces return value
    mock_df = create_mock_search_result()
    mock_search_traces.return_value = mock_df

    # Mock Trace.from_json to return proper trace objects
    mock_trace_1 = create_mock_trace("trace-1", "exp-123", "session-123")
    mock_trace_2 = create_mock_trace("trace-2", "exp-123", "session-123")

    with patch.object(Trace, "from_json", side_effect=[mock_trace_1, mock_trace_2]):
        result = tool.invoke(current_trace)

    # Verify the result
    assert len(result) == 2
    assert all(isinstance(ht, HistoricalTrace) for ht in result)

    # Check first historical trace
    assert result[0].trace_info.trace_id == "trace-1"
    assert result[0].request == "What is machine learning?"
    assert result[0].response == "Machine learning is a subset of AI..."

    # Check second historical trace
    assert result[1].trace_info.trace_id == "trace-2"
    assert result[1].request == "Can you give an example?"
    assert result[1].response == "Sure! A common example is..."

    # Verify search_traces was called with correct parameters
    mock_search_traces.assert_called_once_with(
        experiment_ids=["exp-123"],
        filter_string="tags.`session.id` = 'session-123'",
        max_results=20,
        order_by=["timestamp ASC"],
        extract_fields=["trace_id", "trace", "request", "response"],
    )


@patch("mlflow.search_traces")
def test_get_historical_traces_tool_invoke_custom_parameters(mock_search_traces):
    """Test tool invocation with custom parameters."""
    tool = GetHistoricalTracesTool()

    # Create test trace with session ID
    current_trace = create_mock_trace("current-trace", "exp-123", "session-456")

    # Mock search_traces return value
    mock_df = create_mock_search_result()
    mock_search_traces.return_value = mock_df

    # Mock Trace.from_json
    mock_trace = create_mock_trace("trace-1", "exp-456", "session-456")

    with patch.object(Trace, "from_json", return_value=mock_trace):
        tool.invoke(current_trace, max_results=50, order_by=["timestamp DESC"])

    # Verify search_traces was called with custom parameters
    mock_search_traces.assert_called_once_with(
        experiment_ids=["exp-123"],  # Uses current trace's experiment
        filter_string="tags.`session.id` = 'session-456'",
        max_results=50,
        order_by=["timestamp DESC"],
        extract_fields=["trace_id", "trace", "request", "response"],
    )


def test_get_historical_traces_tool_invoke_no_session_id():
    """Test that tool raises error when no session.id is found in trace tags."""
    tool = GetHistoricalTracesTool()

    # Create trace without session ID
    current_trace = create_mock_trace("current-trace", "exp-123", session_id=None)

    with pytest.raises(MlflowException, match="No session.id found in trace tags") as exc_info:
        tool.invoke(current_trace)

    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_get_historical_traces_tool_invoke_no_experiment_id():
    """Test error when trace has no experiment_id."""
    tool = GetHistoricalTracesTool()

    # Create trace with session ID but no experiment ID
    trace_info = TraceInfo(
        trace_id="current-trace",
        trace_location=TraceLocation.from_experiment_id(None),  # No experiment ID
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=250,
        tags={"session.id": "session-123"},
    )
    current_trace = Trace(info=trace_info, data=None)

    with pytest.raises(MlflowException, match="Current trace has no experiment_id") as exc_info:
        tool.invoke(current_trace)

    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_get_historical_traces_tool_invoke_invalid_session_id():
    """Test error when session.id has invalid format."""
    tool = GetHistoricalTracesTool()

    # Create trace with invalid session ID (contains special characters)
    current_trace = create_mock_trace("current-trace", "exp-123", "session@123!invalid")

    with pytest.raises(MlflowException, match="Invalid session ID format") as exc_info:
        tool.invoke(current_trace)

    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_get_historical_traces_tool_invoke_non_mlflow_experiment():
    """Test error when trace is not from an MLflow experiment."""
    tool = GetHistoricalTracesTool()

    # Create trace with session ID but non-MLflow experiment trace location
    # Simulate a trace from inference table
    trace_location = TraceLocation(
        type="INFERENCE_TABLE",  # Not MLFLOW_EXPERIMENT
        mlflow_experiment=None,
        inference_table=None,
    )
    trace_info = TraceInfo(
        trace_id="current-trace",
        trace_location=trace_location,
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=250,
        tags={"session.id": "session-123"},
    )
    current_trace = Trace(info=trace_info, data=None)

    with pytest.raises(
        MlflowException, match="Current trace is not from an MLflow experiment"
    ) as exc_info:
        tool.invoke(current_trace)

    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@patch("mlflow.search_traces")
def test_get_historical_traces_tool_invoke_search_error(mock_search_traces):
    """Test that tool handles search_traces errors gracefully."""
    tool = GetHistoricalTracesTool()

    # Create test trace with session ID
    current_trace = create_mock_trace("current-trace", "exp-123", "session-123")

    # Mock search_traces to raise an exception
    mock_search_traces.side_effect = Exception("Search failed")

    with pytest.raises(
        MlflowException, match="Failed to search historical traces for session session-123"
    ) as exc_info:
        tool.invoke(current_trace)

    assert exc_info.value.error_code == "INTERNAL_ERROR"


@patch("mlflow.search_traces")
def test_get_historical_traces_tool_invoke_empty_result(mock_search_traces):
    """Test tool behavior when no historical traces are found."""
    tool = GetHistoricalTracesTool()

    # Create test trace with session ID
    current_trace = create_mock_trace("current-trace", "exp-123", "session-123")

    # Mock empty search result
    mock_df = pd.DataFrame(columns=["trace_id", "trace", "request", "response"])
    mock_search_traces.return_value = mock_df

    result = tool.invoke(current_trace)

    assert result == []
    assert len(result) == 0


@patch("mlflow.search_traces")
def test_get_historical_traces_tool_invoke_trace_parsing_error(mock_search_traces):
    """Test that tool handles individual trace parsing errors gracefully."""
    tool = GetHistoricalTracesTool()

    # Create test trace with session ID
    current_trace = create_mock_trace("current-trace", "exp-123", "session-123")

    # Create DataFrame with one good and one bad trace
    trace_data = [
        {
            "trace_id": "trace-1",
            "trace": (
                '{"info": {"trace_id": "trace-1", "experiment_id": "exp-123", '
                '"request_time": 1000, "state": "OK", "execution_duration": 100, '
                '"tags": {"session.id": "session-123"}}}'
            ),
            "request": "Valid request",
            "response": "Valid response",
        },
        {
            "trace_id": "trace-2",
            "trace": "invalid-json",  # This will cause parsing error
            "request": "Invalid request",
            "response": "Invalid response",
        },
    ]
    mock_df = pd.DataFrame(trace_data)
    mock_search_traces.return_value = mock_df

    # Mock Trace.from_json to succeed for first trace, fail for second
    mock_trace = create_mock_trace("trace-1", "exp-123", "session-123")

    def mock_from_json(json_str):
        if "trace-1" in json_str:
            return mock_trace
        else:
            raise ValueError("Invalid JSON")

    with patch.object(Trace, "from_json", side_effect=mock_from_json):
        result = tool.invoke(current_trace)

    # Should return only the valid trace, skip the invalid one
    assert len(result) == 1
    assert result[0].trace_info.trace_id == "trace-1"
    assert result[0].request == "Valid request"
    assert result[0].response == "Valid response"


@patch("mlflow.search_traces")
def test_get_historical_traces_tool_invoke_null_request_response(mock_search_traces):
    """Test tool behavior with null request/response values."""
    tool = GetHistoricalTracesTool()

    # Create test trace with session ID
    current_trace = create_mock_trace("current-trace", "exp-123", "session-123")

    # Create DataFrame with null values
    trace_data = [
        {
            "trace_id": "trace-1",
            "trace": (
                '{"info": {"trace_id": "trace-1", "experiment_id": "exp-123", '
                '"request_time": 1000, "state": "OK", "execution_duration": 100, '
                '"tags": {"session.id": "session-123"}}}'
            ),
            "request": None,
            "response": None,
        },
    ]
    mock_df = pd.DataFrame(trace_data)
    mock_search_traces.return_value = mock_df

    # Mock Trace.from_json
    mock_trace = create_mock_trace("trace-1", "exp-123", "session-123")

    with patch.object(Trace, "from_json", return_value=mock_trace):
        result = tool.invoke(current_trace)

    # Should handle null values gracefully by converting to empty strings
    assert len(result) == 1
    assert result[0].request == ""
    assert result[0].response == ""
