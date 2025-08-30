"""Tests for the ListSpansTool implementation."""

from unittest import mock

import pytest

from mlflow.entities.span import Span
from mlflow.entities.trace import Trace
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.genai.judges.tools.constants import ToolNames
from mlflow.genai.judges.tools.list_spans import ListSpansTool


@pytest.fixture
def mock_trace_with_spans():
    """Fixture that creates a test Trace object with multiple spans."""

    # Create mock spans directly without using from_dict to avoid encoding issues
    mock_span = mock.Mock(spec=Span)
    mock_span.name = "test_span"
    mock_span.span_id = "test-span-id"
    mock_span.trace_id = "test-trace"
    mock_span.to_dict.return_value = {
        "name": "test_span",
        "span_id": "test-span-id",
        "trace_id": "test-trace",
        "attributes": {"mlflow.spanType": "CHAIN"},
    }

    trace_info = TraceInfo(
        trace_id="test-trace",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
    )
    trace_data = TraceData(request="{}", response="{}", spans=[mock_span])
    return Trace(info=trace_info, data=trace_data)


def test_list_spans_tool_name():
    """Test that the tool returns the correct name."""
    tool = ListSpansTool()
    assert tool.name == ToolNames.LIST_SPANS


def test_list_spans_tool_get_definition():
    """Test that the tool returns a valid definition."""
    tool = ListSpansTool()
    definition = tool.get_definition()

    assert definition.function.name == ToolNames.LIST_SPANS
    assert "Retrieve all spans from the trace" in definition.function.description
    assert definition.function.parameters.type == "object"
    assert definition.function.parameters.properties == {}
    assert definition.function.parameters.required == []


def test_list_spans_tool_invoke_success(mock_trace_with_spans):
    """Test that the tool successfully returns span data."""
    tool = ListSpansTool()
    result = tool.invoke(mock_trace_with_spans)

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], dict)
    assert result[0]["name"] == "test_span"


def test_list_spans_tool_invoke_empty_trace():
    """Test that the tool handles traces with no spans."""
    trace_info = TraceInfo(
        trace_id="empty-trace",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
    )
    trace_data = TraceData(request="{}", response="{}", spans=[])
    empty_trace = Trace(info=trace_info, data=trace_data)

    tool = ListSpansTool()
    result = tool.invoke(empty_trace)

    assert isinstance(result, list)
    assert len(result) == 0


def test_list_spans_tool_invoke_returns_dict_format(mock_trace_with_spans):
    """Test that the tool returns spans in dictionary format."""
    tool = ListSpansTool()
    result = tool.invoke(mock_trace_with_spans)

    span_dict = result[0]
    assert "name" in span_dict
    assert "trace_id" in span_dict
    assert "span_id" in span_dict
    assert "attributes" in span_dict
