"""Tests for the ListSpansTool implementation."""

from unittest import mock

import pytest

from mlflow.entities.span import Span
from mlflow.entities.span_status import SpanStatus, SpanStatusCode
from mlflow.entities.trace import Trace
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.genai.judges.tools.list_spans import ListSpansResult, ListSpansTool
from mlflow.genai.judges.tools.types import SpanInfo
from mlflow.types.llm import ToolDefinition


def test_list_spans_tool_name():
    tool = ListSpansTool()
    assert tool.name == "list_spans"


def test_list_spans_tool_get_definition():
    tool = ListSpansTool()
    definition = tool.get_definition()

    assert isinstance(definition, ToolDefinition)
    assert definition.function.name == "list_spans"
    assert "List information about spans within a trace" in definition.function.description
    assert definition.function.parameters.type == "object"
    assert definition.function.parameters.required == []
    assert definition.type == "function"


@pytest.fixture
def mock_trace_with_spans():
    """Fixture that creates a test Trace object with multiple spans."""
    # Create mock spans with required properties
    mock_span1 = mock.Mock(spec=Span)
    mock_span1.span_id = "span-1"
    mock_span1.name = "root_span"
    mock_span1.span_type = "CHAIN"
    mock_span1.start_time_ns = 1234567890000000000
    mock_span1.end_time_ns = 1234567891000000000
    mock_span1.parent_id = None
    mock_span1.status = SpanStatus(SpanStatusCode.OK)
    mock_span1.attributes = {"mlflow.spanType": "CHAIN", "custom_attr": "value1"}

    mock_span2 = mock.Mock(spec=Span)
    mock_span2.span_id = "span-2"
    mock_span2.name = "child_span"
    mock_span2.span_type = "TOOL"
    mock_span2.start_time_ns = 1234567890500000000
    mock_span2.end_time_ns = 1234567890800000000
    mock_span2.parent_id = "span-1"
    mock_span2.status = SpanStatus(SpanStatusCode.OK)
    mock_span2.attributes = {"mlflow.spanType": "TOOL"}

    trace_info = TraceInfo(
        trace_id="test-trace",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
    )
    trace_data = TraceData(request="{}", response="{}", spans=[mock_span1, mock_span2])
    return Trace(info=trace_info, data=trace_data)


def test_list_spans_tool_invoke_success(mock_trace_with_spans):
    """Test that the tool successfully returns span data."""
    tool = ListSpansTool()
    result = tool.invoke(mock_trace_with_spans)

    assert isinstance(result, ListSpansResult)
    assert len(result.spans) == 2
    assert result.next_page_token is None

    # Check first span
    span1 = result.spans[0]
    assert isinstance(span1, SpanInfo)
    assert span1.span_id == "span-1"
    assert span1.name == "root_span"
    assert span1.span_type == "CHAIN"
    assert span1.is_root is True
    assert span1.parent_id is None
    assert span1.duration_ms == 1000.0  # 1 second
    assert span1.attribute_names == ["mlflow.spanType", "custom_attr"]

    # Check second span
    span2 = result.spans[1]
    assert span2.span_id == "span-2"
    assert span2.name == "child_span"
    assert span2.span_type == "TOOL"
    assert span2.is_root is False
    assert span2.parent_id == "span-1"
    assert span2.duration_ms == 300.0  # 0.3 seconds


def test_list_spans_tool_invoke_none_trace():
    """Test that the tool handles None trace gracefully."""
    tool = ListSpansTool()
    result = tool.invoke(None)

    assert isinstance(result, ListSpansResult)
    assert len(result.spans) == 0
    assert result.next_page_token is None


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

    assert isinstance(result, ListSpansResult)
    assert len(result.spans) == 0
    assert result.next_page_token is None


def test_list_spans_tool_invoke_with_pagination(mock_trace_with_spans):
    """Test pagination functionality."""
    tool = ListSpansTool()

    # Test with max_results=1
    result = tool.invoke(mock_trace_with_spans, max_results=1)
    assert len(result.spans) == 1
    assert result.next_page_token == "1"
    assert result.spans[0].name == "root_span"

    # Test second page
    result = tool.invoke(mock_trace_with_spans, max_results=1, page_token="1")
    assert len(result.spans) == 1
    assert result.next_page_token is None
    assert result.spans[0].name == "child_span"


def test_list_spans_tool_invoke_invalid_page_token(mock_trace_with_spans):
    """Test that invalid page tokens raise MlflowException."""
    from mlflow.exceptions import MlflowException

    tool = ListSpansTool()

    # Test with invalid string token - should raise exception
    with pytest.raises(
        MlflowException, match="Invalid page_token 'invalid': must be a valid integer"
    ):
        tool.invoke(mock_trace_with_spans, page_token="invalid")

    # Test with non-string invalid token - should raise exception
    with pytest.raises(
        MlflowException, match="Invalid page_token '\\[\\]': must be a valid integer"
    ):
        tool.invoke(mock_trace_with_spans, page_token=[])
