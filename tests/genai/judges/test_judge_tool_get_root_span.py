import pytest

from mlflow.entities.span import Span
from mlflow.entities.trace import Trace
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.genai.judges.tools.get_root_span import GetRootSpanTool
from mlflow.genai.judges.tools.types import SpanResult
from mlflow.types.llm import ToolDefinition

from tests.tracing.helper import create_mock_otel_span


def test_get_root_span_tool_name():
    tool = GetRootSpanTool()
    assert tool.name == "get_root_span"


def test_get_root_span_tool_get_definition():
    tool = GetRootSpanTool()
    definition = tool.get_definition()

    assert isinstance(definition, ToolDefinition)
    assert definition.function.name == "get_root_span"
    assert "Retrieve the root span of the trace" in definition.function.description
    assert definition.function.parameters.type == "object"
    assert definition.function.parameters.required == []
    assert definition.type == "function"


def test_get_root_span_tool_invoke_success():
    tool = GetRootSpanTool()

    # Create root span (no parent)
    root_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=100,
        name="root-span",
        start_time=1000000000000,
        end_time=1000001000000,
        parent_id=None,
    )
    root_span = Span(root_otel_span)

    # Create child span
    child_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=101,
        name="child-span",
        start_time=1000000500000,
        end_time=1000000800000,
        parent_id=100,
    )
    child_span = Span(child_otel_span)

    trace_data = TraceData(spans=[root_span, child_span])
    trace_info = TraceInfo(
        trace_id="trace-123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=250,
    )
    trace = Trace(info=trace_info, data=trace_data)

    result = tool.invoke(trace)

    assert isinstance(result, SpanResult)
    assert result.span_id == root_span.span_id
    assert result.content is not None
    assert result.error is None
    assert "root-span" in result.content


def test_get_root_span_tool_invoke_no_spans():
    tool = GetRootSpanTool()

    trace_data = TraceData(spans=[])
    trace_info = TraceInfo(
        trace_id="trace-123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=250,
    )
    trace = Trace(info=trace_info, data=trace_data)

    result = tool.invoke(trace)

    assert isinstance(result, SpanResult)
    assert result.span_id is None
    assert result.content is None
    assert result.error == "Trace has no spans"


def test_get_root_span_tool_invoke_no_root_span():
    tool = GetRootSpanTool()

    # Create only child spans (all have parent_id)
    child1_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=101,
        name="child1-span",
        start_time=1000000000000,
        end_time=1000001000000,
        parent_id=100,  # Has parent
    )
    child1_span = Span(child1_otel_span)

    child2_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=102,
        name="child2-span",
        start_time=1000000000000,
        end_time=1000001000000,
        parent_id=100,  # Has parent
    )
    child2_span = Span(child2_otel_span)

    trace_data = TraceData(spans=[child1_span, child2_span])
    trace_info = TraceInfo(
        trace_id="trace-123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=250,
    )
    trace = Trace(info=trace_info, data=trace_data)

    result = tool.invoke(trace)

    assert isinstance(result, SpanResult)
    assert result.span_id is None
    assert result.content is None
    assert result.error == "No root span found in trace"


def test_get_root_span_tool_invoke_with_attributes_filter():
    tool = GetRootSpanTool()

    root_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=100,
        name="root-span",
        start_time=1000000000000,
        end_time=1000001000000,
    )
    root_otel_span.set_attribute("key1", "value1")
    root_otel_span.set_attribute("key2", "value2")
    root_otel_span.set_attribute("key3", "value3")
    root_span = Span(root_otel_span)

    trace_data = TraceData(spans=[root_span])
    trace_info = TraceInfo(
        trace_id="trace-123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=250,
    )
    trace = Trace(info=trace_info, data=trace_data)

    result = tool.invoke(trace, attributes_to_fetch=["key1", "key3"])

    assert isinstance(result, SpanResult)
    assert result.span_id == root_span.span_id
    assert result.content is not None
    assert "key1" in result.content
    assert "key3" in result.content
    assert "key2" not in result.content


def test_get_root_span_tool_invoke_with_pagination():
    tool = GetRootSpanTool()

    root_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=100,
        name="root-span-with-long-content",
        start_time=1000000000000,
        end_time=1000001000000,
    )
    root_otel_span.set_attribute("large_data", "x" * 50000)
    root_span = Span(root_otel_span)

    trace_data = TraceData(spans=[root_span])
    trace_info = TraceInfo(
        trace_id="trace-123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=250,
    )
    trace = Trace(info=trace_info, data=trace_data)

    all_content = ""
    page_token = None
    max_iterations = 100
    iterations = 0

    while iterations < max_iterations:
        result = tool.invoke(trace, max_content_length=1000, page_token=page_token)

        assert isinstance(result, SpanResult)
        assert result.span_id == root_span.span_id
        assert result.content is not None
        assert result.error is None

        all_content += result.content

        if result.page_token is None:
            break

        page_token = result.page_token
        iterations += 1

    # Verify the paginated content matches a complete fetch
    complete_result = tool.invoke(trace, max_content_length=len(all_content) + 1000)
    assert all_content == complete_result.content


def test_get_root_span_tool_invoke_invalid_page_token():
    """Test that invalid page tokens raise MlflowException."""
    from mlflow.exceptions import MlflowException

    tool = GetRootSpanTool()

    root_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=100,
        name="root-span",
        start_time=1000000000000,
        end_time=1000001000000,
    )
    root_span = Span(root_otel_span)

    trace_data = TraceData(spans=[root_span])
    trace_info = TraceInfo(
        trace_id="trace-123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=250,
    )
    trace = Trace(info=trace_info, data=trace_data)

    # Test with invalid string token - should raise exception
    with pytest.raises(
        MlflowException, match="Invalid page_token 'invalid': must be a valid integer"
    ):
        tool.invoke(trace, page_token="invalid")

    # Test with non-string invalid token - should raise exception
    with pytest.raises(
        MlflowException, match="Invalid page_token '\\[\\]': must be a valid integer"
    ):
        tool.invoke(trace, page_token=[])
