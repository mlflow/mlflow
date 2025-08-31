from mlflow.entities.span import Span
from mlflow.entities.trace import Trace
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.genai.judges.tools.get_span import GetSpanResult, GetSpanTool
from mlflow.types.llm import ToolDefinition

from tests.tracing.helper import create_mock_otel_span


def test_get_span_tool_name():
    tool = GetSpanTool()
    assert tool.name == "get_span"


def test_get_span_tool_get_definition():
    tool = GetSpanTool()
    definition = tool.get_definition()

    assert isinstance(definition, ToolDefinition)
    assert definition.function.name == "get_span"
    assert "Retrieve a specific span by its ID" in definition.function.description
    assert definition.function.parameters.type == "object"
    assert definition.function.parameters.required == ["span_id"]
    assert definition.type == "function"


def test_get_span_tool_invoke_success():
    tool = GetSpanTool()

    otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=123,
        name="test-span",
        start_time=1000000000000,
        end_time=1000001000000,
    )
    span = Span(otel_span)

    trace_data = TraceData(spans=[span])
    trace_info = TraceInfo(
        trace_id="trace-123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=250,
    )
    trace = Trace(info=trace_info, data=trace_data)

    result = tool.invoke(trace, span.span_id)

    assert isinstance(result, GetSpanResult)
    assert result.span_id == span.span_id
    assert result.content is not None
    assert result.error is None
    assert "test-span" in result.content


def test_get_span_tool_invoke_span_not_found():
    tool = GetSpanTool()

    otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=123,
        name="test-span",
        start_time=1000000000000,
        end_time=1000001000000,
    )
    span = Span(otel_span)

    trace_data = TraceData(spans=[span])
    trace_info = TraceInfo(
        trace_id="trace-123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=250,
    )
    trace = Trace(info=trace_info, data=trace_data)

    result = tool.invoke(trace, "nonexistent-span")

    assert isinstance(result, GetSpanResult)
    assert result.span_id is None
    assert result.content is None
    assert result.error == "Span with ID 'nonexistent-span' not found in trace"


def test_get_span_tool_invoke_no_spans():
    tool = GetSpanTool()

    trace_data = TraceData(spans=[])
    trace_info = TraceInfo(
        trace_id="trace-123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=250,
    )
    trace = Trace(info=trace_info, data=trace_data)

    result = tool.invoke(trace, "span-123")

    assert isinstance(result, GetSpanResult)
    assert result.span_id is None
    assert result.content is None
    assert result.error == "Trace has no spans"


def test_get_span_tool_invoke_with_attributes_filter():
    tool = GetSpanTool()

    otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=123,
        name="test-span",
        start_time=1000000000000,
        end_time=1000001000000,
    )
    otel_span.set_attribute("key1", "value1")
    otel_span.set_attribute("key2", "value2")
    otel_span.set_attribute("key3", "value3")
    span = Span(otel_span)

    trace_data = TraceData(spans=[span])
    trace_info = TraceInfo(
        trace_id="trace-123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=250,
    )
    trace = Trace(info=trace_info, data=trace_data)

    result = tool.invoke(trace, span.span_id, attributes_to_fetch=["key1", "key3"])

    assert isinstance(result, GetSpanResult)
    assert result.span_id == span.span_id
    assert result.content is not None
    assert "key1" in result.content
    assert "key3" in result.content
    assert "key2" not in result.content


def test_get_span_tool_invoke_with_pagination():
    tool = GetSpanTool()

    otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=123,
        name="test-span-with-long-content",
        start_time=1000000000000,
        end_time=1000001000000,
    )
    otel_span.set_attribute("large_data", "x" * 50000)
    span = Span(otel_span)

    trace_data = TraceData(spans=[span])
    trace_info = TraceInfo(
        trace_id="trace-123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=250,
    )
    trace = Trace(info=trace_info, data=trace_data)

    result = tool.invoke(trace, span.span_id, max_content_length=1000)

    assert isinstance(result, GetSpanResult)
    assert result.span_id == span.span_id
    assert result.content is not None
    assert len(result.content) <= 1000
    assert result.page_token is not None
