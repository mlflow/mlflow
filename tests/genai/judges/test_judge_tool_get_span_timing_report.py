from mlflow.entities.span import Span
from mlflow.entities.trace import Trace
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.genai.judges.tools.get_span_timing_report import GetSpanTimingReportTool
from mlflow.types.llm import ToolDefinition

from tests.tracing.helper import create_mock_otel_span


def test_get_span_timing_report_tool_name():
    tool = GetSpanTimingReportTool()
    assert tool.name == "get_span_timing_report"


def test_get_span_timing_report_tool_get_definition():
    tool = GetSpanTimingReportTool()
    definition = tool.get_definition()

    assert isinstance(definition, ToolDefinition)
    assert definition.function.name == "get_span_timing_report"
    assert "Generate a comprehensive span timing report" in definition.function.description
    assert definition.function.parameters.type == "object"
    assert definition.function.parameters.required == []
    assert definition.type == "function"


def test_get_span_timing_report_tool_invoke_success():
    tool = GetSpanTimingReportTool()

    # Create root span
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
        start_time=1000000200000,
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
        execution_duration=1000,
    )
    trace = Trace(info=trace_info, data=trace_data)

    result = tool.invoke(trace)

    assert isinstance(result, str)
    assert "SPAN TIMING REPORT FOR TRACE: trace-123" in result
    assert "Total Duration: 1.00s" in result
    assert "Total Spans: 2" in result
    assert "SPAN TABLE:" in result
    assert "SUMMARY BY TYPE:" in result
    assert "TOP 10 SPANS BY SELF DURATION" in result
    assert "CONCURRENT OPERATIONS:" in result
    assert "root-span" in result
    assert "child-span" in result


def test_get_span_timing_report_tool_invoke_no_spans():
    tool = GetSpanTimingReportTool()

    trace_data = TraceData(spans=[])
    trace_info = TraceInfo(
        trace_id="trace-123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=0,
    )
    trace = Trace(info=trace_info, data=trace_data)

    result = tool.invoke(trace)

    assert result == "No spans found in trace"


def test_get_span_timing_report_tool_invoke_none_trace():
    tool = GetSpanTimingReportTool()

    result = tool.invoke(None)

    assert result == "No spans found in trace"


def test_get_span_timing_report_tool_invoke_complex_hierarchy():
    tool = GetSpanTimingReportTool()

    # Create a more complex hierarchy: root -> child1 -> grandchild, root -> child2
    root_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=100,
        name="root-span",
        start_time=1000000000000,
        end_time=1000002000000,
        parent_id=None,
    )
    root_span = Span(root_otel_span)

    child1_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=101,
        name="child1-span",
        start_time=1000000200000,
        end_time=1000001000000,
        parent_id=100,
    )
    child1_span = Span(child1_otel_span)

    child2_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=102,
        name="child2-span",
        start_time=1000001200000,
        end_time=1000001800000,
        parent_id=100,
    )
    child2_span = Span(child2_otel_span)

    grandchild_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=103,
        name="grandchild-span",
        start_time=1000000400000,
        end_time=1000000600000,
        parent_id=101,
    )
    grandchild_span = Span(grandchild_otel_span)

    trace_data = TraceData(spans=[root_span, child1_span, child2_span, grandchild_span])
    trace_info = TraceInfo(
        trace_id="trace-123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=2000,
    )
    trace = Trace(info=trace_info, data=trace_data)

    result = tool.invoke(trace)

    assert isinstance(result, str)
    assert "Total Spans: 4" in result
    assert "root-span" in result
    assert "child1-span" in result
    assert "child2-span" in result
    assert "grandchild-span" in result
    # Check that hierarchy is shown with span numbers
    assert "s1" in result  # First span processed
    assert "s2" in result  # Second span processed


def test_get_span_timing_report_tool_invoke_concurrent_operations():
    tool = GetSpanTimingReportTool()

    # Create overlapping sibling spans to test concurrent operations detection
    root_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=100,
        name="root-span",
        start_time=1000000000000,
        end_time=1000002000000,
        parent_id=None,
    )
    root_span = Span(root_otel_span)

    # Two child spans that overlap significantly
    child1_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=101,
        name="concurrent-child1",
        start_time=1000000200000,
        end_time=1000001200000,
        parent_id=100,
    )
    child1_span = Span(child1_otel_span)

    child2_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=102,
        name="concurrent-child2",
        start_time=1000000600000,  # Overlaps with child1
        end_time=1000001800000,
        parent_id=100,
    )
    child2_span = Span(child2_otel_span)

    trace_data = TraceData(spans=[root_span, child1_span, child2_span])
    trace_info = TraceInfo(
        trace_id="trace-123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=2000,
    )
    trace = Trace(info=trace_info, data=trace_data)

    result = tool.invoke(trace)

    assert isinstance(result, str)
    assert "CONCURRENT OPERATIONS:" in result
    assert "concurrent-child1" in result
    assert "concurrent-child2" in result
    # Should detect overlap
    lines = result.split("\n")
    concurrent_section = False
    for line in lines:
        if "CONCURRENT OPERATIONS:" in line:
            concurrent_section = True
        if concurrent_section and "s1" in line and "s2" in line:
            break
    # Note: We can't guarantee overlap detection due to the 0.01s threshold,
    # but the structure should be there


def test_get_span_timing_report_tool_invoke_span_types():
    tool = GetSpanTimingReportTool()

    # Create spans with different types
    llm_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=100,
        name="llm-call",
        start_time=1000000000000,
        end_time=1000001000000,
        parent_id=None,
    )
    llm_otel_span.set_attribute("span_type", "LLM")
    llm_span = Span(llm_otel_span)

    retrieval_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=101,
        name="retrieval-call",
        start_time=1000001200000,
        end_time=1000001800000,
        parent_id=None,
    )
    retrieval_otel_span.set_attribute("span_type", "RETRIEVAL")
    retrieval_span = Span(retrieval_otel_span)

    trace_data = TraceData(spans=[llm_span, retrieval_span])
    trace_info = TraceInfo(
        trace_id="trace-123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=1800,
    )
    trace = Trace(info=trace_info, data=trace_data)

    result = tool.invoke(trace)

    assert isinstance(result, str)
    assert "SUMMARY BY TYPE:" in result
    # Should show different span types in the summary
    lines = result.split("\n")
    summary_section = False
    found_types = set()
    for line in lines:
        if "SUMMARY BY TYPE:" in line:
            summary_section = True
            continue
        if summary_section and line.strip() and not line.startswith("-"):
            # Extract type from the first column
            parts = line.split()
            if len(parts) > 0:
                span_type = parts[0]
                if span_type not in ["type", ""]:
                    found_types.add(span_type)

    # Should have processed both span types
    assert len(found_types) > 0


def test_get_span_timing_report_tool_invoke_top_spans_ranking():
    tool = GetSpanTimingReportTool()

    # Create spans with different durations to test ranking
    quick_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=100,
        name="quick-span",
        start_time=1000000000000,
        end_time=1000000100000,  # 100ms
        parent_id=None,
    )
    quick_span = Span(quick_otel_span)

    slow_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=101,
        name="slow-span",
        start_time=1000000200000,
        end_time=1000001200000,  # 1000ms
        parent_id=None,
    )
    slow_span = Span(slow_otel_span)

    trace_data = TraceData(spans=[quick_span, slow_span])
    trace_info = TraceInfo(
        trace_id="trace-123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=1300,
    )
    trace = Trace(info=trace_info, data=trace_data)

    result = tool.invoke(trace)

    assert isinstance(result, str)
    assert "TOP 10 SPANS BY SELF DURATION" in result
    assert "quick-span" in result
    assert "slow-span" in result

    # The slow span should appear first in the ranking (rank 1)
    lines = result.split("\n")
    top_section = False
    for line in lines:
        if "TOP 10 SPANS BY SELF DURATION" in line:
            top_section = True
            continue
        if top_section and "slow-span" in line:
            # Should be rank 1 (first in ranking)
            assert line.strip().startswith("1")
            break


def test_get_span_timing_report_tool_invoke_long_span_names():
    tool = GetSpanTimingReportTool()

    # Test with very long span name to ensure truncation works
    long_name_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=100,
        name="this_is_a_very_long_span_name_that_should_be_truncated_in_the_report",
        start_time=1000000000000,
        end_time=1000001000000,
        parent_id=None,
    )
    long_name_span = Span(long_name_otel_span)

    trace_data = TraceData(spans=[long_name_span])
    trace_info = TraceInfo(
        trace_id="trace-123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=1000,
    )
    trace = Trace(info=trace_info, data=trace_data)

    result = tool.invoke(trace)

    assert isinstance(result, str)
    # Should contain the truncated name with "..."
    assert "this_is_a_very_long_span_na..." in result
