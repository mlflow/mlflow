from mlflow.entities.span import Span
from mlflow.entities.trace import Trace
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.genai.judges.tools.get_span_performance_and_timing_report import (
    GetSpanPerformanceAndTimingReportTool,
)
from mlflow.types.llm import ToolDefinition

from tests.tracing.helper import create_mock_otel_span


def test_get_span_timing_report_tool_name():
    tool = GetSpanPerformanceAndTimingReportTool()
    assert tool.name == "get_span_performance_and_timing_report"


def test_get_span_timing_report_tool_get_definition():
    tool = GetSpanPerformanceAndTimingReportTool()
    definition = tool.get_definition()

    assert isinstance(definition, ToolDefinition)
    assert definition.function.name == "get_span_performance_and_timing_report"
    assert "Generate a comprehensive span timing report" in definition.function.description
    assert definition.function.parameters.type == "object"
    assert definition.function.parameters.required == []
    assert definition.type == "function"


def test_get_span_timing_report_tool_invoke_success():
    tool = GetSpanPerformanceAndTimingReportTool()

    root_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=100,
        name="root-span",
        start_time=1000000000000,
        end_time=1000001000000,
        parent_id=None,
    )
    root_span = Span(root_otel_span)

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
    tool = GetSpanPerformanceAndTimingReportTool()

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
    tool = GetSpanPerformanceAndTimingReportTool()

    result = tool.invoke(None)

    assert result == "No spans found in trace"


def test_get_span_timing_report_tool_invoke_complex_hierarchy():
    tool = GetSpanPerformanceAndTimingReportTool()

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
    assert "s1" in result
    assert "s2" in result


def test_get_span_timing_report_tool_invoke_concurrent_operations():
    tool = GetSpanPerformanceAndTimingReportTool()

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
        start_time=1000000600000,
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


def test_get_span_timing_report_tool_invoke_span_types():
    tool = GetSpanPerformanceAndTimingReportTool()

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
    lines = result.split("\n")
    summary_section = False
    found_types = set()
    for line in lines:
        if "SUMMARY BY TYPE:" in line:
            summary_section = True
            continue
        if summary_section and line.strip() and not line.startswith("-"):
            parts = line.split()
            if len(parts) > 0:
                span_type = parts[0]
                if span_type not in ["type", ""]:
                    found_types.add(span_type)

    assert len(found_types) > 0


def test_get_span_timing_report_tool_invoke_top_spans_ranking():
    tool = GetSpanPerformanceAndTimingReportTool()

    quick_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=100,
        name="quick-span",
        start_time=1000000000000,
        end_time=1000000100000,
        parent_id=None,
    )
    quick_span = Span(quick_otel_span)

    slow_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=101,
        name="slow-span",
        start_time=1000000200000,
        end_time=1000001200000,
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

    lines = result.split("\n")
    top_section = False
    for line in lines:
        if "TOP 10 SPANS BY SELF DURATION" in line:
            top_section = True
            continue
        if top_section and "slow-span" in line:
            assert line.strip().startswith("1")
            break


def test_get_span_timing_report_tool_invoke_long_span_names():
    tool = GetSpanPerformanceAndTimingReportTool()

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
    assert "this_is_a_very_long_span_na..." in result


def test_get_span_timing_report_tool_self_duration_calculation():
    tool = GetSpanPerformanceAndTimingReportTool()

    parent_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=100,
        name="parent",
        start_time=1000000000000000000,
        end_time=1000001000000000000,
        parent_id=None,
    )
    parent_span = Span(parent_otel_span)

    child1_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=101,
        name="child1",
        start_time=1000000100000000000,
        end_time=1000000300000000000,
        parent_id=100,
    )
    child1_span = Span(child1_otel_span)

    child2_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=102,
        name="child2",
        start_time=1000000250000000000,
        end_time=1000000450000000000,
        parent_id=100,
    )
    child2_span = Span(child2_otel_span)

    trace_data = TraceData(spans=[parent_span, child1_span, child2_span])
    trace_info = TraceInfo(
        trace_id="trace-123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=1000,
    )
    trace = Trace(info=trace_info, data=trace_data)

    result = tool.invoke(trace)

    lines = result.split("\n")
    for line in lines:
        if "parent" in line and "s1" in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if "." in part and i > 3:
                    self_dur = float(parts[4])
                    assert 640 < self_dur < 660
                    break
            break


def test_get_span_timing_report_tool_empty_trace_data():
    tool = GetSpanPerformanceAndTimingReportTool()

    trace = Trace(info=None, data=None)
    result = tool.invoke(trace)

    assert result == "No spans found in trace"


def test_get_span_timing_report_tool_concurrent_pairs_limit():
    tool = GetSpanPerformanceAndTimingReportTool()

    root_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=100,
        name="root",
        start_time=1000000000000,
        end_time=1000010000000,
        parent_id=None,
    )
    root_span = Span(root_otel_span)

    concurrent_spans = []
    for i in range(30):
        otel_span = create_mock_otel_span(
            trace_id=12345,
            span_id=101 + i,
            name=f"concurrent-{i}",
            start_time=1000001000000 + i * 100000,
            end_time=1000008000000 + i * 100000,
            parent_id=100,
        )
        concurrent_spans.append(Span(otel_span))

    trace_data = TraceData(spans=[root_span] + concurrent_spans)
    trace_info = TraceInfo(
        trace_id="trace-123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=10000,
    )
    trace = Trace(info=trace_info, data=trace_data)

    result = tool.invoke(trace)

    concurrent_count = result.count("concurrent-")
    assert concurrent_count <= 2 * tool.MAX_CONCURRENT_PAIRS + 10


def test_truncate_name_method_directly():
    tool = GetSpanPerformanceAndTimingReportTool()

    test_cases = [
        ("short", "short"),
        ("a" * 30, "a" * 30),
        ("a" * 31, "a" * 27 + "..."),
        ("a" * 50, "a" * 27 + "..."),
        ("", ""),
        ("123456789012345678901234567890", "123456789012345678901234567890"),
        ("1234567890123456789012345678901", "123456789012345678901234567..."),
    ]

    for input_name, expected in test_cases:
        result = tool._truncate_name(input_name)
        assert result == expected
        assert len(result) <= 30


def test_truncation_in_multiple_report_sections():
    tool = GetSpanPerformanceAndTimingReportTool()

    spans = []
    test_names = [
        ("short", False),
        ("a" * 30, False),
        ("a" * 31, True),
        ("this_is_a_very_long_span_name_that_should_definitely_be_truncated", True),
    ]

    for i, (name, should_truncate) in enumerate(test_names):
        otel_span = create_mock_otel_span(
            trace_id=12345,
            span_id=100 + i,
            name=name,
            start_time=1000000000000 + i * 1000000000,
            end_time=1000000000000 + (i + 1) * 1000000000,
            parent_id=None,
        )
        spans.append(Span(otel_span))

    trace_data = TraceData(spans=spans)
    trace_info = TraceInfo(
        trace_id="test-truncation",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=len(test_names) * 1000,
    )
    trace = Trace(info=trace_info, data=trace_data)

    result = tool.invoke(trace)

    for name, should_truncate in test_names:
        if should_truncate:
            truncated = name[:27] + "..." if len(name) > 30 else name
            assert truncated in result
            if len(name) > 30:
                assert name not in result
        else:
            assert name in result


def test_truncation_edge_cases():
    tool = GetSpanPerformanceAndTimingReportTool()

    unicode_name = "测试" + "很" * 30
    result = tool._truncate_name(unicode_name)
    assert len(result) <= 30
    if len(unicode_name) > 30:
        assert result.endswith("...")

    special = "!@#$%^&*()" * 5
    result = tool._truncate_name(special)
    assert result == special[:27] + "..."
    assert len(result) == 30

    spaces = "word " * 10
    result = tool._truncate_name(spaces)
    if len(spaces) > 30:
        assert result == spaces[:27] + "..."
        assert len(result) == 30

    mixed = "test_123_" * 5
    result = tool._truncate_name(mixed)
    assert result == mixed[:27] + "..."
    assert len(result) == 30


def test_truncation_preserves_table_formatting():
    tool = GetSpanPerformanceAndTimingReportTool()

    spans = []
    for i in range(5):
        name = f"span_{i}_" + "x" * (40 + i * 10)
        otel_span = create_mock_otel_span(
            trace_id=12345,
            span_id=100 + i,
            name=name,
            start_time=1000000000000 + i * 1000000000,
            end_time=1000000000000 + (i + 1) * 1000000000,
            parent_id=None,
        )
        spans.append(Span(otel_span))

    trace_data = TraceData(spans=spans)
    trace_info = TraceInfo(
        trace_id="test-format",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=5000,
    )
    trace = Trace(info=trace_info, data=trace_data)

    result = tool.invoke(trace)

    lines = result.split("\n")
    table_lines = []
    in_table = False

    for line in lines:
        if "SPAN TABLE:" in line:
            in_table = True
        elif in_table and line.startswith("-" * 200):
            in_table = False
        elif in_table and line.strip() and not line.startswith("-"):
            table_lines.append(line)

    for line in table_lines[1:]:
        parts = line.split()
        if len(parts) >= 3:
            assert "..." in line or all(len(p) <= 30 for p in parts[2:3])


def test_max_name_length_constant_usage():
    tool = GetSpanPerformanceAndTimingReportTool()

    assert hasattr(tool, "MAX_NAME_LENGTH")
    assert tool.MAX_NAME_LENGTH == 30

    name = "a" * (tool.MAX_NAME_LENGTH + 1)
    truncated = tool._truncate_name(name)
    assert len(truncated) == tool.MAX_NAME_LENGTH
    assert truncated == "a" * (tool.MAX_NAME_LENGTH - 3) + "..."


def test_truncation_in_concurrent_spans_section():
    tool = GetSpanPerformanceAndTimingReportTool()

    root_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=100,
        name="root",
        start_time=1000000000000000000,
        end_time=1000010000000000000,
        parent_id=None,
    )
    root_span = Span(root_otel_span)

    long_name1 = "very_long_concurrent_span_name_number_one_that_exceeds_limit"
    long_name2 = "another_extremely_long_concurrent_span_name_number_two_exceeds"

    span1_otel = create_mock_otel_span(
        trace_id=12345,
        span_id=101,
        name=long_name1,
        start_time=1000001000000000000,
        end_time=1000005000000000000,
        parent_id=100,
    )
    span1 = Span(span1_otel)

    span2_otel = create_mock_otel_span(
        trace_id=12345,
        span_id=102,
        name=long_name2,
        start_time=1000002000000000000,
        end_time=1000006000000000000,
        parent_id=100,
    )
    span2 = Span(span2_otel)

    trace_data = TraceData(spans=[root_span, span1, span2])
    trace_info = TraceInfo(
        trace_id="test-concurrent",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=10000,
    )
    trace = Trace(info=trace_info, data=trace_data)

    result = tool.invoke(trace)

    if "CONCURRENT OPERATIONS:" in result and "No significant concurrent" not in result:
        truncated1 = long_name1[:27] + "..."
        truncated2 = long_name2[:27] + "..."
        assert truncated1 in result or truncated2 in result
        assert long_name1 not in result
        assert long_name2 not in result


def test_min_overlap_threshold_enforcement():
    tool = GetSpanPerformanceAndTimingReportTool()

    root_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=100,
        name="root",
        start_time=1000000000000000000,
        end_time=1000010000000000000,
        parent_id=None,
    )
    root_span = Span(root_otel_span)

    span1_otel = create_mock_otel_span(
        trace_id=12345,
        span_id=101,
        name="span1",
        start_time=1000001000000000000,
        end_time=1000002000000000000,
        parent_id=100,
    )
    span1 = Span(span1_otel)

    span2_otel = create_mock_otel_span(
        trace_id=12345,
        span_id=102,
        name="span2",
        start_time=1000001999991000000,
        end_time=1000003000000000000,
        parent_id=100,
    )
    span2 = Span(span2_otel)

    span3_otel = create_mock_otel_span(
        trace_id=12345,
        span_id=103,
        name="span3",
        start_time=1000004000000000000,
        end_time=1000005000000000000,
        parent_id=100,
    )
    span3 = Span(span3_otel)

    span4_otel = create_mock_otel_span(
        trace_id=12345,
        span_id=104,
        name="span4",
        start_time=1000004980000000000,
        end_time=1000006000000000000,
        parent_id=100,
    )
    span4 = Span(span4_otel)

    trace_data = TraceData(spans=[root_span, span1, span2, span3, span4])
    trace_info = TraceInfo(
        trace_id="test-overlap-threshold",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=10000,
    )
    trace = Trace(info=trace_info, data=trace_data)

    result = tool.invoke(trace)

    assert tool.MIN_OVERLAP_THRESHOLD_S == 0.01
    if "CONCURRENT OPERATIONS:" in result:
        lines = result.split("\n")
        concurrent_section = False
        found_span3_span4 = False
        found_span1_span2 = False

        for line in lines:
            if "CONCURRENT OPERATIONS:" in line:
                concurrent_section = True
                continue
            if concurrent_section:
                if line.startswith("-") or "span1" in line and "span2" in line and "name1" in line:
                    continue
                if "span3" in line and "span4" in line and "." in line:
                    found_span3_span4 = True
                if "span1" in line and "span2" in line and "." in line:
                    found_span1_span2 = True

        assert found_span3_span4
        assert not found_span1_span2


def test_top_spans_count_limit():
    tool = GetSpanPerformanceAndTimingReportTool()

    spans = []
    for i in range(15):
        otel_span = create_mock_otel_span(
            trace_id=12345,
            span_id=100 + i,
            name=f"span_{i:02d}",
            start_time=1000000000000000000 + i * 100000000000,
            end_time=1000000000000000000 + (i + 1) * 100000000000,
            parent_id=None,
        )
        spans.append(Span(otel_span))

    trace_data = TraceData(spans=spans)
    trace_info = TraceInfo(
        trace_id="test-top-spans-limit",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=15000,
    )
    trace = Trace(info=trace_info, data=trace_data)

    result = tool.invoke(trace)

    assert tool.TOP_SPANS_COUNT == 10
    lines = result.split("\n")
    top_section = False
    span_count = 0

    for line in lines:
        if "TOP 10 SPANS BY SELF DURATION" in line:
            top_section = True
            continue
        if top_section and line.startswith("-"):
            continue
        if top_section and line.strip() and not line.startswith("-"):
            parts = line.strip().split()
            if parts and parts[0].isdigit():
                span_count += 1
        if top_section and line.strip() == "":
            break

    assert span_count == 10


def test_max_concurrent_pairs_exact_limit():
    tool = GetSpanPerformanceAndTimingReportTool()

    root_otel_span = create_mock_otel_span(
        trace_id=12345,
        span_id=100,
        name="root",
        start_time=1000000000000000000,
        end_time=1000100000000000000,
        parent_id=None,
    )
    root_span = Span(root_otel_span)

    spans = [root_span]

    for i in range(25):
        otel_span = create_mock_otel_span(
            trace_id=12345,
            span_id=101 + i,
            name=f"concurrent_{i:02d}",
            start_time=1000010000000000000 + i * 1000000000,
            end_time=1000050000000000000 + i * 1000000000,
            parent_id=100,
        )
        spans.append(Span(otel_span))

    trace_data = TraceData(spans=spans)
    trace_info = TraceInfo(
        trace_id="test-max-concurrent",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=100000,
    )
    trace = Trace(info=trace_info, data=trace_data)

    result = tool.invoke(trace)

    assert tool.MAX_CONCURRENT_PAIRS == 20
    lines = result.split("\n")
    concurrent_section = False
    pair_count = 0

    for line in lines:
        if "CONCURRENT OPERATIONS:" in line:
            concurrent_section = True
            continue
        if concurrent_section and "No significant concurrent" in line:
            break
        if concurrent_section and line.strip() and not line.startswith("-"):
            if "concurrent_" in line and "s" in line:
                pair_count += 1
        if concurrent_section and line.strip() == "":
            break

    assert pair_count <= tool.MAX_CONCURRENT_PAIRS


def test_all_constants_exist_and_have_expected_values():
    tool = GetSpanPerformanceAndTimingReportTool()

    assert hasattr(tool, "MAX_NAME_LENGTH")
    assert hasattr(tool, "MIN_OVERLAP_THRESHOLD_S")
    assert hasattr(tool, "TOP_SPANS_COUNT")
    assert hasattr(tool, "MAX_CONCURRENT_PAIRS")

    assert tool.MAX_NAME_LENGTH == 30
    assert tool.MIN_OVERLAP_THRESHOLD_S == 0.01
    assert tool.TOP_SPANS_COUNT == 10
    assert tool.MAX_CONCURRENT_PAIRS == 20

    assert isinstance(tool.MAX_NAME_LENGTH, int)
    assert isinstance(tool.MIN_OVERLAP_THRESHOLD_S, float)
    assert isinstance(tool.TOP_SPANS_COUNT, int)
    assert isinstance(tool.MAX_CONCURRENT_PAIRS, int)

    assert 10 <= tool.MAX_NAME_LENGTH <= 100
    assert 0.001 <= tool.MIN_OVERLAP_THRESHOLD_S <= 1.0
    assert 5 <= tool.TOP_SPANS_COUNT <= 50
    assert 10 <= tool.MAX_CONCURRENT_PAIRS <= 100
