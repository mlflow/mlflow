from unittest import mock

from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.genai.judges.tools.get_trace_info import GetTraceInfoTool
from mlflow.types.llm import ToolDefinition


def test_get_trace_info_tool_name():
    tool = GetTraceInfoTool()
    assert tool.name == "get_trace_info"


def test_get_trace_info_tool_get_definition():
    tool = GetTraceInfoTool()
    definition = tool.get_definition()

    assert isinstance(definition, ToolDefinition)
    assert definition.function.name == "get_trace_info"
    assert "metadata about the trace" in definition.function.description
    assert definition.function.parameters.type == "object"
    assert "trace_id" in definition.function.parameters.properties
    assert definition.function.parameters.required == ["trace_id"]
    assert definition.type == "function"


def test_get_trace_info_tool_invoke_success():
    tool = GetTraceInfoTool()

    trace_info = TraceInfo(
        trace_id="test-trace-123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=250,
    )
    trace = Trace(info=trace_info, data=None)

    with mock.patch("mlflow.get_trace", return_value=trace):
        result = tool.invoke(trace_id=trace_info.trace_id)

    assert isinstance(result, dict)
    assert result["trace_id"] == "test-trace-123"
    assert result["execution_duration_ms"] == 250
    assert "assessments" not in result


def test_get_trace_info_tool_invoke_returns_dict_without_assessments():
    tool = GetTraceInfoTool()

    trace_info = TraceInfo(
        trace_id="test-trace-simple",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1000000000,
        state=TraceState.OK,
        execution_duration=100,
    )
    trace = Trace(info=trace_info, data=None)

    with mock.patch("mlflow.get_trace", return_value=trace):
        result = tool.invoke(trace_id="test-trace-simple")

    assert isinstance(result, dict)
    assert result["trace_id"] == "test-trace-simple"
    assert "assessments" not in result


def test_get_trace_info_tool_invoke_different_states():
    tool = GetTraceInfoTool()

    trace_info = TraceInfo(
        trace_id="test-trace-456",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=9876543210,
        state=TraceState.ERROR,
        execution_duration=500,
    )
    trace = Trace(info=trace_info, data=None)

    with mock.patch("mlflow.get_trace", return_value=trace):
        result = tool.invoke(trace_id=trace_info.trace_id)

    assert isinstance(result, dict)
    assert result["trace_id"] == "test-trace-456"
    assert result["state"] == "ERROR"
    assert "assessments" not in result
