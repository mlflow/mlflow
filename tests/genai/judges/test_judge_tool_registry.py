import inspect
import json

import pytest

import mlflow
from mlflow.entities.span import SpanType
from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.tools import (
    JudgeToolRegistry,
    invoke_judge_tool,
    list_judge_tools,
    register_judge_tool,
)
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.genai.judges.tools.constants import ToolNames
from mlflow.types.llm import FunctionToolCallArguments, ToolCall, ToolDefinition


@pytest.fixture
def restore_global_registry():
    from mlflow.genai.judges.tools.registry import _judge_tool_registry

    original_tools = _judge_tool_registry._tools.copy()
    yield
    _judge_tool_registry._tools = original_tools


class MockTool(JudgeTool):
    @property
    def name(self) -> str:
        return "mock_tool"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            function={
                "name": "mock_tool",
                "description": "A mock tool for testing",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            type="function",
        )

    def invoke(self, trace: Trace, **kwargs) -> str:
        return f"mock_result_with_{len(kwargs)}_args"


def test_registry_register_and_list_tools():
    registry = JudgeToolRegistry()
    mock_tool = MockTool()

    assert len(registry.list_tools()) == 0

    registry.register(mock_tool)

    tools = registry.list_tools()
    assert len(tools) == 1
    assert tools[0].name == "mock_tool"


@pytest.mark.parametrize("tracing_enabled", [True, False])
def test_registry_invoke_tool_success(tracing_enabled, monkeypatch):
    if tracing_enabled:
        monkeypatch.setenv("MLFLOW_GENAI_EVAL_ENABLE_SCORER_TRACING", "true")

    registry = JudgeToolRegistry()
    mock_tool = MockTool()
    registry.register(mock_tool)

    trace_info = TraceInfo(
        trace_id="test-trace-id",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=100,
    )
    trace = Trace(info=trace_info, data=None)

    tool_call = ToolCall(
        function=FunctionToolCallArguments(
            name="mock_tool", arguments=json.dumps({"param": "value"})
        )
    )

    result = registry.invoke(tool_call, trace)
    assert result == "mock_result_with_1_args"

    if tracing_enabled:
        traces = mlflow.search_traces(return_type="list")
        assert len(traces) == 1
        # Tool itself only creates one span. In real case, it will be under the parent scorer trace.
        assert len(traces[0].data.spans) == 1
        assert traces[0].data.spans[0].name == "mock_tool"
        assert traces[0].data.spans[0].span_type == SpanType.TOOL


def test_registry_invoke_tool_not_found():
    registry = JudgeToolRegistry()

    trace_info = TraceInfo(
        trace_id="test-trace-id",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=100,
    )
    trace = Trace(info=trace_info, data=None)

    tool_call = ToolCall(
        function=FunctionToolCallArguments(name="nonexistent_tool", arguments=json.dumps({}))
    )

    with pytest.raises(MlflowException, match="Tool 'nonexistent_tool' not found in registry"):
        registry.invoke(tool_call, trace)


def test_registry_invoke_tool_invalid_json():
    registry = JudgeToolRegistry()
    mock_tool = MockTool()
    registry.register(mock_tool)

    trace_info = TraceInfo(
        trace_id="test-trace-id",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=100,
    )
    trace = Trace(info=trace_info, data=None)

    tool_call = ToolCall(
        function=FunctionToolCallArguments(name="mock_tool", arguments="invalid json {{")
    )

    with pytest.raises(MlflowException, match="Invalid JSON arguments for tool 'mock_tool'"):
        registry.invoke(tool_call, trace)


def test_registry_invoke_tool_invalid_arguments():
    registry = JudgeToolRegistry()

    class StrictTool(JudgeTool):
        @property
        def name(self) -> str:
            return "strict_tool"

        def get_definition(self) -> ToolDefinition:
            return ToolDefinition(function={}, type="function")

        def invoke(self, trace: Trace, required_param: str) -> str:
            return f"result_{required_param}"

    strict_tool = StrictTool()
    registry.register(strict_tool)

    trace_info = TraceInfo(
        trace_id="test-trace-id",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=100,
    )
    trace = Trace(info=trace_info, data=None)

    tool_call = ToolCall(
        function=FunctionToolCallArguments(name="strict_tool", arguments=json.dumps({}))
    )

    with pytest.raises(MlflowException, match="Invalid arguments for tool 'strict_tool'"):
        registry.invoke(tool_call, trace)


def test_global_functions_work(restore_global_registry):
    mock_tool = MockTool()
    register_judge_tool(mock_tool)

    tools = list_judge_tools()
    tool_names = [t.name for t in tools]
    assert "mock_tool" in tool_names

    trace_info = TraceInfo(
        trace_id="test-trace-id",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=100,
    )
    trace = Trace(info=trace_info, data=None)

    tool_call = ToolCall(
        function=FunctionToolCallArguments(name="mock_tool", arguments=json.dumps({}))
    )

    result = invoke_judge_tool(tool_call, trace)
    assert result == "mock_result_with_0_args"


def test_builtin_tools_are_properly_registered():
    tools = list_judge_tools()
    registered_tool_names = {t.name for t in tools if not isinstance(t, MockTool)}

    # Only include tool constants that don't start with underscore (public tools)
    all_tool_constants = {
        value
        for name, value in inspect.getmembers(ToolNames)
        if not name.startswith("_") and isinstance(value, str)
    }

    assert all_tool_constants == registered_tool_names

    for tool in tools:
        if tool.name in all_tool_constants:
            assert isinstance(tool, JudgeTool)
