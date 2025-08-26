import json

import pytest

from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.tools import (
    GetTraceInfoTool,
    JudgeToolRegistry,
    invoke_judge_tool,
    list_judge_tools,
    register_judge_tool,
)
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.types.llm import FunctionToolCallArguments, ToolCall, ToolDefinition


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


def test_registry_invoke_tool_success():
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


def test_global_functions_work():
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


def test_builtin_registration():
    tools = list_judge_tools()
    tool_names = [t.name for t in tools]

    assert "get_trace_info" in tool_names

    get_trace_info_tools = [t for t in tools if t.name == "get_trace_info"]
    assert len(get_trace_info_tools) == 1
    assert isinstance(get_trace_info_tools[0], GetTraceInfoTool)
