import json
from dataclasses import dataclass
from unittest import mock

import pytest

from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.genai.judges.utils.tool_calling_utils import _process_tool_calls
from mlflow.types.llm import ChatMessage, ToolCall


@pytest.fixture
def mock_trace():
    trace_info = TraceInfo(
        trace_id="test-trace",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
    )
    return Trace(info=trace_info, data=None)


def _make_tool_call(call_id, name, arguments="{}"):
    return ToolCall(
        id=call_id,
        function={"name": name, "arguments": arguments},
    )


def test_process_tool_calls_success(mock_trace):
    tool_call = _make_tool_call("call_123", "test_tool", '{"arg": "value"}')

    with mock.patch(
        "mlflow.genai.judges.tools.registry._judge_tool_registry.invoke"
    ) as mock_invoke:
        mock_invoke.return_value = {"result": "success"}

        result = _process_tool_calls(tool_calls=[tool_call], trace=mock_trace)

    assert len(result) == 1
    assert isinstance(result[0], ChatMessage)
    assert result[0].role == "tool"
    assert result[0].tool_call_id == "call_123"
    assert result[0].name == "test_tool"
    assert json.loads(result[0].content) == {"result": "success"}


def test_process_tool_calls_with_error(mock_trace):
    tool_call = _make_tool_call("call_456", "failing_tool")

    with mock.patch(
        "mlflow.genai.judges.tools.registry._judge_tool_registry.invoke"
    ) as mock_invoke:
        mock_invoke.side_effect = RuntimeError("Tool execution failed")

        result = _process_tool_calls(tool_calls=[tool_call], trace=mock_trace)

    assert len(result) == 1
    assert result[0].role == "tool"
    assert result[0].tool_call_id == "call_456"
    assert "Error: Tool execution failed" in result[0].content


def test_process_tool_calls_multiple(mock_trace):
    tool_call_1 = _make_tool_call("call_1", "tool_1")
    tool_call_2 = _make_tool_call("call_2", "tool_2")

    with mock.patch(
        "mlflow.genai.judges.tools.registry._judge_tool_registry.invoke"
    ) as mock_invoke:
        mock_invoke.side_effect = [{"result": "first"}, {"result": "second"}]

        result = _process_tool_calls(tool_calls=[tool_call_1, tool_call_2], trace=mock_trace)

    assert len(result) == 2
    assert result[0].tool_call_id == "call_1"
    assert result[1].tool_call_id == "call_2"
    assert json.loads(result[0].content) == {"result": "first"}
    assert json.loads(result[1].content) == {"result": "second"}


def test_process_tool_calls_with_dataclass(mock_trace):
    @dataclass
    class ToolResult:
        status: str
        count: int

    tool_call = _make_tool_call("call_789", "dataclass_tool")

    with mock.patch(
        "mlflow.genai.judges.tools.registry._judge_tool_registry.invoke"
    ) as mock_invoke:
        mock_invoke.return_value = ToolResult(status="ok", count=42)

        result = _process_tool_calls(tool_calls=[tool_call], trace=mock_trace)

    assert len(result) == 1
    assert result[0].role == "tool"
    content = json.loads(result[0].content)
    assert content == {"status": "ok", "count": 42}


def test_process_tool_calls_with_string_result(mock_trace):
    tool_call = _make_tool_call("call_str", "string_tool")

    with mock.patch(
        "mlflow.genai.judges.tools.registry._judge_tool_registry.invoke"
    ) as mock_invoke:
        mock_invoke.return_value = "Plain string result"

        result = _process_tool_calls(tool_calls=[tool_call], trace=mock_trace)

    assert len(result) == 1
    assert result[0].role == "tool"
    assert result[0].content == "Plain string result"


def test_process_tool_calls_mixed_success_and_error(mock_trace):
    tool_call_1 = _make_tool_call("call_success", "success_tool")
    tool_call_2 = _make_tool_call("call_error", "error_tool")

    with mock.patch(
        "mlflow.genai.judges.tools.registry._judge_tool_registry.invoke"
    ) as mock_invoke:
        mock_invoke.side_effect = [{"result": "success"}, RuntimeError("Failed")]

        result = _process_tool_calls(tool_calls=[tool_call_1, tool_call_2], trace=mock_trace)

    assert len(result) == 2
    assert result[0].tool_call_id == "call_success"
    assert json.loads(result[0].content) == {"result": "success"}
    assert result[1].tool_call_id == "call_error"
    assert "Error: Failed" in result[1].content
