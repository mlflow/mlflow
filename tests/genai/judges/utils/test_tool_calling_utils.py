import json
from dataclasses import dataclass
from unittest import mock

import litellm
import pytest

from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.genai.judges.utils.tool_calling_utils import _process_tool_calls


@pytest.fixture
def mock_trace():
    trace_info = TraceInfo(
        trace_id="test-trace",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
    )
    return Trace(info=trace_info, data=None)


def test_process_tool_calls_success(mock_trace):
    mock_tool_call = mock.Mock()
    mock_tool_call.id = "call_123"
    mock_tool_call.function.name = "test_tool"
    mock_tool_call.function.arguments = '{"arg": "value"}'

    with mock.patch(
        "mlflow.genai.judges.tools.registry._judge_tool_registry.invoke"
    ) as mock_invoke:
        mock_invoke.return_value = {"result": "success"}

        result = _process_tool_calls(tool_calls=[mock_tool_call], trace=mock_trace)

    assert len(result) == 1
    assert isinstance(result[0], litellm.Message)
    assert result[0].role == "tool"
    assert result[0].tool_call_id == "call_123"
    assert result[0].name == "test_tool"
    assert json.loads(result[0].content) == {"result": "success"}


def test_process_tool_calls_with_error(mock_trace):
    mock_tool_call = mock.Mock()
    mock_tool_call.id = "call_456"
    mock_tool_call.function.name = "failing_tool"
    mock_tool_call.function.arguments = "{}"

    with mock.patch(
        "mlflow.genai.judges.tools.registry._judge_tool_registry.invoke"
    ) as mock_invoke:
        mock_invoke.side_effect = RuntimeError("Tool execution failed")

        result = _process_tool_calls(tool_calls=[mock_tool_call], trace=mock_trace)

    assert len(result) == 1
    assert result[0].role == "tool"
    assert result[0].tool_call_id == "call_456"
    assert "Error: Tool execution failed" in result[0].content


def test_process_tool_calls_multiple(mock_trace):
    mock_tool_call_1 = mock.Mock()
    mock_tool_call_1.id = "call_1"
    mock_tool_call_1.function.name = "tool_1"
    mock_tool_call_1.function.arguments = "{}"

    mock_tool_call_2 = mock.Mock()
    mock_tool_call_2.id = "call_2"
    mock_tool_call_2.function.name = "tool_2"
    mock_tool_call_2.function.arguments = "{}"

    with mock.patch(
        "mlflow.genai.judges.tools.registry._judge_tool_registry.invoke"
    ) as mock_invoke:
        mock_invoke.side_effect = [{"result": "first"}, {"result": "second"}]

        result = _process_tool_calls(
            tool_calls=[mock_tool_call_1, mock_tool_call_2], trace=mock_trace
        )

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

    mock_tool_call = mock.Mock()
    mock_tool_call.id = "call_789"
    mock_tool_call.function.name = "dataclass_tool"
    mock_tool_call.function.arguments = "{}"

    with mock.patch(
        "mlflow.genai.judges.tools.registry._judge_tool_registry.invoke"
    ) as mock_invoke:
        mock_invoke.return_value = ToolResult(status="ok", count=42)

        result = _process_tool_calls(tool_calls=[mock_tool_call], trace=mock_trace)

    assert len(result) == 1
    assert result[0].role == "tool"
    content = json.loads(result[0].content)
    assert content == {"status": "ok", "count": 42}


def test_process_tool_calls_with_string_result(mock_trace):
    mock_tool_call = mock.Mock()
    mock_tool_call.id = "call_str"
    mock_tool_call.function.name = "string_tool"
    mock_tool_call.function.arguments = "{}"

    with mock.patch(
        "mlflow.genai.judges.tools.registry._judge_tool_registry.invoke"
    ) as mock_invoke:
        mock_invoke.return_value = "Plain string result"

        result = _process_tool_calls(tool_calls=[mock_tool_call], trace=mock_trace)

    assert len(result) == 1
    assert result[0].role == "tool"
    assert result[0].content == "Plain string result"


def test_process_tool_calls_mixed_success_and_error(mock_trace):
    mock_tool_call_1 = mock.Mock()
    mock_tool_call_1.id = "call_success"
    mock_tool_call_1.function.name = "success_tool"
    mock_tool_call_1.function.arguments = "{}"

    mock_tool_call_2 = mock.Mock()
    mock_tool_call_2.id = "call_error"
    mock_tool_call_2.function.name = "error_tool"
    mock_tool_call_2.function.arguments = "{}"

    with mock.patch(
        "mlflow.genai.judges.tools.registry._judge_tool_registry.invoke"
    ) as mock_invoke:
        mock_invoke.side_effect = [{"result": "success"}, RuntimeError("Failed")]

        result = _process_tool_calls(
            tool_calls=[mock_tool_call_1, mock_tool_call_2], trace=mock_trace
        )

    assert len(result) == 2
    assert result[0].tool_call_id == "call_success"
    assert json.loads(result[0].content) == {"result": "success"}
    assert result[1].tool_call_id == "call_error"
    assert "Error: Failed" in result[1].content
