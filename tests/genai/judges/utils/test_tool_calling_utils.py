import json
from dataclasses import dataclass
from unittest import mock

import pytest

from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.genai.judges.tools.get_span_image import SpanImageResult
from mlflow.genai.judges.utils.tool_calling_utils import (
    _get_image_turn_tool_call_id,
    _process_tool_calls,
    _remove_oldest_tool_call_pair,
)
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


def test_process_tool_calls_image_result_yields_tool_ack_and_user_image(mock_trace):
    tool_call = _make_tool_call("call_img", "get_span_image")
    data_url = "data:image/png;base64,QUJD"

    with mock.patch(
        "mlflow.genai.judges.tools.registry._judge_tool_registry.invoke"
    ) as mock_invoke:
        mock_invoke.return_value = SpanImageResult(
            span_id="span-1", content_type="image/png", data_url=data_url
        )

        result = _process_tool_calls(tool_calls=[tool_call], trace=mock_trace)

    # An image result expands into a tool ack + a multimodal user turn.
    assert len(result) == 2

    ack, image_turn = result
    assert ack.role == "tool"
    assert ack.tool_call_id == "call_img"
    assert isinstance(ack.content, str)
    assert "span-1" in ack.content

    assert image_turn.role == "user"
    assert image_turn.tool_call_id is None
    assert isinstance(image_turn.content, list)
    text_part, image_part = image_turn.content
    assert text_part == {"type": "text", "text": "Fetched image for span span-1:"}
    assert image_part == {"type": "image_url", "image_url": {"url": data_url}}
    # The injected user turn is tagged with its originating tool_call_id for pruning.
    assert _get_image_turn_tool_call_id(image_turn) == "call_img"


def test_process_tool_calls_normal_result_still_single_tool_message(mock_trace):
    tool_call = _make_tool_call("call_norm", "get_span")

    with mock.patch(
        "mlflow.genai.judges.tools.registry._judge_tool_registry.invoke"
    ) as mock_invoke:
        mock_invoke.return_value = {"result": "plain"}

        result = _process_tool_calls(tool_calls=[tool_call], trace=mock_trace)

    assert len(result) == 1
    assert result[0].role == "tool"
    assert json.loads(result[0].content) == {"result": "plain"}


def test_process_tool_calls_batched_calls_emit_all_tool_responses_before_image_turn(mock_trace):
    # OpenAI requires all role="tool" responses for an assistant turn to be consecutive,
    # immediately after the assistant message, with no other role interleaved. When one
    # assistant turn batches an image tool call and a normal one, every tool response
    # (including the image ack) must precede the injected user image turn.
    image_call = _make_tool_call("call_img", "get_span_image")
    normal_call = _make_tool_call("call_norm", "get_span")

    with mock.patch(
        "mlflow.genai.judges.tools.registry._judge_tool_registry.invoke"
    ) as mock_invoke:
        mock_invoke.side_effect = [
            SpanImageResult(
                span_id="span-1", content_type="image/png", data_url="data:image/png;base64,QUJD"
            ),
            {"result": "normal data"},
        ]

        result = _process_tool_calls(tool_calls=[image_call, normal_call], trace=mock_trace)

    # ack(image) + tool(normal) + user(image) — the user turn comes strictly last.
    assert [msg.role for msg in result] == ["tool", "tool", "user"]

    tool_msgs = [msg for msg in result if msg.role == "tool"]
    # Each tool_call_id has exactly one tool response.
    assert sorted(msg.tool_call_id for msg in tool_msgs) == ["call_img", "call_norm"]

    image_turn = result[-1]
    assert isinstance(image_turn.content, list)
    assert _get_image_turn_tool_call_id(image_turn) == "call_img"


def test_remove_oldest_tool_call_pair_drops_injected_image_turn(mock_trace):
    tool_call = _make_tool_call("call_img", "get_span_image")

    with mock.patch(
        "mlflow.genai.judges.tools.registry._judge_tool_registry.invoke"
    ) as mock_invoke:
        mock_invoke.return_value = SpanImageResult(
            span_id="span-1", content_type="image/png", data_url="data:image/png;base64,QUJD"
        )
        tool_responses = _process_tool_calls(tool_calls=[tool_call], trace=mock_trace)

    assistant_msg = ChatMessage(role="assistant", content=None, tool_calls=[tool_call])
    messages = [
        ChatMessage(role="user", content="analyze this trace"),
        assistant_msg,
        *tool_responses,
        ChatMessage(role="assistant", content="final answer"),
    ]

    pruned = _remove_oldest_tool_call_pair(messages)

    assert pruned is not None
    # The assistant tool-call message, its tool ack, AND the injected image user-turn
    # are all removed together — no orphaned multimodal turn remains.
    assert assistant_msg not in pruned
    assert all(msg.role != "tool" for msg in pruned)
    assert all(not isinstance(msg.content, list) for msg in pruned)
    assert all(_get_image_turn_tool_call_id(msg) is None for msg in pruned)
    # The unrelated user and final assistant messages survive.
    assert [msg.role for msg in pruned] == ["user", "assistant"]


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
