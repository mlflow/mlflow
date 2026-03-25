from mlflow.genai.judges.types import JudgeMessage


def test_judge_message_basic():
    msg = JudgeMessage(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"
    assert msg.tool_calls is None
    assert msg.tool_call_id is None
    assert msg.name is None


def test_judge_message_tool_response():
    msg = JudgeMessage(
        role="tool",
        content='{"result": "ok"}',
        tool_call_id="call_123",
        name="get_root_span",
    )
    assert msg.role == "tool"
    assert msg.tool_call_id == "call_123"
    assert msg.name == "get_root_span"


def test_judge_message_with_tool_calls():
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "list_spans", "arguments": "{}"},
        }
    ]
    msg = JudgeMessage(role="assistant", tool_calls=tool_calls)
    assert msg.content is None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0]["function"]["name"] == "list_spans"
