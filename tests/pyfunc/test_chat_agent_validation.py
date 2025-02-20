import pytest

from mlflow.types.agent import ChatAgentMessage


def test_chat_message_throws_on_invalid_data():
    # Missing 'content' or 'tool_calls'
    data = {"role": "user", "name": "test_user"}
    with pytest.raises(ValueError, match="Either 'content' or 'tool_calls'"):
        ChatAgentMessage(**data)

    # Missing 'name' for tool message
    data = {"role": "tool", "content": "test_content", "tool_call_id": "test_tool_call_id"}
    with pytest.raises(ValueError, match="Both 'name' and 'tool_call_id'."):
        ChatAgentMessage(**data)

    # Missing 'tool_call_id' for tool message
    data = {"role": "tool", "name": "test_user", "content": "test_content"}
    with pytest.raises(ValueError, match="Both 'name' and 'tool_call_id'."):
        ChatAgentMessage(**data)
