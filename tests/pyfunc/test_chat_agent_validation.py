import pytest

from mlflow.types.agent import ChatAgentChunk, ChatAgentMessage


def test_chat_agent_message_throws_on_invalid_data():
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

    # Missing both 'name' and 'tool_call_id' for tool message
    data = {"role": "tool", "content": "test_content"}
    with pytest.raises(ValueError, match="Both 'name' and 'tool_call_id'."):
        ChatAgentMessage(**data)


def test_chat_agent_chunk_throws_on_missing_id():
    data = {"delta": {"role": "user", "content": "a"}}
    with pytest.raises(ValueError, match="The field `delta` of ChatAgentChunk"):
        ChatAgentChunk(**data)


def test_chat_agent_chunk_throws_on_updated_id():
    data = {"delta": {"role": "user", "content": "a", "id": "1"}}
    chunk = ChatAgentChunk(**data)
    with pytest.raises(ValueError, match="The field `delta` of ChatAgentChunk"):
        chunk.delta = ChatAgentMessage(**{"role": "user", "content": "b"})
