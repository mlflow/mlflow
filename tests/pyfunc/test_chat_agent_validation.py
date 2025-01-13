import pytest

from mlflow.types.agent import ChatAgentMessage


def test_chat_message_throws_on_invalid_data():
    # Missing 'content' or 'tool_calls'
    data = {"role": "user", "name": "test_user"}
    with pytest.raises(ValueError, match="Either 'content' or 'tool_calls'"):
        ChatAgentMessage(**data)
