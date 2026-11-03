import pytest

from mlflow.anthropic.chat import convert_message_to_mlflow_chat
from mlflow.exceptions import MlflowException


@pytest.mark.parametrize(
    ("result_content", "expected_texts"),
    [
        ("72F", ["72F"]),
        ([{"type": "text", "text": "72F"}, {"type": "text", "text": "sunny"}], ["72F", "sunny"]),
    ],
)
def test_convert_tool_result(result_content, expected_texts):
    message = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": result_content}],
    }
    converted = convert_message_to_mlflow_chat(message)
    assert converted.role == "tool"
    assert converted.tool_call_id == "toolu_1"
    assert [part.text for part in converted.content] == expected_texts


def test_convert_redacted_thinking_is_skipped():
    message = {
        "role": "assistant",
        "content": [
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "text", "text": "hi"},
        ],
    }
    converted = convert_message_to_mlflow_chat(message)
    assert [part.text for part in converted.content] == ["hi"]


def test_convert_unknown_content_type_raises():
    message = {
        "role": "user",
        "content": [{"type": "not_a_real_block"}],
    }
    with pytest.raises(MlflowException, match="Unknown content type: not_a_real_block"):
        convert_message_to_mlflow_chat(message)
