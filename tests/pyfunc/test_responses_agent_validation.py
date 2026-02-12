import pytest
from pydantic import ValidationError

from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    responses_to_cc,
)
from mlflow.types.responses_helpers import FunctionCallOutput, Message


def test_responses_request_validation():
    with pytest.raises(ValueError, match="content.0.text"):
        ResponsesAgentRequest(
            **{
                "input": [
                    {
                        "type": "message",
                        "id": "1",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                            }
                        ],
                    }
                ],
            }
        )

    with pytest.raises(ValueError, match="role"):
        ResponsesAgentRequest(
            **{
                "input": [
                    {
                        "type": "message",
                        "id": "1",
                        "status": "completed",
                        "role": "asdf",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "asdf",
                            }
                        ],
                    }
                ],
            }
        )


def test_message_content_validation():
    # Test that None content is rejected (by Pydantic validation)
    with pytest.raises(ValidationError, match="Input should be a valid"):
        Message(role="assistant", content=None, type="message")

    # Test that empty string content is allowed
    message_empty_str = Message(role="assistant", content="", type="message")
    assert message_empty_str.content == ""

    # Test that empty list content is allowed
    message_empty_list = Message(role="assistant", content=[], type="message")
    assert message_empty_list.content == []


def test_responses_response_validation():
    with pytest.raises(ValueError, match="output.0.content.0.text"):
        ResponsesAgentResponse(
            **{
                "output": [
                    {
                        "type": "message",
                        "id": "1",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                            }
                        ],
                    }
                ],
            }
        )


def test_responses_stream_event_validation():
    with pytest.raises(ValueError, match="content must not be an empty"):
        ResponsesAgentStreamEvent(
            **{
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                    "id": "1",
                },
            }
        )

    with pytest.raises(ValueError, match="Invalid status"):
        ResponsesAgentStreamEvent(
            **{
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "status": "asdf",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "asdf",
                        }
                    ],
                    "id": "1",
                },
            }
        )

    with pytest.raises(ValueError, match="item.content.0.annotations.0.url"):
        ResponsesAgentStreamEvent(
            **{
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {
                    "type": "message",
                    "id": "msg_67ed73ed2c288191b0f0f445e21c66540fbd8030171e9b0c",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "On T",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "start_index": 359,
                                    "end_index": 492,
                                    "title": "NBA roundup:",
                                },
                            ],
                        }
                    ],
                },
            },
        )
    with pytest.raises(ValueError, match="delta"):
        ResponsesAgentStreamEvent(
            **{
                "type": "response.output_text.delta",
                "item_id": "msg_67eda402cba48191a1c35b84af04fc3c0a4363ad71e9395a",
                "output_index": 0,
                "content_index": 0,
            },
        )

    with pytest.raises(ValueError, match="annotation.url"):
        ResponsesAgentStreamEvent(
            **{
                "type": "response.output_text.annotation.added",
                "item_id": "msg_67ed73ed2c288191b0f0f445e21c66540fbd8030171e9b0c",
                "output_index": 1,
                "content_index": 0,
                "annotation_index": 0,
                "annotation": {
                    "type": "url_citation",
                    "start_index": 359,
                    "end_index": 492,
                    "title": "NBA roundup: Wolves overcome Nikola",
                },
            },
        )


def test_function_call_output_accepts_string_and_list():
    output_with_string = FunctionCallOutput(
        call_id="call_123",
        output="Hello, world!",
    )
    assert output_with_string.output == "Hello, world!"

    output_with_list = FunctionCallOutput(
        call_id="call_456",
        output=[
            {"type": "input_text", "text": "Result from tool"},
            {"type": "input_image", "image_url": "https://example.com/image.png"},
        ],
    )
    assert isinstance(output_with_list.output, list)
    assert len(output_with_list.output) == 2
    assert output_with_list.output[0]["type"] == "input_text"


def test_function_call_output_stream_event_with_list_output():
    event = ResponsesAgentStreamEvent(
        **{
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call_output",
                "call_id": "call_789",
                "output": [
                    {"type": "input_text", "text": "Tool execution result"},
                ],
            },
        }
    )
    assert event.type == "response.output_item.done"


def test_responses_to_cc_converts_list_output_to_json_string():
    message_with_string_output = {
        "type": "function_call_output",
        "call_id": "call_123",
        "output": "Hello, world!",
    }
    result = responses_to_cc(message_with_string_output)
    assert len(result) == 1
    assert result[0]["role"] == "tool"
    assert result[0]["content"] == "Hello, world!"
    assert result[0]["tool_call_id"] == "call_123"

    message_with_list_output = {
        "type": "function_call_output",
        "call_id": "call_456",
        "output": [
            {"type": "input_text", "text": "Result from tool"},
            {"type": "input_image", "image_url": "https://example.com/image.png"},
        ],
    }
    result = responses_to_cc(message_with_list_output)
    assert len(result) == 1
    assert result[0]["role"] == "tool"
    assert result[0]["tool_call_id"] == "call_456"
    # List output should be converted to JSON string
    assert isinstance(result[0]["content"], str)
    assert "input_text" in result[0]["content"]
    assert "Result from tool" in result[0]["content"]


def test_responses_to_cc_handles_non_serializable_list_output():
    class NonSerializable:
        def __str__(self):
            return "non-serializable-object"

    message_with_non_serializable = {
        "type": "function_call_output",
        "call_id": "call_789",
        "output": [NonSerializable(), "some text"],
    }
    result = responses_to_cc(message_with_non_serializable)
    assert len(result) == 1
    assert result[0]["role"] == "tool"
    assert result[0]["tool_call_id"] == "call_789"
    # Should fall back to str() when json.dumps fails
    assert isinstance(result[0]["content"], str)
