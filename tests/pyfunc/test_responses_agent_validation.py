import pytest
from pydantic import ValidationError

from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    responses_to_cc,
    to_chat_completions_input,
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


@pytest.mark.parametrize(
    "output",
    [
        "Hello, world!",
        [{"type": "input_text", "text": "Result"}],
    ],
)
def test_function_call_output_accepts_string_and_list(output):
    item = FunctionCallOutput(call_id="call_123", output=output)
    assert item.output == output


def test_function_call_output_stream_event_with_list_output():
    event = ResponsesAgentStreamEvent(
        type="response.output_item.done",
        item={"type": "function_call_output", "call_id": "c", "output": [{"type": "input_text"}]},
    )
    assert event.type == "response.output_item.done"


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("hello", "hello"),
        ([{"key": "value"}], '[{"key": "value"}]'),
        ({"a": 1}, '{"a": 1}'),
        (12345, "12345"),
    ],
)
def test_responses_to_cc_stringifies_function_call_output(output, expected):
    result = responses_to_cc({"type": "function_call_output", "call_id": "c", "output": output})
    assert result[0]["content"] == expected


def test_responses_to_cc_fallback_to_str_on_non_serializable():
    class NonSerializable:
        pass

    result = responses_to_cc(
        {"type": "function_call_output", "call_id": "c", "output": [NonSerializable()]}
    )
    assert isinstance(result[0]["content"], str)


def test_function_call_output_with_openai_agents_sdk_format():
    """Test real-world output format from OpenAI Agents SDK tool calls."""
    raw_item = {
        "call_id": "toolu_bdrk_017fvUyTS6oaCDYg6GVL3X7j",
        "output": [
            {
                "type": "input_text",
                "text": '{"content":{"queryAttachments":[]},"status":"COMPLETED"}',
            }
        ],
        "type": "function_call_output",
    }
    # Validation should pass
    item = FunctionCallOutput(**raw_item)
    assert isinstance(item.output, list)

    # Stream event should pass
    event = ResponsesAgentStreamEvent(type="response.output_item.done", item=raw_item)
    assert event.type == "response.output_item.done"

    # Conversion to ChatCompletion should produce valid string content
    result = responses_to_cc(raw_item)
    assert result[0]["role"] == "tool"
    assert isinstance(result[0]["content"], str)
    assert "input_text" in result[0]["content"]


def test_function_call_output_round_trip():
    """Test round-trip: Responses API items -> ChatCompletion -> back works with list output."""
    responses_input = [
        {"role": "user", "content": "What's the weather?", "type": "message"},
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_123",
            "name": "get_weather",
            "arguments": '{"city": "Seattle"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": [{"type": "input_text", "text": "Sunny, 72°F"}],
        },
    ]
    # Convert to ChatCompletion format
    cc_messages = to_chat_completions_input(responses_input)

    assert len(cc_messages) == 3
    assert cc_messages[0]["role"] == "user"
    assert cc_messages[1]["role"] == "assistant"
    assert cc_messages[2]["role"] == "tool"
    assert isinstance(cc_messages[2]["content"], str)
    assert cc_messages[2]["tool_call_id"] == "call_123"
