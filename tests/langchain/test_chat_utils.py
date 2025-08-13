from unittest.mock import MagicMock, patch

import langchain
import pytest
from langchain.agents import AgentExecutor
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_core.outputs.generation import Generation
from packaging.version import Version

from mlflow.langchain.utils.chat import (
    convert_lc_message_to_chat_message,
    parse_token_usage,
    transform_request_json_for_chat_if_necessary,
    try_transform_response_iter_to_chat_format,
    try_transform_response_to_chat_format,
)
from mlflow.types.chat import ChatMessage, Function
from mlflow.types.chat import ToolCall as _ToolCall


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            AIMessage(content="foo", id="123"),
            ChatMessage(role="assistant", content="foo", id="123"),
        ),
        (
            ToolMessage(content="foo", tool_call_id="123"),
            ChatMessage(role="tool", content="foo", tool_call_id="123"),
        ),
        (
            SystemMessage(content="foo"),
            ChatMessage(role="system", content="foo"),
        ),
        (
            HumanMessage(content="foo"),
            ChatMessage(role="user", content="foo"),
        ),
    ],
)
def test_convert_lc_message_to_chat_message(message, expected):
    assert convert_lc_message_to_chat_message(message) == expected


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.1.0"),
    reason="AIMessage does not have tool_calls attribute",
)
@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            AIMessage(
                content=[
                    {"type": "text", "text": "Response text"},
                    {"type": "tool_use", "id": "123", "name": "tool"},
                ],
                tool_calls=[{"id": "123", "name": "tool", "args": {}, "type": "tool_call"}],
            ),
            ChatMessage(
                role="assistant",
                content=[{"type": "text", "text": "Response text"}],
                tool_calls=[
                    _ToolCall(
                        id="123",
                        type="function",
                        function=Function(name="tool", arguments="{}"),
                    )
                ],
            ),
        ),
        (
            AIMessage(
                content="",
                tool_calls=[{"id": "123", "name": "tool_name", "args": {"arg1": "val1"}}],
            ),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    _ToolCall(
                        id="123",
                        type="function",
                        function=Function(name="tool_name", arguments='{"arg1": "val1"}'),
                    )
                ],
            ),
        ),
    ],
)
def test_convert_lc_message_to_chat_message_tool_calls(message, expected):
    assert convert_lc_message_to_chat_message(message) == expected


def test_transform_response_to_chat_format_no_conversion():
    response = ["list_response"]
    assert try_transform_response_to_chat_format(response) == response

    response = {"dict_response": "response"}
    assert try_transform_response_to_chat_format(response) == response


def test_transform_response_to_chat_format_conversion():
    response = "string_response"
    converted_response = try_transform_response_to_chat_format(response)
    assert isinstance(converted_response, dict)
    assert converted_response["id"] is None
    assert converted_response["choices"][0]["message"]["content"] == response

    response = AIMessage(content="ai_message_response")
    converted_response = try_transform_response_to_chat_format(response)
    assert isinstance(converted_response, dict)
    assert converted_response["id"] == getattr(response, "id", None)
    assert converted_response["choices"][0]["message"]["content"] == response.content


def test_transform_response_iter_to_chat_format_no_conversion():
    response = [{"dict_response": "response"}]
    converted_response = list(try_transform_response_iter_to_chat_format(response))
    assert len(converted_response) == 1
    assert converted_response[0] == response[0]


def test_transform_response_iter_to_chat_format_ai_message():
    response = ["string response"]
    converted_response = list(try_transform_response_iter_to_chat_format(response))
    assert len(converted_response) == 1
    assert converted_response[0]["id"] is None
    assert converted_response[0]["choices"][0]["delta"]["content"] == response[0]

    response = [
        AIMessage(
            content="ai_message_response", id="123", response_metadata={"finish_reason": "done"}
        )
    ]
    converted_response = list(try_transform_response_iter_to_chat_format(response))
    assert len(converted_response) == 1
    assert converted_response[0]["id"] == getattr(response[0], "id", None)
    assert converted_response[0]["choices"][0]["delta"]["content"] == response[0].content
    assert converted_response[0]["choices"][0]["finish_reason"] == "stop"

    response = [
        AIMessageChunk(
            content="ai_message_chunk_response",
            id="123",
            response_metadata={"finish_reason": "done"},
        ),
        AIMessageChunk(
            content="ai_message_chunk_response",
            id="456",
            response_metadata={"finish_reason": "stop"},
        ),
    ]
    converted_response = list(try_transform_response_iter_to_chat_format(response))
    assert len(converted_response) == 2
    for i in range(2):
        assert converted_response[i]["id"] == getattr(response[i], "id", None)
        assert converted_response[i]["choices"][0]["delta"]["content"] == response[i].content
        assert (
            converted_response[i]["choices"][0]["finish_reason"]
            == response[i].response_metadata["finish_reason"]
        )


def test_transform_request_json_for_chat_if_necessary_no_conversion():
    model = MagicMock(spec=AgentExecutor)
    request_json = {"messages": [{"role": "user", "content": "some_input"}]}
    assert transform_request_json_for_chat_if_necessary(request_json, model) == (
        request_json,
        False,
    )


def test_transform_request_json_for_chat_if_necessary_conversion():
    model = MagicMock(spec=SimpleChatModel)
    request_json = {"messages": [{"role": "user", "content": "some_input"}]}

    with patch("mlflow.langchain.utils.chat._get_lc_model_input_fields", return_value={"messages"}):
        transformed_request = transform_request_json_for_chat_if_necessary(request_json, model)
        assert transformed_request == (request_json, False)

    with patch(
        "mlflow.langchain.utils.chat._get_lc_model_input_fields",
        return_value={},
    ):
        transformed_request = transform_request_json_for_chat_if_necessary(request_json, model)
        assert transformed_request[0][0] == HumanMessage(content="some_input")
        assert transformed_request[1] is True

    request_json = [
        {"messages": [{"role": "system", "content": "You are a helpful assistant."}]},
        {"messages": [{"role": "assistant", "content": "What would you like to ask?"}]},
        {"messages": [{"role": "user", "content": "Who owns MLflow?"}]},
    ]
    with patch(
        "mlflow.langchain.utils.chat._get_lc_model_input_fields",
        return_value={},
    ):
        transformed_request = transform_request_json_for_chat_if_necessary(request_json, model)
        assert transformed_request[0][0][0] == SystemMessage(content="You are a helpful assistant.")
        assert transformed_request[0][1][0] == AIMessage(content="What would you like to ask?")
        assert transformed_request[0][2][0] == HumanMessage(content="Who owns MLflow?")
        assert transformed_request[1] is True


@pytest.mark.parametrize(
    ("generation", "expected"),
    [
        (ChatGeneration(message=AIMessage(content="foo", id="123")), None),
        (
            ChatGeneration(
                message=AIMessage(
                    content="foo",
                    id="123",
                    usage_metadata={"input_tokens": 5, "output_tokens": 10, "total_tokens": 15},
                )
            ),
            {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15},
        ),
        (
            ChatGeneration(
                message=AIMessageChunk(
                    content="foo",
                    id="123",
                    usage_metadata={"input_tokens": 5, "output_tokens": 10, "total_tokens": 15},
                )
            ),
            {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15},
        ),
        (
            ChatGeneration(
                message=AIMessage(
                    content="foo",
                    id="123",
                    response_metadata={
                        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
                    },
                )
            ),
            {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15},
        ),
        # Legacy completion generation object
        (Generation(text="foo"), None),
    ],
)
@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.2.0"),
    reason="Old version of LangChain does not support usage metadata",
)
def test_parse_token_usage(generation, expected):
    assert parse_token_usage([generation]) == expected
