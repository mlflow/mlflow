from unittest.mock import MagicMock, patch

from langchain.agents import AgentExecutor
from langchain.chat_models.base import SimpleChatModel
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.ai import AIMessageChunk

from mlflow.langchain.utils.chat import (
    transform_request_json_for_chat_if_necessary,
    try_transform_response_iter_to_chat_format,
    try_transform_response_to_chat_format,
)


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
