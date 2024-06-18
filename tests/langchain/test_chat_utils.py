from unittest.mock import MagicMock, patch

import pytest
from langchain.agents import AgentExecutor
from langchain.chat_models.base import SimpleChatModel
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.ai import AIMessageChunk

from mlflow.langchain.chat_utils import (
    _transform_request_json_for_chat_if_necessary,
    _try_transform_response_iter_to_chat_format,
    _try_transform_response_to_chat_format,
)


@pytest.mark.parametrize(
    "response",
    [
        "string_response",
        {"dict_response": "response"},
        ["list_response"],
        AIMessage(content="ai_message_response"),
    ],
)
def test_transform_response_to_chat_format(response):
    if isinstance(response, (list, dict)):
        assert _try_transform_response_to_chat_format(response) == response
    else:
        if isinstance(response, str):
            converted_response = _try_transform_response_to_chat_format(response)
            assert isinstance(converted_response, dict)
            assert converted_response["id"] is None
            assert converted_response["choices"][0]["message"]["content"] == response
        elif isinstance(response, AIMessage):
            converted_response = _try_transform_response_to_chat_format(response)
            assert isinstance(converted_response, dict)
            assert converted_response["id"] == getattr(response, "id", None)
            assert converted_response["choices"][0]["message"]["content"] == response.content


@pytest.mark.parametrize(
    "response",
    [
        ["string_response"],
        [{"dict_response": "response"}],
        [AIMessage(content="ai_message_response")],
        [
            AIMessageChunk(
                content="ai_message_chunk_response",
                id="123",
                response_metadata={"finish_reason": "done"},
            )
        ],
    ],
)
def test_transform_response_iter_to_chat_format(response):
    converted_response = list(_try_transform_response_iter_to_chat_format(response))
    assert len(converted_response) == 1
    if isinstance(response[0], (list, dict)):
        assert converted_response[0] == response[0]
    else:
        if isinstance(converted_response[0], str):
            assert converted_response[0]["id"] is None
            assert converted_response[0]["choices"][0]["delta"]["content"] == response[0]
        elif isinstance(response[0], AIMessageChunk):
            assert converted_response[0]["id"] == getattr(response[0], "id", None)
            assert converted_response[0]["choices"][0]["delta"]["content"] == response[0].content
            assert converted_response[0]["choices"][0]["finish_reason"] == "done"
        elif isinstance(response[0], AIMessage):
            assert converted_response[0]["id"] == getattr(response[0], "id", None)
            assert converted_response[0]["choices"][0]["delta"]["content"] == response[0].content
            assert converted_response[0]["choices"][0]["finish_reason"] == "stop"


def test_transform_request_json_for_chat_if_necessary_no_conversion():
    model = MagicMock(spec=AgentExecutor)
    request_json = {"messages": [{"role": "user", "content": "some_input"}]}
    assert _transform_request_json_for_chat_if_necessary(request_json, model) == (
        request_json,
        False,
    )


def test_transform_request_json_for_chat_if_necessary_conversion():
    model = MagicMock(spec=SimpleChatModel)
    request_json = {"messages": [{"role": "user", "content": "some_input"}]}

    with patch("mlflow.langchain.chat_utils._get_lc_model_input_fields", return_value={"messages"}):
        transformed_request = _transform_request_json_for_chat_if_necessary(request_json, model)
        assert transformed_request == (request_json, False)

    with patch(
        "mlflow.langchain.chat_utils._get_lc_model_input_fields",
        return_value={},
    ):
        transformed_request = _transform_request_json_for_chat_if_necessary(request_json, model)
        assert transformed_request[0][0] == HumanMessage(content="some_input")
        assert transformed_request[1] is True

    request_json = [
        {"messages": [{"role": "system", "content": "You are a helpful assistant."}]},
        {"messages": [{"role": "assistant", "content": "What would you like to ask?"}]},
        {"messages": [{"role": "user", "content": "Who owns MLflow?"}]},
    ]
    with patch(
        "mlflow.langchain.chat_utils._get_lc_model_input_fields",
        return_value={},
    ):
        transformed_request = _transform_request_json_for_chat_if_necessary(request_json, model)
        assert transformed_request[0][0][0] == SystemMessage(content="You are a helpful assistant.")
        assert transformed_request[0][1][0] == AIMessage(content="What would you like to ask?")
        assert transformed_request[0][2][0] == HumanMessage(content="Who owns MLflow?")
        assert transformed_request[1] is True
