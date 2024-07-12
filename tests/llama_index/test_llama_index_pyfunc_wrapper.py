import pandas as pd
import pytest
from llama_index.core import QueryBundle
from llama_index.core.base.response.schema import (
    NodeWithScore,
    Response,
)
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
)
from llama_index.core.llms import ChatMessage

from mlflow.llama_index.pyfunc_wrapper import (
    _CHAT_MESSAGE_HISTORY_PARAMETER_NAME,
    CHAT_ENGINE_NAME,
    QUERY_ENGINE_NAME,
    RETRIEVER_ENGINE_NAME,
    SUPPORTED_ENGINES,
    create_engine_wrapper,
)


@pytest.mark.parametrize("engine_type", SUPPORTED_ENGINES)
def test_create_create_engine_wrapper(single_index, engine_type):
    wrapped_model = create_engine_wrapper(single_index, engine_type)
    assert wrapped_model is not None
    assert wrapped_model.engine_type == engine_type
    assert engine_type in str(wrapped_model.predict_callable)


################## Inferece Input #################
def test_format_predict_input_str_chat(single_index):
    wrapped_model = create_engine_wrapper(single_index, CHAT_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input("string")
    assert formatted_data == "string"


def test_format_predict_input_dict_chat(single_index):
    wrapped_model = create_engine_wrapper(single_index, CHAT_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input({"query": "string"})
    assert isinstance(formatted_data, dict)


def test_format_predict_input_message_history_chat(single_index):
    payload = {
        "message": "string",
        _CHAT_MESSAGE_HISTORY_PARAMETER_NAME: [{"role": "user", "content": "hi"}] * 3,
    }
    wrapped_model = create_engine_wrapper(single_index, CHAT_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input(payload)

    assert isinstance(formatted_data, dict)
    assert formatted_data["message"] == payload["message"]
    assert isinstance(formatted_data[_CHAT_MESSAGE_HISTORY_PARAMETER_NAME], list)
    assert all(
        isinstance(x, ChatMessage) for x in formatted_data[_CHAT_MESSAGE_HISTORY_PARAMETER_NAME]
    )


@pytest.mark.parametrize(
    "data",
    [
        [
            {
                "query": "string",
                _CHAT_MESSAGE_HISTORY_PARAMETER_NAME: [{"role": "user", "content": "hi"}] * 3,
            }
        ]
        * 3,
        pd.DataFrame(
            [
                {
                    "query": "string",
                    _CHAT_MESSAGE_HISTORY_PARAMETER_NAME: [{"role": "user", "content": "hi"}] * 3,
                }
            ]
            * 3
        ),
    ],
)
def test_format_predict_input_message_history_chat_iterable(single_index, data):
    wrapped_model = create_engine_wrapper(single_index, CHAT_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input(data)

    if isinstance(data, pd.DataFrame):
        data = data.to_dict("records")

    assert isinstance(formatted_data, list)
    assert formatted_data[0]["query"] == data[0]["query"]
    assert isinstance(formatted_data[0][_CHAT_MESSAGE_HISTORY_PARAMETER_NAME], list)
    assert all(
        isinstance(x, ChatMessage) for x in formatted_data[0][_CHAT_MESSAGE_HISTORY_PARAMETER_NAME]
    )


def test_format_predict_input_message_history_chat_invalid_type(single_index):
    payload = {
        "message": "string",
        _CHAT_MESSAGE_HISTORY_PARAMETER_NAME: ["invalid history string", "user: hi"],
    }
    wrapped_model = create_engine_wrapper(single_index, CHAT_ENGINE_NAME)
    with pytest.raises(ValueError, match="It must be a list of dicts"):
        _ = wrapped_model._format_predict_input(payload)


@pytest.mark.parametrize(
    "data",
    [
        "string",
        ["string"],  # iterables of length 1 should be treated non-iterables
        {"query_str": "string"},
        {"query_str": "string", "custom_embedding_strs": ["string"], "embedding": [1.0]},
        pd.DataFrame(
            {"query_str": ["string"], "custom_embedding_strs": [["string"]], "embedding": [[1.0]]}
        ),
    ],
)
def test_format_predict_input_no_iterable_query(single_index, data):
    wrapped_model = create_engine_wrapper(single_index, QUERY_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input(data)
    assert isinstance(formatted_data, QueryBundle)


@pytest.mark.parametrize(
    "data",
    [
        ["string", "string"],
        [{"query_str": "string"}] * 4,
        [{"query_str": "string", "custom_embedding_strs": ["string"], "embedding": [1.0]}] * 4,
        [
            pd.DataFrame(
                {
                    "query_str": ["string"],
                    "custom_embedding_strs": [["string"]],
                    "embedding": [[1.0]],
                }
            )
        ]
        * 2,
    ],
)
def test_format_predict_input_iterable_query(single_index, data):
    wrapped_model = create_engine_wrapper(single_index, QUERY_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input(data)

    assert isinstance(formatted_data, list)
    assert all(isinstance(x, QueryBundle) for x in formatted_data)


@pytest.mark.parametrize(
    "data",
    [
        "string",
        ["string"],  # iterables of length 1 should be treated non-iterables
        {"query_str": "string"},
        {"query_str": "string", "custom_embedding_strs": ["string"], "embedding": [1.0]},
        pd.DataFrame(
            {"query_str": ["string"], "custom_embedding_strs": [["string"]], "embedding": [[1.0]]}
        ),
    ],
)
def test_format_predict_input_no_iterable_retriever(single_index, data):
    wrapped_model = create_engine_wrapper(single_index, RETRIEVER_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input(data)
    assert isinstance(formatted_data, QueryBundle)


@pytest.mark.parametrize(
    "data",
    [
        ["string", "string"],
        [{"query_str": "string"}] * 4,
        [{"query_str": "string", "custom_embedding_strs": ["string"], "embedding": [1.0]}] * 4,
        [
            pd.DataFrame(
                {
                    "query_str": ["string"],
                    "custom_embedding_strs": [["string"]],
                    "embedding": [[1.0]],
                }
            )
        ]
        * 2,
    ],
)
def test_format_predict_input_iterable_retriever(single_index, data):
    wrapped_model = create_engine_wrapper(single_index, RETRIEVER_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input(data)
    assert isinstance(formatted_data, list)
    assert all(isinstance(x, QueryBundle) for x in formatted_data)


#### E2E Inference ####
def test_predict_query(single_index):
    payload = "string"
    wrapped_model = create_engine_wrapper(single_index, "chat")
    predictions = wrapped_model.predict(payload)
    assert isinstance(predictions, AgentChatResponse)
    assert predictions.response


def test_predict_query(single_index):
    payload = "string"
    wrapped_model = create_engine_wrapper(single_index, "query")
    predictions = wrapped_model.predict(payload)
    assert isinstance(predictions, Response)
    assert predictions.response


def test_predict_query(single_index):
    payload = "string"
    wrapped_model = create_engine_wrapper(single_index, "retriever")
    predictions = wrapped_model.predict(payload)
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], NodeWithScore)
