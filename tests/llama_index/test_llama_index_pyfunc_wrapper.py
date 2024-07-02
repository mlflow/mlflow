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
    _CHAT_MESSAGE_HISTORY_PARAMETER_NAMES,
    _CHAT_MESSAGE_PARAMETER_NAMES,
    CHAT_ENGINE_NAME,
    QUERY_ENGINE_NAME,
    RETRIEVER_ENGINE_NAME,
    SUPPORTED_ENGINES,
    create_engine_wrapper,
)

from tests.llama_index._llama_index_test_fixtures import (
    document,  # noqa: F401
    embed_model,  # noqa: F401
    multi_index,  # noqa: F401
    settings,  # noqa: F401
    single_graph,  # noqa: F401
    single_index,  # noqa: F401
    spark,  # noqa: F401
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
    assert isinstance(formatted_data, str)


def test_format_predict_input_dict_chat(single_index):
    wrapped_model = create_engine_wrapper(single_index, CHAT_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input({"query": "string"})
    assert isinstance(formatted_data, dict)


@pytest.mark.parametrize(
    ("query_key", "chat_history_key"),
    [
        (query_key, chat_history_key)
        for query_key in _CHAT_MESSAGE_PARAMETER_NAMES
        for chat_history_key in _CHAT_MESSAGE_HISTORY_PARAMETER_NAMES
    ],
)
def test_format_predict_input_message_history_chat(single_index, query_key, chat_history_key):
    payload = {
        "query": "string",
        "message_history": [{"role": "user", "content": "hi"}] * 3,
    }
    wrapped_model = create_engine_wrapper(single_index, CHAT_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input(payload)

    assert isinstance(formatted_data, dict)
    assert formatted_data["query"] == payload["query"]
    assert isinstance(formatted_data["message_history"], list)
    assert all(isinstance(x, dict) for x in formatted_data["message_history"])
    assert ChatMessage(**formatted_data["message_history"][0])


@pytest.mark.parametrize(
    "data",
    [
        [
            {
                "query": "string",
                "message_history": [{"role": "user", "content": "hi"}] * 3,
            }
        ]
        * 3,
        pd.DataFrame(
            [
                {
                    "query": "string",
                    "message_history": [{"role": "user", "content": "hi"}] * 3,
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
    assert isinstance(formatted_data[0]["message_history"], list)
    assert all(isinstance(x, dict) for x in formatted_data[0]["message_history"])
    assert ChatMessage(**formatted_data[0]["message_history"][0])


def test_format_predict_input_message_history_chat_invalid(single_index):
    payload = {"query": "string", "message_history": ["invalid history string", "user: hi"]}
    wrapped_model = create_engine_wrapper(single_index, CHAT_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input(payload)

    assert isinstance(formatted_data, dict)
    assert formatted_data["query"] == payload["query"]
    assert isinstance(formatted_data["message_history"], list)
    assert not any(isinstance(x, ChatMessage) for x in formatted_data["message_history"])


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
