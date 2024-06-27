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
@pytest.mark.parametrize(
    "data",
    [
        "string",
        ["string"],  # iterables of length 1 should be treated non-iterables
        {"query_str": "string"},
        {"some_key": "string"},
    ],
)
def test_format_predict_input_str_chat(single_index, data):
    wrapped_model = create_engine_wrapper(single_index, CHAT_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input(data)
    assert isinstance(formatted_data, str)


def test_format_predict_input_message_history_chat(single_index):
    payload = {
        "query": "string",
        "message_history": [str(ChatMessage(role="user", content="string"))] * 3,
    }
    wrapped_model = create_engine_wrapper(single_index, CHAT_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input(payload)

    assert isinstance(formatted_data, dict)
    assert formatted_data["query"] == payload["query"]
    assert isinstance(formatted_data["message_history"], list)
    assert all(isinstance(x, ChatMessage) for x in formatted_data["message_history"])
    assert formatted_data["message_history"][0] == ChatMessage(role="user", content="string")


@pytest.mark.parametrize(
    "data",
    [
        [
            {
                "query": "string",
                "message_history": [str(ChatMessage(role="user", content="string"))] * 3,
            }
        ]
        * 3,
        pd.DataFrame(
            [
                {
                    "query": "string",
                    "message_history": [str(ChatMessage(role="user", content="string"))] * 3,
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
    assert all(isinstance(x, ChatMessage) for x in formatted_data[0]["message_history"])
    assert formatted_data[0]["message_history"][0] == ChatMessage(role="user", content="string")


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
