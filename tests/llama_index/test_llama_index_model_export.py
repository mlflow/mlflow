import numpy as np
import pandas as pd
import pytest
from llama_index.core import QueryBundle
from llama_index.core.base.response.schema import (
    Response,
)
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
)
from llama_index.core.llms import ChatMessage

import mlflow
import mlflow.llama_index
import mlflow.pyfunc
from mlflow.llama_index.pyfunc_wrapper import (
    _CHAT_MESSAGE_HISTORY_PARAMETER_NAME,
    create_engine_wrapper,
)

_EMBEDDING_DIM = 1536


@pytest.fixture
def model_path(tmp_path):
    return tmp_path / "model"


@pytest.mark.parametrize(
    "index_fixture",
    [
        "single_index",
        "multi_index",
        "single_graph",
    ],
)
def test_llama_index_native_save_and_load_model(request, index_fixture, model_path):
    index = request.getfixturevalue(index_fixture)
    mlflow.llama_index.save_model(index, model_path, engine_type="query")
    loaded_model = mlflow.llama_index.load_model(model_path)

    assert type(loaded_model) == type(index)
    assert loaded_model.as_query_engine().query("Spell llamaindex").response != ""


@pytest.mark.parametrize(
    "index_fixture",
    [
        "single_index",
        "multi_index",
        "single_graph",
    ],
)
def test_llama_index_native_log_and_load_model(request, index_fixture):
    index = request.getfixturevalue(index_fixture)
    with mlflow.start_run():
        logged_model = mlflow.llama_index.log_model(index, "model", engine_type="query")

    loaded_model = mlflow.llama_index.load_model(logged_model.model_uri)

    assert "llama_index" in logged_model.flavors
    assert type(loaded_model) == type(index)
    engine = loaded_model.as_query_engine()
    assert engine is not None
    assert engine.query("Spell llamaindex").response != ""


@pytest.mark.parametrize(
    "engine_type",
    ["query", "retriever"],
)
def test_format_predict_input_correct(single_index, engine_type):
    wrapped_model = create_engine_wrapper(single_index, engine_type)

    assert isinstance(
        wrapped_model._format_predict_input(pd.DataFrame({"query_str": ["hi"]})), QueryBundle
    )
    assert isinstance(wrapped_model._format_predict_input(np.array(["hi"])), QueryBundle)
    assert isinstance(wrapped_model._format_predict_input({"query_str": ["hi"]}), QueryBundle)
    assert isinstance(wrapped_model._format_predict_input({"query_str": "hi"}), QueryBundle)
    assert isinstance(wrapped_model._format_predict_input(["hi"]), QueryBundle)
    assert isinstance(wrapped_model._format_predict_input("hi"), QueryBundle)


@pytest.mark.parametrize(
    "engine_type",
    ["query", "retriever"],
)
def test_format_predict_input_incorrect_schema(single_index, engine_type):
    wrapped_model = create_engine_wrapper(single_index, engine_type)

    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        wrapped_model._format_predict_input(pd.DataFrame({"incorrect": ["hi"]}))
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        wrapped_model._format_predict_input({"incorrect": ["hi"]})


@pytest.mark.parametrize(
    "engine_type",
    ["query", "retriever"],
)
def test_format_predict_input_correct_schema_complex(single_index, engine_type):
    wrapped_model = create_engine_wrapper(single_index, engine_type)

    payload = {
        "query_str": "hi",
        "image_path": "some/path",
        "custom_embedding_strs": [["a"]],
        "embedding": [[1.0]],
    }
    assert isinstance(wrapped_model._format_predict_input(pd.DataFrame(payload)), QueryBundle)
    payload.update(
        {
            "custom_embedding_strs": ["a"],
            "embedding": [1.0],
        }
    )
    assert isinstance(wrapped_model._format_predict_input(payload), QueryBundle)


@pytest.mark.parametrize("with_input_example", [True, False])
def test_query_engine_str(tmp_path, single_index, with_input_example):
    payload = "string"

    input_example = payload if with_input_example else None
    mlflow.llama_index.save_model(
        index=single_index, input_example=input_example, path=tmp_path, engine_type="query"
    )
    model = mlflow.pyfunc.load_model(tmp_path)
    predictions = model.predict(payload)
    assert isinstance(predictions, Response)
    assert predictions.response


@pytest.mark.parametrize("with_input_example", [True, False])
def test_query_engine_numeric(tmp_path, single_index, with_input_example):
    payload = 1

    input_example = payload if with_input_example else None
    if with_input_example:
        with pytest.raises(ValueError, match="Unsupported input type"):
            mlflow.llama_index.save_model(
                index=single_index, input_example=input_example, path=tmp_path, engine_type="query"
            )
    else:
        mlflow.llama_index.save_model(
            index=single_index, input_example=input_example, path=tmp_path, engine_type="query"
        )
        model = mlflow.pyfunc.load_model(tmp_path)
        with pytest.raises(ValueError, match="Unsupported input type"):
            _ = model.predict(payload)


@pytest.mark.parametrize("with_input_example", [True, False])
def test_query_engine_list(tmp_path, single_index, with_input_example):
    payload = ["string", "string"]

    input_example = payload if with_input_example else None
    mlflow.llama_index.save_model(
        index=single_index, input_example=input_example, path=tmp_path, engine_type="query"
    )
    model = mlflow.pyfunc.load_model(tmp_path)
    predictions = model.predict(payload)
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], Response)
    assert predictions[0].response


@pytest.mark.parametrize("with_input_example", [True, False])
def test_query_engine_array(tmp_path, single_index, with_input_example):
    payload = np.array(["string", "string"])

    input_example = payload if with_input_example else None
    mlflow.llama_index.save_model(
        index=single_index, input_example=input_example, path=tmp_path, engine_type="query"
    )
    model = mlflow.pyfunc.load_model(tmp_path)
    predictions = model.predict(payload)
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], Response)
    assert predictions[0].response


@pytest.mark.parametrize("with_input_example", [True, False])
def test_query_engine_pandas_dataframe(tmp_path, single_index, with_input_example):
    payload = pd.DataFrame({"query_str": ["string", "string"]})

    input_example = payload if with_input_example else None
    mlflow.llama_index.save_model(
        index=single_index, input_example=input_example, path=tmp_path, engine_type="query"
    )
    model = mlflow.pyfunc.load_model(tmp_path)
    predictions = model.predict(payload)
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], Response)
    assert predictions[0].response


@pytest.mark.parametrize("with_input_example", [True, False])
def test_pyfunc_predict_with_index_valid_schema_pandas(tmp_path, single_index, with_input_example):
    payload = pd.DataFrame(
        {
            "query_str": ["hi"],
            "custom_embedding_strs": [["a"] * _EMBEDDING_DIM],
            "embedding": [[1.0] * _EMBEDDING_DIM],
        }
    )

    input_example = payload if with_input_example else None
    mlflow.llama_index.save_model(
        index=single_index, input_example=input_example, path=tmp_path, engine_type="query"
    )
    model = mlflow.pyfunc.load_model(tmp_path)
    predictions = model.predict(payload)
    assert isinstance(predictions, Response)
    assert predictions.response


@pytest.mark.parametrize("with_input_example", [True, False])
def test_pyfunc_predict_with_index_valid_schema_dict(tmp_path, single_index, with_input_example):
    payload = {
        "query_str": "hi",
        "custom_embedding_strs": ["a"] * _EMBEDDING_DIM,
        "embedding": [1.0] * _EMBEDDING_DIM,
    }

    input_example = payload if with_input_example else None
    mlflow.llama_index.save_model(
        index=single_index, input_example=input_example, path=tmp_path, engine_type="query"
    )
    model = mlflow.pyfunc.load_model(tmp_path)
    predictions = model.predict(payload)
    assert isinstance(predictions, Response)
    assert predictions.response


@pytest.mark.parametrize("with_input_example", [True, False])
def test_chat_engine_str(tmp_path, single_index, with_input_example):
    payload = "string"

    input_example = payload if with_input_example else None
    mlflow.llama_index.save_model(
        index=single_index, input_example=input_example, path=tmp_path, engine_type="chat"
    )
    model = mlflow.pyfunc.load_model(tmp_path)
    predictions = model.predict(payload)
    assert isinstance(predictions, AgentChatResponse)
    assert predictions.response


@pytest.mark.parametrize("with_input_example", [True, False])
def test_chat_engine_dict(tmp_path, single_index, with_input_example):
    payload = {
        "message": "string",
        _CHAT_MESSAGE_HISTORY_PARAMETER_NAME: [{"role": "user", "content": "string"}] * 3,
    }

    input_example = payload if with_input_example else None
    mlflow.llama_index.save_model(
        index=single_index, input_example=input_example, path=tmp_path, engine_type="chat"
    )
    model = mlflow.pyfunc.load_model(tmp_path)
    predictions = model.predict(payload)
    assert isinstance(predictions, AgentChatResponse)
    assert predictions.response


@pytest.mark.parametrize("with_input_example", [True, False])
def test_chat_engine_dict_raises(tmp_path, single_index, with_input_example):
    payload = {
        "message": "string",
        "key_that_no_exist": [str(ChatMessage(role="user", content="string"))],
    }

    input_example = payload if with_input_example else None
    if with_input_example:
        with pytest.raises(TypeError, match="got an unexpected keyword argument"):
            mlflow.llama_index.save_model(
                index=single_index, input_example=input_example, path=tmp_path, engine_type="chat"
            )
    else:
        mlflow.llama_index.save_model(
            index=single_index, input_example=input_example, path=tmp_path, engine_type="chat"
        )

        model = mlflow.pyfunc.load_model(tmp_path)
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            _ = model.predict(payload)
