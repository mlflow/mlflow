import numpy as np
import pandas as pd
import pytest
from llama_index.core import QueryBundle
from llama_index.core.llms import ChatMessage

import mlflow
import mlflow.llama_index
import mlflow.pyfunc
from mlflow.llama_index.pyfunc_wrapper import (
    _CHAT_MESSAGE_HISTORY_PARAMETER_NAME,
    create_engine_wrapper,
)
from mlflow.types.schema import ColSpec, DataType, Schema

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
@pytest.mark.parametrize(
    "payload",
    [
        "string",
        pd.DataFrame({"query_str": ["string"]}),
        # Dict with custom schema
        {
            "query_str": "hi",
            "custom_embedding_strs": ["a"] * _EMBEDDING_DIM,
            "embedding": [1.0] * _EMBEDDING_DIM,
        },
    ],
)
def test_query_engine_predict(single_index, with_input_example, payload):
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            index=single_index,
            artifact_path="model",
            input_example=payload if with_input_example else None,
            engine_type="query",
        )

    if with_input_example:
        assert model_info.signature.inputs is not None
        assert model_info.signature.outputs == Schema([ColSpec(type=DataType.string)])

    model = mlflow.pyfunc.load_model(model_info.model_uri)

    prediction = model.predict(payload)
    assert isinstance(prediction, str)
    assert prediction.startswith('[{"role": "system",')


@pytest.mark.parametrize("with_input_example", [True, False])
@pytest.mark.parametrize(
    "payload",
    [
        ["string", "string"],
        np.array(["string", "string"]),
        pd.DataFrame({"query_str": ["string", "string"]}),
        pd.DataFrame(
            {
                "query_str": ["hi"] * 2,
                "custom_embedding_strs": [["a"] * _EMBEDDING_DIM] * 2,
                "embedding": [[1.0] * _EMBEDDING_DIM] * 2,
            }
        ),
    ],
)
def test_query_engine_predict_list(single_index, with_input_example, payload):
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            index=single_index,
            artifact_path="model",
            input_example=payload if with_input_example else None,
            engine_type="query",
        )

    if with_input_example:
        assert model_info.signature.inputs is not None
        assert model_info.signature.outputs == Schema([ColSpec(type=DataType.string)])

    model = mlflow.pyfunc.load_model(model_info.model_uri)
    predictions = model.predict(payload)

    assert isinstance(predictions, list)
    assert len(predictions) == 2
    assert all(isinstance(p, str) for p in predictions)
    assert all(p.startswith('[{"role": "system",') for p in predictions)


@pytest.mark.parametrize("with_input_example", [True, False])
def test_query_engine_predict_numeric(model_path, single_index, with_input_example):
    payload = 1

    input_example = payload if with_input_example else None
    if with_input_example:
        with pytest.raises(ValueError, match="Unsupported input type"):
            mlflow.llama_index.save_model(
                index=single_index,
                input_example=input_example,
                path=model_path,
                engine_type="query",
            )
    else:
        mlflow.llama_index.save_model(index=single_index, path=model_path, engine_type="query")
        model = mlflow.pyfunc.load_model(model_path)
        with pytest.raises(ValueError, match="Unsupported input type"):
            _ = model.predict(payload)


@pytest.mark.parametrize("with_input_example", [True, False])
@pytest.mark.parametrize(
    "payload",
    [
        "string",
        {
            "message": "string",
            _CHAT_MESSAGE_HISTORY_PARAMETER_NAME: [{"role": "user", "content": "string"}] * 3,
        },
        pd.DataFrame(
            {
                "message": ["string"],
                _CHAT_MESSAGE_HISTORY_PARAMETER_NAME: [[{"role": "user", "content": "string"}]],
            }
        ),
    ],
)
def test_chat_engine_predict(single_index, with_input_example, payload):
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            index=single_index,
            artifact_path="model",
            input_example=payload if with_input_example else None,
            engine_type="chat",
        )

    if with_input_example:
        assert model_info.signature.inputs is not None
        assert model_info.signature.outputs == Schema([ColSpec(type=DataType.string)])

    model = mlflow.pyfunc.load_model(model_info.model_uri)
    prediction = model.predict(payload)
    assert isinstance(prediction, str)
    assert prediction.startswith('[{"role": "user",')


@pytest.mark.parametrize("with_input_example", [True, False])
def test_chat_engine_dict_raises(model_path, single_index, with_input_example):
    payload = {
        "message": "string",
        "key_that_no_exist": [str(ChatMessage(role="user", content="string"))],
    }

    input_example = payload if with_input_example else None
    if with_input_example:
        with pytest.raises(TypeError, match="got an unexpected keyword argument"):
            mlflow.llama_index.save_model(
                index=single_index, input_example=input_example, path=model_path, engine_type="chat"
            )
    else:
        mlflow.llama_index.save_model(
            index=single_index, input_example=input_example, path=model_path, engine_type="chat"
        )

        model = mlflow.pyfunc.load_model(model_path)
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            _ = model.predict(payload)


@pytest.mark.parametrize("with_input_example", [True, False])
def test_retriever_engine_predict(single_index, with_input_example):
    payload = "string"
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            index=single_index,
            artifact_path="model",
            input_example=payload if with_input_example else None,
            engine_type="retriever",
        )

    if with_input_example:
        assert model_info.signature.inputs is not None
        # TODO: Inferring signature from retriever output fails because the schema
        # does not allow None value. This is a bug in the schema inference.
        # assert model_info.signature.outputs is not None

    model = mlflow.pyfunc.load_model(model_info.model_uri)

    predictions = model.predict(payload)
    assert all(p["class_name"] == "NodeWithScore" for p in predictions)
