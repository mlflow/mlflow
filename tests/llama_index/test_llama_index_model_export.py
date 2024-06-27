import json

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
from mlflow.llama_index.pyfunc_wrapper import create_engine_wrapper
from mlflow.models import infer_signature

from tests.llama_index._llama_index_test_fixtures import (
    document,  # noqa: F401
    embed_model,  # noqa: F401
    multi_index,  # noqa: F401
    settings,  # noqa: F401
    single_graph,  # noqa: F401
    single_index,  # noqa: F401
    spark,  # noqa: F401
)


@pytest.mark.parametrize(
    "index_fixture",
    [
        "single_index",
        "multi_index",
        "single_graph",
    ],
)
def test_llama_index_native_save_and_load_model(request, index_fixture, tmp_path):
    index = request.getfixturevalue(index_fixture)
    mlflow.llama_index.save_model(index, tmp_path, engine_type="query")
    loaded_model = mlflow.llama_index.load_model(tmp_path)

    assert type(loaded_model) == type(index)
    assert loaded_model.as_query_engine().query("Spell llamaindex").response.lower() != ""


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

    with pytest.raises(ValueError, match="correct schema"):
        wrapped_model._format_predict_input(pd.DataFrame({"incorrect": ["hi"]}))
    with pytest.raises(ValueError, match="correct schema"):
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


def test_spark_udf_query(tmp_path, spark, single_index):
    df = spark.createDataFrame(
        [
            ("dummy text", "dummy text"),
        ],
        ["x"],
    )
    signature = infer_signature(model_input={"query_str": "hi"}, model_output=["output"])
    mlflow.llama_index.save_model(
        index=single_index, signature=signature, path=tmp_path, engine_type="query"
    )
    udf = mlflow.pyfunc.spark_udf(spark, tmp_path)
    pdf = df.withColumn("z", udf("x")).toPandas()
    assert list(map(json.loads, pdf["z"])) == [
        [{"content": "a b", "role": "user"}],
        [{"content": "c d", "role": "user"}],
    ]


@pytest.mark.parametrize("with_signature", [True, False])
def test_query_engine_str(tmp_path, single_index, with_signature):
    payload = "string"

    signature = infer_signature(model_input=payload) if with_signature else None
    mlflow.llama_index.save_model(
        index=single_index, signature=signature, path=tmp_path, engine_type="query"
    )
    model = mlflow.pyfunc.load_model(tmp_path)
    predictions = model.predict(payload)
    assert isinstance(predictions, Response)
    assert predictions.response


@pytest.mark.parametrize("with_signature", [True, False])
def test_query_engine_numeric(tmp_path, single_index, with_signature):
    payload = 1

    signature = infer_signature(model_input=payload) if with_signature else None
    mlflow.llama_index.save_model(
        index=single_index, signature=signature, path=tmp_path, engine_type="query"
    )
    model = mlflow.pyfunc.load_model(tmp_path)
    with pytest.raises(ValueError, match="Unsupported input type"):
        _ = model.predict(payload)


@pytest.mark.parametrize("with_signature", [True, False])
def test_query_engine_list(tmp_path, single_index, with_signature):
    payload = ["string", "string"]

    signature = infer_signature(model_input=payload) if with_signature else None
    mlflow.llama_index.save_model(
        index=single_index, signature=signature, path=tmp_path, engine_type="query"
    )
    model = mlflow.pyfunc.load_model(tmp_path)
    predictions = model.predict(payload)
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], Response)
    assert predictions[0].response


@pytest.mark.parametrize("with_signature", [True, False])
def test_query_engine_array(tmp_path, single_index, with_signature):
    payload = np.array(["string", "string"])

    signature = infer_signature(model_input=payload) if with_signature else None
    mlflow.llama_index.save_model(
        index=single_index, signature=signature, path=tmp_path, engine_type="query"
    )
    model = mlflow.pyfunc.load_model(tmp_path)
    predictions = model.predict(payload)
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], Response)
    assert predictions[0].response


@pytest.mark.parametrize("with_signature", [True, False])
def test_query_engine_pandas_dataframe(tmp_path, single_index, with_signature):
    payload = pd.DataFrame({"query_str": ["string", "string"]})

    signature = infer_signature(model_input=payload) if with_signature else None
    mlflow.llama_index.save_model(
        index=single_index, signature=signature, path=tmp_path, engine_type="query"
    )
    model = mlflow.pyfunc.load_model(tmp_path)
    predictions = model.predict(payload)
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], Response)
    assert predictions[0].response


@pytest.mark.parametrize("with_signature", [True, False])
def test_pyfunc_predict_with_index_valid_schema_pandas(tmp_path, single_index, with_signature):
    payload = pd.DataFrame(
        {"query_str": ["hi"], "custom_embedding_strs": [["a"]], "embedding": [[1.0]]}
    )

    signature = infer_signature(model_input=payload) if with_signature else None
    mlflow.llama_index.save_model(
        index=single_index, signature=signature, path=tmp_path, engine_type="query"
    )
    model = mlflow.pyfunc.load_model(tmp_path)
    predictions = model.predict(payload)
    assert isinstance(predictions, Response)
    assert predictions.response


@pytest.mark.parametrize("with_signature", [True, False])
def test_pyfunc_predict_with_index_valid_schema_dict(tmp_path, single_index, with_signature):
    payload = {"query_str": "hi", "custom_embedding_strs": ["a"], "embedding": [1.0]}

    signature = infer_signature(model_input=payload) if with_signature else None
    mlflow.llama_index.save_model(
        index=single_index, signature=signature, path=tmp_path, engine_type="query"
    )
    model = mlflow.pyfunc.load_model(tmp_path)
    predictions = model.predict(payload)
    assert isinstance(predictions, Response)
    assert predictions.response


@pytest.mark.parametrize("with_signature", [True, False])
def test_chat_engine_str(tmp_path, single_index, with_signature):
    payload = "string"

    signature = infer_signature(model_input=payload) if with_signature else None
    mlflow.llama_index.save_model(
        index=single_index, signature=signature, path=tmp_path, engine_type="chat"
    )
    model = mlflow.pyfunc.load_model(tmp_path)
    predictions = model.predict(payload)
    assert isinstance(predictions, AgentChatResponse)
    assert predictions.response


@pytest.mark.parametrize("with_signature", [True, False])
def test_chat_engine_dict(tmp_path, single_index, with_signature):
    payload = {
        "message": "string",
        "chat_history": [str(ChatMessage(role="user", content="string"))],
    }

    signature = infer_signature(model_input=payload) if with_signature else None
    mlflow.llama_index.save_model(
        index=single_index, signature=signature, path=tmp_path, engine_type="chat"
    )
    model = mlflow.pyfunc.load_model(tmp_path)
    predictions = model.predict(payload)
    assert isinstance(predictions, AgentChatResponse)
    assert predictions.response


@pytest.mark.parametrize("with_signature", [True, False])
def test_chat_engine_dict_raises(tmp_path, single_index, with_signature):
    payload = {
        "message": "string",
        "key_that_no_exist": [str(ChatMessage(role="user", content="string"))],
    }

    signature = infer_signature(model_input=payload) if with_signature else None
    mlflow.llama_index.save_model(
        index=single_index, signature=signature, path=tmp_path, engine_type="chat"
    )

    model = mlflow.pyfunc.load_model(tmp_path)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _ = model.predict(payload)
