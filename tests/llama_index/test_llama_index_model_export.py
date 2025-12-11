import json
import os
from pathlib import Path
from typing import Any
from unittest import mock

import llama_index.core
import numpy as np
import pandas as pd
import pytest
from llama_index.core import QueryBundle, Settings, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.llms import ChatMessage
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.embeddings.databricks import DatabricksEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.databricks import Databricks
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from packaging.version import Version

import mlflow
import mlflow.llama_index
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
from mlflow.llama_index.pyfunc_wrapper import (
    _CHAT_MESSAGE_HISTORY_PARAMETER_NAME,
    ChatEngineWrapper,
    QueryEngineWrapper,
    create_pyfunc_wrapper,
)
from mlflow.models.utils import load_serving_example
from mlflow.pyfunc.scoring_server import CONTENT_TYPE_JSON
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types.schema import ColSpec, DataType, Schema

from tests.helper_functions import pyfunc_scoring_endpoint

_EMBEDDING_DIM = 1536
_TEST_QUERY = "Spell llamaindex"


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
    assert loaded_model.as_query_engine().query(_TEST_QUERY).response != ""


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
        logged_model = mlflow.llama_index.log_model(index, name="model", engine_type="query")

    loaded_model = mlflow.llama_index.load_model(logged_model.model_uri)

    assert "llama_index" in logged_model.flavors
    assert type(loaded_model) == type(index)
    engine = loaded_model.as_query_engine()
    assert engine is not None
    assert engine.query(_TEST_QUERY).response != ""


def test_llama_index_save_invalid_object_raise(single_index):
    with pytest.raises(MlflowException, match="The provided object of type "):
        mlflow.llama_index.save_model(llama_index_model=OpenAI(), path="model", engine_type="query")

    with pytest.raises(MlflowException, match="Saving a non-index object is only supported"):
        mlflow.llama_index.save_model(
            llama_index_model=single_index.as_query_engine(),
            path="model",
        )


def test_llama_index_load_with_model_config(single_index):
    from llama_index.core.response_synthesizers import Refine

    with mlflow.start_run():
        logged_model = mlflow.llama_index.log_model(
            single_index,
            name="model",
            engine_type="query",
            model_config={"response_mode": "refine"},
        )

    loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
    engine = loaded_model.get_raw_model()

    assert isinstance(engine._response_synthesizer, Refine)


@pytest.mark.parametrize(
    "engine_type",
    ["query", "retriever"],
)
def test_format_predict_input_correct(single_index, engine_type):
    wrapped_model = create_pyfunc_wrapper(single_index, engine_type)

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
    wrapped_model = create_pyfunc_wrapper(single_index, engine_type)

    exception_error = (
        r"__init__\(\) got an unexpected keyword argument 'incorrect'"
        if Version(llama_index.core.__version__) >= Version("0.11.0")
        else r"missing 1 required positional argument"
    )

    with pytest.raises(TypeError, match=exception_error):
        wrapped_model._format_predict_input(pd.DataFrame({"incorrect": ["hi"]}))
    with pytest.raises(TypeError, match=exception_error):
        wrapped_model._format_predict_input({"incorrect": ["hi"]})


@pytest.mark.parametrize(
    "engine_type",
    ["query", "retriever"],
)
def test_format_predict_input_correct_schema_complex(single_index, engine_type):
    wrapped_model = create_pyfunc_wrapper(single_index, engine_type)

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
            single_index,
            name="model",
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
            single_index,
            name="model",
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
                llama_index_model=single_index,
                input_example=input_example,
                path=model_path,
                engine_type="query",
            )
    else:
        mlflow.llama_index.save_model(
            llama_index_model=single_index, path=model_path, engine_type="query"
        )
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
            single_index,
            name="model",
            input_example=payload if with_input_example else None,
            engine_type="chat",
        )

    if with_input_example:
        assert model_info.signature.inputs is not None
        assert model_info.signature.outputs == Schema([ColSpec(type=DataType.string)])

    model = mlflow.pyfunc.load_model(model_info.model_uri)
    prediction = model.predict(payload)
    assert isinstance(prediction, str)
    # a default prompt is added in llama-index 0.13.0
    # https://github.com/run-llama/llama_index/blob/1e02c7a2324838f7bd5a52c811d35c30dc6a6bd2/llama-index-core/llama_index/core/chat_engine/condense_plus_context.py#L40
    assert '{"role": "user", "content": "string"}' in prediction


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
                llama_index_model=single_index,
                input_example=input_example,
                path=model_path,
                engine_type="chat",
            )
    else:
        mlflow.llama_index.save_model(
            llama_index_model=single_index,
            input_example=input_example,
            path=model_path,
            engine_type="chat",
        )

        model = mlflow.pyfunc.load_model(model_path)
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            _ = model.predict(payload)


@pytest.mark.parametrize("with_input_example", [True, False])
def test_retriever_engine_predict(single_index, with_input_example):
    payload = "string"
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            single_index,
            name="model",
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


def test_llama_index_databricks_integration(monkeypatch, document, model_path, mock_openai):
    monkeypatch.setenv("DATABRICKS_TOKEN", "test")
    monkeypatch.setenv("DATABRICKS_SERVING_ENDPOINT", mock_openai)
    monkeypatch.setattr(Settings, "llm", Databricks(model="dbrx-instruct"))
    monkeypatch.setattr(
        Settings, "embed_model", DatabricksEmbedding(model="databricks-bge-large-en")
    )

    index = VectorStoreIndex(nodes=[document])
    mlflow.llama_index.save_model(
        llama_index_model=index, path=model_path, input_example="hi", engine_type="query"
    )

    with model_path.joinpath("requirements.txt").open() as file:
        requirements = file.read()
    reqs = {req.split("==")[0] for req in requirements.split("\n") if requirements}
    assert "llama-index" in reqs
    # The subpackage should be listed as requirements while they are dependencies
    # of the root "llama-index" package. This is to ensure that the version of
    # the subpackage is fixed.
    assert "llama-index-core" in reqs

    # Subpackage that is not declared as a dependency of the root package
    assert "llama-index-llms-databricks" in reqs
    assert "llama-index-embeddings-databricks" in reqs

    # Reset setting before loading back to validate the setting deserialization
    class _FakeLLM(OpenAI):
        def chat(self, *args, **kwargs: Any):
            raise Exception("Should not be called")

    class _FakeEmbedding(OpenAIEmbedding):
        def _get_query_embedding(self, *args, **kwargs: Any):
            raise Exception("Should not be called")

    # Set dummy settings to validate the deserialization
    monkeypatch.setattr(Settings, "llm", _FakeLLM())
    monkeypatch.setattr(Settings, "embed_model", _FakeEmbedding())

    # validate if the mocking works
    with pytest.raises(Exception, match="Should not be called"):
        index.as_query_engine().query(_TEST_QUERY)

    loaded_model = mlflow.pyfunc.load_model(model_path)

    response = loaded_model.predict(_TEST_QUERY)
    assert isinstance(response, str)
    assert _TEST_QUERY in response


@pytest.mark.parametrize(
    ("index_code_path", "vector_store_class"),
    [
        (
            "tests/llama_index/sample_code/basic_vector_store.py",
            SimpleVectorStore,
        ),
        (
            "tests/llama_index/sample_code/external_vector_store.py",
            QdrantVectorStore,
        ),
    ],
)
def test_save_load_index_as_code_index(index_code_path, vector_store_class):
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            index_code_path,
            name="model",
            engine_type="query",
            input_example="hi",
        )

    artifact_path = Path(_download_artifact_from_uri(model_info.model_uri))
    assert os.path.exists(artifact_path / os.path.basename(index_code_path))
    assert not os.path.exists(artifact_path / "index")
    assert os.path.exists(artifact_path / "settings.json")

    loaded_index = mlflow.llama_index.load_model(model_info.model_uri)
    assert isinstance(loaded_index.vector_store, vector_store_class)

    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert isinstance(pyfunc_loaded_model.get_raw_model(), BaseQueryEngine)
    assert _TEST_QUERY in pyfunc_loaded_model.predict(_TEST_QUERY)


def test_save_load_query_engine_as_code():
    index_code_path = "tests/llama_index/sample_code/query_engine_with_reranker.py"
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            index_code_path,
            name="model",
            input_example="hi",
        )

    loaded_engine = mlflow.llama_index.load_model(model_info.model_uri)
    assert isinstance(loaded_engine, BaseQueryEngine)
    processors = loaded_engine._node_postprocessors
    assert len(processors) == 2
    assert processors[0].__class__.__name__ == "LLMRerank"
    assert processors[1].__class__.__name__ == "CustomNodePostprocessor"

    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert isinstance(pyfunc_loaded_model._model_impl, QueryEngineWrapper)
    assert isinstance(pyfunc_loaded_model.get_raw_model(), BaseQueryEngine)
    assert pyfunc_loaded_model.predict(_TEST_QUERY) != ""
    custom_processor = pyfunc_loaded_model.get_raw_model()._node_postprocessors[1]
    assert custom_processor.call_count == 1


def test_save_load_chat_engine_as_code():
    index_code_path = "tests/llama_index/sample_code/basic_chat_engine.py"
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            index_code_path,
            name="model",
            input_example="hi",
        )

    loaded_engine = mlflow.llama_index.load_model(model_info.model_uri)
    # The sample code sets chat mode to SIMPLE, so it should be a SimpleChatEngine
    assert isinstance(loaded_engine, SimpleChatEngine)

    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert isinstance(pyfunc_loaded_model._model_impl, ChatEngineWrapper)
    assert isinstance(pyfunc_loaded_model.get_raw_model(), SimpleChatEngine)
    assert pyfunc_loaded_model.predict(_TEST_QUERY) != ""


@pytest.mark.parametrize(
    ("index_code_path", "model_config"),
    [
        (
            "tests/llama_index/sample_code/with_model_config.py",
            {
                "model_name": "gpt-4o-mini",
                "temperature": 0.7,
            },
        ),
        (
            "tests/llama_index/sample_code/with_model_config_yaml_file.py",
            "tests/llama_index/sample_code/model_config.yaml",
        ),
    ],
)
def test_save_load_as_code_with_model_config(index_code_path, model_config):
    with mlflow.start_run():
        logged_model = mlflow.llama_index.log_model(
            index_code_path,
            name="model",
            model_config=model_config,
        )

    loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
    engine = loaded_model.get_raw_model()
    assert engine._llm.model == "gpt-4o-mini"
    assert engine._llm.temperature == 0.7


def test_save_engine_with_engine_type_issues_warning(model_path):
    index_code_path = "tests/llama_index/sample_code/query_engine_with_reranker.py"

    with mock.patch("mlflow.llama_index.model._logger") as mock_logger:
        mlflow.llama_index.save_model(
            llama_index_model=index_code_path,
            path=model_path,
            engine_type="query",
        )

    assert mock_logger.warning.call_count == 1
    assert "The `engine_type` argument" in mock_logger.warning.call_args[0][0]


@pytest.mark.skipif(
    Version(llama_index.core.__version__) < Version("0.11.0"),
    reason="Workflow was introduced in 0.11.0",
)
@pytest.mark.asyncio
async def test_save_load_workflow_as_code():
    from llama_index.core.workflow import Workflow

    index_code_path = "tests/llama_index/sample_code/simple_workflow.py"
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            index_code_path,
            name="model",
            input_example={"topic": "pirates"},
        )

    # Signature
    assert model_info.signature.inputs == Schema([ColSpec(type=DataType.string, name="topic")])
    assert model_info.signature.outputs == Schema([ColSpec(DataType.string)])

    # Native inference
    loaded_workflow = mlflow.llama_index.load_model(model_info.model_uri)
    assert isinstance(loaded_workflow, Workflow)
    result = await loaded_workflow.run(topic="pirates")
    assert isinstance(result, str)
    assert "pirates" in result

    # Pyfunc inference
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert isinstance(pyfunc_loaded_model.get_raw_model(), Workflow)
    result = pyfunc_loaded_model.predict({"topic": "pirates"})
    assert isinstance(result, str)
    assert "pirates" in result

    # Batch inference
    batch_result = pyfunc_loaded_model.predict(
        [
            {"topic": "pirates"},
            {"topic": "ninjas"},
            {"topic": "robots"},
        ]
    )
    assert len(batch_result) == 3
    assert all(isinstance(r, str) for r in batch_result)

    # Serve
    inference_payload = load_serving_example(model_info.model_uri)

    with pyfunc_scoring_endpoint(
        model_uri=model_info.model_uri,
        extra_args=["--env-manager", "local"],
    ) as endpoint:
        # Single input
        response = endpoint.invoke(inference_payload, content_type=CONTENT_TYPE_JSON)
        assert response.status_code == 200, response.text
        assert response.json()["predictions"] == result

        # Batch input
        df = pd.DataFrame({"topic": ["pirates", "ninjas", "robots"]})
        response = endpoint.invoke(
            json.dumps({"dataframe_split": df.to_dict(orient="split")}),
            content_type=CONTENT_TYPE_JSON,
        )
        assert response.status_code == 200, response.text
        assert response.json()["predictions"] == batch_result
