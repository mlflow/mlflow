# Cases
# - Graph
# - Single index
# - Mutli index

import os
import pathlib
import shutil

import pytest
from llama_index.core import Document, KnowledgeGraphIndex, VectorStoreIndex

import mlflow
import mlflow.llama_index
from mlflow.models import Model
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env

# TODO: figure out how to mock llama index llm
os.environ["OPENAI_API_KEY"] = "sk-"


@pytest.fixture
def document():
    # Contents of document is information about LLMs and their usage
    return Document.example()


@pytest.fixture
def single_index(document):
    return VectorStoreIndex.from_documents([document])


@pytest.fixture
def single_graph(document):
    return KnowledgeGraphIndex.from_documents([document])


@pytest.fixture
def model_path(tmp_path):
    model_path = tmp_path.joinpath("model")
    yield model_path

    # Pytest keeps the temporary directory created by `tmp_path` fixture for 3 recent test sessions
    # by default. This is useful for debugging during local testing, but in CI it just wastes the
    # disk space.
    if os.getenv("GITHUB_ACTIONS") == "true":
        shutil.rmtree(model_path, ignore_errors=True)


@pytest.mark.parametrize(
    ("index_fixture", "should_start_run"),
    [
        ("single_index", True),
        ("single_index", False),
        ("single_graph", True),
        ("single_graph", False),
    ],
)
def test_log_and_load_index(request, tmp_path, index_fixture, should_start_run):
    try:
        if should_start_run:
            mlflow.start_run()
        artifact_path = "index"
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["llama_index"])
        index = request.getfixturevalue(index_fixture)
        model_info = mlflow.llama_index.log_model(
            index=index,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        assert model_info.model_uri == model_uri
        reloaded_model = mlflow.llama_index.load_model(model_uri=model_uri)
        assert (
            reloaded_model.as_query_engine(temperature=0.0)
            .query("Spell llamaindex")
            .response.lower()
            != ""
        )
        model_path = pathlib.Path(_download_artifact_from_uri(artifact_uri=model_uri))
        model_config = Model.load(str(model_path.joinpath("MLmodel")))
        assert mlflow.pyfunc.FLAVOR_NAME in model_config.flavors
        assert mlflow.pyfunc.ENV in model_config.flavors[mlflow.pyfunc.FLAVOR_NAME]
        env_path = model_config.flavors[mlflow.pyfunc.FLAVOR_NAME][mlflow.pyfunc.ENV]["conda"]
        assert model_path.joinpath(env_path).exists()
    finally:
        mlflow.end_run()
