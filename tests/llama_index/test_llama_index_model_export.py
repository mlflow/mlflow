import os
import shutil
from contextlib import contextmanager

import pytest
from llama_index.core import Document, KnowledgeGraphIndex, Settings, VectorStoreIndex
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.llms.mock import MockLLM

import mlflow
import mlflow.llama_index
import mlflow.pyfunc
from mlflow.utils.environment import _mlflow_conda_env

# BERK NOTE: when below are None, mocks are used by default
# Settings.embed_model = "default"
# Settings.llm = "default"


# TODO
@contextmanager
def settings_context(do_llm: bool = False, do_embed: bool = False):
    if do_llm:
        Settings.llm = MockLLM()
    if do_embed:
        Settings.embed_model = MockEmbedding(embed_dim=1)

    try:
        yield Settings
    finally:
        Settings.embed_model = "default"
        Settings.llm = "default"


@pytest.fixture
def document():
    # Contents of document is information about LLMs and their usage
    return Document.example()


@pytest.fixture
def single_index(document):
    # service_context = ServiceContext.from_defaults(
    # llm_predictor=MockLLM(), embed_model=MockEmbedding(embed_dim=1))
    return VectorStoreIndex(nodes=[document], embed_model=MockEmbedding(embed_dim=1))
    # index = VectorStoreIndex.from_documents([document])
    # index.service_context = service_context
    # return index


@pytest.fixture
def multi_index(document):
    return VectorStoreIndex(nodes=[document] * 5, embed_model=MockEmbedding(embed_dim=1))


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


# @pytest.mark.parametrize(
#     ("index_fixture", "should_start_run"),
#     [
#         ("single_index", True),
#         ("single_index", False),
#     ],
# )
# def test_log_and_load_index(request, tmp_path, index_fixture, should_start_run):
#       " THIS IS GOOD!"
#     try:
#         if should_start_run:
#             mlflow.start_run()
#         artifact_path = "index"
#         conda_env = tmp_path.joinpath("conda_env.yaml")
#         _mlflow_conda_env(conda_env, additional_pip_deps=["llama_index"])
#         index = request.getfixturevalue(index_fixture)
#         model_info = mlflow.llama_index.log_model(
#             index=index,
#             artifact_path=artifact_path,
#             engine_type='retriever',
#             engine_config={},
#             conda_env=str(conda_env),
#         )
#         model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
#         assert model_info.model_uri == model_uri
#         reloaded_model = mlflow.llama_index.load_model(model_uri=model_uri)
#         Settings.llm = MockLLM()
#         assert reloaded_model.as_chat_engine().chat("Spell llamaindex") != ""
#         model_path = pathlib.Path(_download_artifact_from_uri(artifact_uri=model_uri))
#         model_config = Model.load(str(model_path.joinpath("MLmodel")))
#         assert mlflow.pyfunc.FLAVOR_NAME in model_config.flavors
#         assert mlflow.pyfunc.ENV in model_config.flavors[mlflow.pyfunc.FLAVOR_NAME]
#         env_path = model_config.flavors[mlflow.pyfunc.FLAVOR_NAME][mlflow.pyfunc.ENV]["conda"]
#         assert model_path.joinpath(env_path).exists()
#     finally:
#         mlflow.end_run()


def test_log_and_load_index_pyfunc(request, tmp_path, single_index):
    try:
        with settings_context(do_embed=True, do_llm=True):
            mlflow.start_run()
            artifact_path = "index"
            conda_env = tmp_path.joinpath("conda_env.yaml")
            _mlflow_conda_env(conda_env, additional_pip_deps=["llama_index"])
            index = single_index
            model_info = mlflow.llama_index.log_model(
                index=index,
                artifact_path=artifact_path,
                engine_type="query",
                engine_config={},
                conda_env=str(conda_env),
            )
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
            assert model_info.model_uri == model_uri
            # TODO: context manager doesn't work
            # reloaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)

    finally:
        mlflow.end_run()


# @pytest.mark.parametrize(
#     ("index_fixture", "should_start_run"),
#     [
#         ("mutli_index", True),
#         ("multi_index", False),
#     ],
# )
# def test_log_and_load_mutli_index(request, tmp_path, index_fixture, should_start_run):
#     try:
#         if should_start_run:
#             mlflow.start_run()
#         artifact_path = "index"
#         conda_env = tmp_path.joinpath("conda_env.yaml")
#         _mlflow_conda_env(conda_env, additional_pip_deps=["llama_index"])
#         index = request.getfixturevalue(index_fixture)
#         model_info = mlflow.llama_index.log_model(
#             index=index,
#             artifact_path=artifact_path,
#             conda_env=str(conda_env),
#         )
#         model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
#         assert model_info.model_uri == model_uri
#         reloaded_model = mlflow.llama_index.load_model(model_uri=model_uri)
#         assert (
#             reloaded_model.as_query_engine(temperature=0.0)
#             .query("Spell llamaindex")
#             .response.lower()
#             != ""
#         )
#         model_path = pathlib.Path(_download_artifact_from_uri(artifact_uri=model_uri))
#         model_config = Model.load(str(model_path.joinpath("MLmodel")))
#         assert mlflow.pyfunc.FLAVOR_NAME in model_config.flavors
#         assert mlflow.pyfunc.ENV in model_config.flavors[mlflow.pyfunc.FLAVOR_NAME]
#         env_path = model_config.flavors[mlflow.pyfunc.FLAVOR_NAME][mlflow.pyfunc.ENV]["conda"]
#         assert model_path.joinpath(env_path).exists()
#     finally:
#         mlflow.end_run()

# @pytest.mark.parametrize(
#     ("index_fixture", "should_start_run"),
#     [
#         ("single_graph", True),
#         ("single_graph", False),
#     ],
# )
# def test_log_and_load_graph(request, tmp_path, index_fixture, should_start_run):
#     try:
#         if should_start_run:
#             mlflow.start_run()
#         artifact_path = "index"
#         conda_env = tmp_path.joinpath("conda_env.yaml")
#         _mlflow_conda_env(conda_env, additional_pip_deps=["llama_index"])
#         index = request.getfixturevalue(index_fixture)
#         model_info = mlflow.llama_index.log_model(
#             index=index,
#             artifact_path=artifact_path,
#             conda_env=str(conda_env),
#         )
#         model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
#         assert model_info.model_uri == model_uri
#         reloaded_model = mlflow.llama_index.load_model(model_uri=model_uri)
#         assert (
#             reloaded_model.as_query_engine(temperature=0.0)
#             .query("Spell llamaindex")
#             .response.lower()
#             != ""
#         )
#         model_path = pathlib.Path(_download_artifact_from_uri(artifact_uri=model_uri))
#         model_config = Model.load(str(model_path.joinpath("MLmodel")))
#         assert mlflow.pyfunc.FLAVOR_NAME in model_config.flavors
#         assert mlflow.pyfunc.ENV in model_config.flavors[mlflow.pyfunc.FLAVOR_NAME]
#         env_path = model_config.flavors[mlflow.pyfunc.FLAVOR_NAME][mlflow.pyfunc.ENV]["conda"]
#         assert model_path.joinpath(env_path).exists()
#     finally:
#         mlflow.end_run()

# TODO: multi-index graph e2e
# TODO: prompt template
