import pathlib

import pytest

import mlflow
import mlflow.llama_index
import mlflow.pyfunc
from mlflow.models.model import Model
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env

from tests.llama_index._llama_index_test_fixtures import (  # noqa
    embed_model,
    llm,
    settings,
    single_index,
)


@pytest.mark.parametrize(
    ("index_fixture", "should_start_run"),
    [
        ("single_index", True),
        ("single_index", False),
    ],
)
def test_flavor_log_and_load_single_index(request, tmp_path, index_fixture, should_start_run):
    try:
        # with settings_context(do_embed=True, do_llm=True):
        if should_start_run:
            mlflow.start_run()
        artifact_path = "index"
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["llama_index"])
        index = request.getfixturevalue(index_fixture)
        model_info = mlflow.llama_index.log_model(
            index=index,
            artifact_path=artifact_path,
            engine_type="query",
            engine_kwargs={},
            conda_env=str(conda_env),
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        assert model_info.model_uri == model_uri
        reloaded_model = mlflow.llama_index.load_model(model_uri=model_uri)
        assert reloaded_model.as_query_engine().query("Spell llamaindex") != ""
        model_path = pathlib.Path(_download_artifact_from_uri(artifact_uri=model_uri))
        model_config = Model.load(str(model_path.joinpath("MLmodel")))
        assert mlflow.pyfunc.FLAVOR_NAME in model_config.flavors
        assert mlflow.pyfunc.ENV in model_config.flavors[mlflow.pyfunc.FLAVOR_NAME]
        env_path = model_config.flavors[mlflow.pyfunc.FLAVOR_NAME][mlflow.pyfunc.ENV]["conda"]
        assert model_path.joinpath(env_path).exists()

    finally:
        mlflow.end_run()


@pytest.mark.parametrize(
    ("index_fixture", "should_start_run"),
    [
        ("single_index", True),
        ("single_index", False),
    ],
)
def test_log_and_pyfunc_load_single_index(request, tmp_path, index_fixture, should_start_run):
    try:
        # with settings_context(do_embed=True, do_llm=True):
        if should_start_run:
            mlflow.start_run()
        artifact_path = "index"
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["llama_index"])
        index = request.getfixturevalue(index_fixture)
        model_info = mlflow.llama_index.log_model(
            index=index,
            artifact_path=artifact_path,
            engine_type="query",
            engine_kwargs={},
            conda_env=str(conda_env),
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        assert model_info.model_uri == model_uri
        reloaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)
        assert reloaded_model.predict("say hi") != ""
        model_path = pathlib.Path(_download_artifact_from_uri(artifact_uri=model_uri))
        model_config = Model.load(str(model_path.joinpath("MLmodel")))
        assert mlflow.pyfunc.FLAVOR_NAME in model_config.flavors
        assert mlflow.pyfunc.ENV in model_config.flavors[mlflow.pyfunc.FLAVOR_NAME]
        env_path = model_config.flavors[mlflow.pyfunc.FLAVOR_NAME][mlflow.pyfunc.ENV]["conda"]
        assert model_path.joinpath(env_path).exists()

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
