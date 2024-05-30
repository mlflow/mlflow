import pytest

import mlflow
import mlflow.llama_index
import mlflow.pyfunc

from tests.llama_index._llama_index_test_fixtures import (
    embed_model,  # noqa: F401
    llm,  # noqa: F401
    multi_index,  # noqa: F401
    settings,  # noqa: F401
    single_graph,  # noqa: F401
    single_index,  # noqa: F401
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
    assert loaded_model.as_chat_engine().query("Spell llamaindex").response.lower() != ""


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
    assert loaded_model.as_chat_engine().query("Spell llamaindex").response.lower() != ""
