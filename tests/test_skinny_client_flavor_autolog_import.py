# Regression test for https://github.com/mlflow/mlflow/issues/27005:
# mlflow-skinny users couldn't use mlflow.openai.autolog() because importing the flavor
# eagerly imported its model logging APIs, which depend on numpy and pandas.

import importlib
import importlib.util
import os

import pytest


@pytest.fixture(autouse=True)
def is_skinny():
    if "MLFLOW_SKINNY" not in os.environ:
        pytest.skip("This test is only valid for the skinny client")


@pytest.mark.parametrize("flavor", ["openai", "langchain", "llama_index", "dspy"])
def test_flavor_autolog_importable_without_numpy_and_pandas(flavor):
    # Verify numpy and pandas are genuinely not installed (not just not yet imported)
    assert importlib.util.find_spec("numpy") is None
    assert importlib.util.find_spec("pandas") is None

    module = importlib.import_module(f"mlflow.{flavor}")
    assert callable(module.autolog)
    assert not hasattr(module, "log_model")
