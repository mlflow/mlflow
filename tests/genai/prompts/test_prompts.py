import warnings
from contextlib import contextmanager

import pytest

import mlflow


@contextmanager
def no_future_warning():
    with warnings.catch_warnings():
        # Translate future warning into an exception
        warnings.simplefilter("error", FutureWarning)
        yield


def test_suppress_prompt_api_migration_warning():
    with no_future_warning():
        mlflow.genai.register_prompt("test_prompt", "test_template")
        mlflow.genai.search_prompts()
        mlflow.genai.load_prompt("prompts:/test_prompt/1")
        mlflow.genai.set_prompt_alias("test_prompt", "test_alias", 1)
        mlflow.genai.delete_prompt_alias("test_prompt", "test_alias")


def test_prompt_api_migration_warning():
    with pytest.warns(FutureWarning, match="The `mlflow.register_prompt` API is"):
        mlflow.register_prompt("test_prompt", "test_template")

    with pytest.warns(FutureWarning, match="The `mlflow.search_prompts` API is"):
        mlflow.search_prompts()

    with pytest.warns(FutureWarning, match="The `mlflow.load_prompt` API is"):
        mlflow.load_prompt("prompts:/test_prompt/1")

    with pytest.warns(FutureWarning, match="The `mlflow.set_prompt_alias` API is"):
        mlflow.set_prompt_alias("test_prompt", "test_alias", 1)

    with pytest.warns(FutureWarning, match="The `mlflow.delete_prompt_alias` API is"):
        mlflow.delete_prompt_alias("test_prompt", "test_alias")
