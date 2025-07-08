import json
import warnings
from contextlib import contextmanager

import pytest

import mlflow
from mlflow.telemetry import get_telemetry_client


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


def test_register_prompt_sends_telemetry_record(mock_requests):
    """Test that register_prompt sends telemetry records."""
    mlflow.genai.register_prompt("test_prompt", "test template {{var}}")
    get_telemetry_client().flush()

    assert len(mock_requests) == 1
    record = mock_requests[0]
    data = json.loads(record["data"])
    assert data["api_module"] == mlflow.genai.register_prompt.__module__
    assert data["api_name"] == "register_prompt"
    assert data["params"] is None
    assert data["status"] == "success"


def test_load_prompt_sends_telemetry_record(mock_requests):
    """Test that load_prompt sends telemetry records."""
    mlflow.genai.register_prompt("test_prompt_load", "test template")
    mlflow.genai.load_prompt("prompts:/test_prompt_load/1")
    get_telemetry_client().flush()

    # Two records: one for register, one for load
    assert len(mock_requests) == 2
    # Check the load_prompt record (second one)
    record = mock_requests[1]
    data = json.loads(record["data"])
    assert data["api_module"] == mlflow.genai.load_prompt.__module__
    assert data["api_name"] == "load_prompt"
    assert data["params"] is None
    assert data["status"] == "success"
