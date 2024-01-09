import importlib

import openai
import pytest


@pytest.fixture(autouse=True)
def set_envs(monkeypatch):
    monkeypatch.setenvs(
        {
            "MLFLOW_TESTING": "true",
            "OPENAI_API_KEY": "test",
            "SERPAPI_API_KEY": "test",
        }
    )
    importlib.reload(openai)


@pytest.fixture(scope="module")
def model_path(tmp_path):
    return tmp_path / "model"
