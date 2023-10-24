import os

from mlflow.openai import _OAITokenHolder


def test_set_api_key_on_tokenholder_init(monkeypatch):
    # if the user sets the API key after the openai module,
    # expect `openai.api_key` to not be set.
    assert "OPENAI_API_KEY" not in os.environ
    import openai

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert openai.api_key is None

    # when OAITokenHolder is initialized, expect it to set `openai.api_key`
    _OAITokenHolder("open_ai")
    assert openai.api_key == "test-key"
