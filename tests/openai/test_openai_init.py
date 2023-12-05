import os
from importlib import reload

from mlflow.openai import _OAITokenHolder


def test_set_api_key_on_tokenholder_init(monkeypatch):
    # if the user sets the API key after the openai module,
    # expect `openai.api_key` to not be set.
    monkeypatch.delenv("OPENAI_API_KEY", False)
    assert "OPENAI_API_KEY" not in os.environ

    import openai

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert openai.api_key is None

    # when OAITokenHolder is initialized, expect it to set `openai.api_key`
    token_holder = _OAITokenHolder("open_ai")
    assert openai.api_key == "test-key"
    assert token_holder._key_configured

    # reload the module to simulate the env var being set before
    # load. in this case we'd expect the API key to be present
    reload(openai)
    assert openai.api_key == "test-key"
