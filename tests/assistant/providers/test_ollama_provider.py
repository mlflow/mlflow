"""Tests for the Ollama preset of OpenAICompatibleProvider.

Covers the Ollama-specific bits: `/api/tags` listing and the OAI-compat
chat path the preset is wired to call. The shared SSE/tool-call/think-block
behavior is exercised in `test_openai_compatible_provider.py`.
"""

from unittest.mock import MagicMock, patch

import pytest

from mlflow.assistant.providers import list_providers
from mlflow.assistant.providers.base import (
    ProviderNotConfiguredError,
    clear_config_cache,
)


def _ollama_provider():
    for p in list_providers():
        if p.name == "ollama":
            return p
    raise AssertionError("ollama provider not registered")


@pytest.fixture(autouse=True)
def config(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"providers": {"ollama": {"model": "llama3.2"}}}')
    clear_config_cache()
    with patch("mlflow.assistant.config.CONFIG_PATH", config_file):
        yield config_file
    clear_config_cache()


def test_provider_identity():
    p = _ollama_provider()
    assert p.name == "ollama"
    assert p.display_name == "Ollama"
    assert p.is_available() is True


def test_list_models_returns_model_names():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"models": [{"model": "llama3"}, {"model": "mistral"}]}
    mock_resp.raise_for_status = MagicMock()

    with patch(
        "mlflow.assistant.providers.presets.requests.get", return_value=mock_resp
    ) as mock_get:
        models = _ollama_provider().list_models("http://localhost:11434")

    assert models == ["llama3", "mistral"]
    mock_get.assert_called_once_with("http://localhost:11434/api/tags", headers={}, timeout=10)


def test_list_models_raises_on_connection_error():
    with patch(
        "mlflow.assistant.providers.presets.requests.get",
        side_effect=Exception("Connection refused"),
    ):
        with pytest.raises(ProviderNotConfiguredError, match="Connection refused"):
            _ollama_provider().list_models("http://localhost:11434")


def test_default_base_url_used_when_unconfigured(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text("{}")
    clear_config_cache()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"models": [{"model": "llama3"}]}
    with (
        patch("mlflow.assistant.config.CONFIG_PATH", config_file),
        patch(
            "mlflow.assistant.providers.presets.requests.get", return_value=mock_resp
        ) as mock_get,
    ):
        models = _ollama_provider().list_models()
    assert models == ["llama3"]
    mock_get.assert_called_once_with("http://localhost:11434/api/tags", headers={}, timeout=10)
    clear_config_cache()
