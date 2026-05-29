from unittest.mock import patch

import pytest

from mlflow.assistant.config import AssistantConfig, PermissionsConfig
from mlflow.assistant.providers.base import clear_config_cache


@pytest.fixture(autouse=True)
def config_file(tmp_path):
    config_path = tmp_path / "config.json"
    clear_config_cache()
    with patch("mlflow.assistant.config.CONFIG_PATH", config_path):
        yield config_path
    clear_config_cache()


def test_update_provider_preserves_selection():
    config = AssistantConfig()
    config.set_provider("claude_code", "claude-4", base_url=None)
    assert config.providers["claude_code"].selected is True

    config.update_provider("ollama", model="llama3.2", base_url="http://localhost:11434")

    assert config.providers["claude_code"].selected is True
    assert config.providers["ollama"].selected is False
    assert config.providers["ollama"].model == "llama3.2"
    assert config.providers["ollama"].base_url == "http://localhost:11434"


def test_update_provider_creates_entry_when_missing():
    config = AssistantConfig()
    assert "ollama" not in config.providers

    config.update_provider("ollama", model="llama3.2", base_url="http://localhost:11434")

    assert "ollama" in config.providers
    assert config.providers["ollama"].model == "llama3.2"
    assert config.providers["ollama"].selected is False


def test_update_provider_does_not_deselect_existing():
    config = AssistantConfig()
    config.set_provider("ollama", "llama3.2")
    assert config.providers["ollama"].selected is True

    config.update_provider("ollama", base_url="http://newhost:11434")

    assert config.providers["ollama"].selected is True
    assert config.providers["ollama"].base_url == "http://newhost:11434"


def test_set_provider_deselects_others():
    config = AssistantConfig()
    config.set_provider("claude_code", "claude-4")
    config.update_provider("ollama", model="llama3.2")

    config.set_provider("ollama", "llama3.2")

    assert config.providers["claude_code"].selected is False
    assert config.providers["ollama"].selected is True


def test_get_selected_provider_returns_none_when_empty():
    config = AssistantConfig()
    assert config.get_selected_provider() is None


def test_config_round_trip(config_file):
    config = AssistantConfig()
    config.set_provider("ollama", "llama3.2", base_url="http://localhost:11434")
    config.save()

    loaded = AssistantConfig.load()
    assert loaded.providers["ollama"].model == "llama3.2"
    assert loaded.providers["ollama"].selected is True
    assert loaded.providers["ollama"].base_url == "http://localhost:11434"


def test_update_provider_preserves_permissions():
    config = AssistantConfig()
    config.set_provider("ollama", "llama3.2")
    new_perms = PermissionsConfig(allow_edit_files=False, full_access=False)

    config.update_provider("ollama", permissions=new_perms)

    assert config.providers["ollama"].permissions.allow_edit_files is False
