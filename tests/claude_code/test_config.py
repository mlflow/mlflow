"""Tests for mlflow.claude_code.config module."""

import json

import pytest

from mlflow.claude_code.config import (
    MLFLOW_TRACING_ENABLED,
    get_env_var,
    get_tracing_status,
    load_claude_config,
    save_claude_config,
    setup_environment_config,
)


@pytest.fixture
def temp_settings_path(tmp_path):
    """Provide a temporary settings.json path for tests."""
    return tmp_path / "settings.json"


def test_load_claude_config_valid_json(temp_settings_path):
    """Test loading a valid Claude configuration file."""
    config_data = {"tools": {"computer_20241022": {"name": "computer"}}}
    with open(temp_settings_path, "w") as f:
        json.dump(config_data, f)

    result = load_claude_config(temp_settings_path)
    assert result == config_data


def test_load_claude_config_missing_file(tmp_path):
    """Test loading configuration when file doesn't exist returns empty dict."""
    non_existent_path = tmp_path / "non_existent.json"
    result = load_claude_config(non_existent_path)
    assert result == {}


def test_load_claude_config_invalid_json(temp_settings_path):
    """Test loading configuration with invalid JSON returns empty dict."""
    with open(temp_settings_path, "w") as f:
        f.write("invalid json content")

    result = load_claude_config(temp_settings_path)
    assert result == {}


def test_save_claude_config_creates_file(temp_settings_path):
    """Test that save_claude_config creates a new configuration file."""
    config_data = {"test": "value"}
    save_claude_config(temp_settings_path, config_data)

    assert temp_settings_path.exists()
    saved_data = json.loads(temp_settings_path.read_text())
    assert saved_data == config_data


def test_save_claude_config_creates_directory(tmp_path):
    """Test that save_claude_config creates parent directories if they don't exist."""
    nested_path = tmp_path / "nested" / "dir" / "settings.json"
    config_data = {"test": "value"}

    save_claude_config(nested_path, config_data)

    assert nested_path.exists()
    saved_data = json.loads(nested_path.read_text())
    assert saved_data == config_data


def test_get_env_var_from_os_environment(monkeypatch):
    """Test get_env_var returns value from OS environment when available."""
    test_value = "test_os_value"
    monkeypatch.setenv(MLFLOW_TRACING_ENABLED, test_value)

    result = get_env_var(MLFLOW_TRACING_ENABLED, "default")
    assert result == test_value


def test_get_env_var_from_claude_settings_fallback(tmp_path, monkeypatch):
    """Test get_env_var falls back to Claude settings when OS env var not set."""
    # Ensure OS env var is not set
    monkeypatch.delenv(MLFLOW_TRACING_ENABLED, raising=False)

    # Create settings file with environment variable
    config_data = {"environment": {MLFLOW_TRACING_ENABLED: "claude_value"}}
    claude_settings_path = tmp_path / ".claude" / "settings.json"
    claude_settings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(claude_settings_path, "w") as f:
        json.dump(config_data, f)

    # Change to the temp directory so .claude/settings.json is found
    monkeypatch.chdir(tmp_path)
    result = get_env_var(MLFLOW_TRACING_ENABLED, "default")
    assert result == "claude_value"


def test_get_env_var_default_when_not_found(tmp_path, monkeypatch):
    """Test get_env_var returns default when variable not found anywhere."""
    # Ensure OS env var is not set
    monkeypatch.delenv(MLFLOW_TRACING_ENABLED, raising=False)

    # Create empty settings file in .claude directory
    claude_settings_path = tmp_path / ".claude" / "settings.json"
    claude_settings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(claude_settings_path, "w") as f:
        json.dump({}, f)

    # Change to temp directory so .claude/settings.json is found
    monkeypatch.chdir(tmp_path)
    result = get_env_var(MLFLOW_TRACING_ENABLED, "default_value")
    assert result == "default_value"


def test_get_tracing_status_enabled(temp_settings_path):
    """Test get_tracing_status returns enabled status when tracing is enabled."""
    # Create settings with tracing enabled
    config_data = {"environment": {MLFLOW_TRACING_ENABLED: "true"}}
    with open(temp_settings_path, "w") as f:
        json.dump(config_data, f)

    status = get_tracing_status(temp_settings_path)
    assert status.enabled is True
    assert hasattr(status, "tracking_uri")


def test_get_tracing_status_disabled(temp_settings_path):
    """Test get_tracing_status returns disabled status when tracing is disabled."""
    # Create settings with tracing disabled
    config_data = {"environment": {MLFLOW_TRACING_ENABLED: "false"}}
    with open(temp_settings_path, "w") as f:
        json.dump(config_data, f)

    status = get_tracing_status(temp_settings_path)
    assert status.enabled is False


def test_get_tracing_status_no_config(tmp_path):
    """Test get_tracing_status returns disabled when no configuration exists."""
    non_existent_path = tmp_path / "missing.json"
    status = get_tracing_status(non_existent_path)
    assert status.enabled is False
    assert status.reason == "No configuration found"


def test_setup_environment_config_new_file(temp_settings_path):
    """Test setup_environment_config creates new configuration file."""
    tracking_uri = "test://localhost"
    experiment_id = "123"

    setup_environment_config(temp_settings_path, tracking_uri, experiment_id)

    # Verify file was created
    assert temp_settings_path.exists()

    # Verify configuration contents
    config = json.loads(temp_settings_path.read_text())

    env_vars = config["environment"]
    assert env_vars[MLFLOW_TRACING_ENABLED] == "true"
    assert env_vars["MLFLOW_TRACKING_URI"] == tracking_uri
    assert env_vars["MLFLOW_EXPERIMENT_ID"] == experiment_id


def test_setup_environment_config_experiment_id_precedence(temp_settings_path):
    """Test that experiment_id parameter takes precedence over existing config."""
    # Create existing config with different experiment ID
    existing_config = {
        "environment": {
            MLFLOW_TRACING_ENABLED: "true",
            "MLFLOW_EXPERIMENT_ID": "old_id",
            "MLFLOW_TRACKING_URI": "old_uri",
        }
    }
    with open(temp_settings_path, "w") as f:
        json.dump(existing_config, f)

    new_tracking_uri = "new://localhost"
    new_experiment_id = "new_id"

    setup_environment_config(temp_settings_path, new_tracking_uri, new_experiment_id)

    # Verify configuration was updated
    config = json.loads(temp_settings_path.read_text())

    env_vars = config["environment"]
    assert env_vars[MLFLOW_TRACING_ENABLED] == "true"
    assert env_vars["MLFLOW_TRACKING_URI"] == new_tracking_uri
    assert env_vars["MLFLOW_EXPERIMENT_ID"] == new_experiment_id
