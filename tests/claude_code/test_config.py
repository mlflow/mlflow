"""Tests for mlflow.claude_code.config module."""

import json
import os
from pathlib import Path

import pytest

from mlflow.claude_code.config import (
    MLFLOW_TRACING_ENABLED,
    get_env_var,
    get_tracing_status,
    load_claude_config,
    save_claude_config,
    setup_environment_config,
)


class TestClaudeConfig:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        self.temp_dir = tmp_path
        self.settings_path = tmp_path / "settings.json"
        self.monkeypatch = monkeypatch

    def test_load_claude_config_valid_json(self):
        """Test loading a valid Claude configuration file."""
        config_data = {"tools": {"computer_20241022": {"name": "computer"}}}
        with open(self.settings_path, "w") as f:
            json.dump(config_data, f)

        result = load_claude_config(self.settings_path)
        assert result == config_data

    def test_load_claude_config_missing_file(self):
        """Test loading configuration when file doesn't exist returns empty dict."""
        non_existent_path = Path(self.temp_dir) / "non_existent.json"
        result = load_claude_config(non_existent_path)
        assert result == {}

    def test_load_claude_config_invalid_json(self):
        """Test loading configuration with invalid JSON returns empty dict."""
        with open(self.settings_path, "w") as f:
            f.write("invalid json content")

        result = load_claude_config(self.settings_path)
        assert result == {}

    def test_save_claude_config_creates_file(self):
        """Test that save_claude_config creates a new configuration file."""
        config_data = {"test": "value"}
        save_claude_config(self.settings_path, config_data)

        assert self.settings_path.exists()
        with open(self.settings_path) as f:
            saved_data = json.load(f)
        assert saved_data == config_data

    def test_save_claude_config_creates_directory(self):
        """Test that save_claude_config creates parent directories if they don't exist."""
        nested_path = Path(self.temp_dir) / "nested" / "dir" / "settings.json"
        config_data = {"test": "value"}

        save_claude_config(nested_path, config_data)

        assert nested_path.exists()
        with open(nested_path) as f:
            saved_data = json.load(f)
        assert saved_data == config_data

    def test_get_env_var_from_os_environment(self):
        """Test get_env_var returns value from OS environment when available."""
        test_value = "test_os_value"
        self.monkeypatch.setenv(MLFLOW_TRACING_ENABLED, test_value)

        result = get_env_var(MLFLOW_TRACING_ENABLED, "default")
        assert result == test_value

    def test_get_env_var_from_claude_settings_fallback(self):
        """Test get_env_var falls back to Claude settings when OS env var not set."""
        # Ensure OS env var is not set
        self.monkeypatch.delenv(MLFLOW_TRACING_ENABLED, raising=False)

        # Create settings file with environment variable
        config_data = {"environment": {MLFLOW_TRACING_ENABLED: "claude_value"}}
        claude_settings_path = self.temp_dir / ".claude" / "settings.json"
        claude_settings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(claude_settings_path, "w") as f:
            json.dump(config_data, f)

        # Change to the temp directory so .claude/settings.json is found
        original_cwd = os.getcwd()
        try:
            os.chdir(str(self.temp_dir))
            result = get_env_var(MLFLOW_TRACING_ENABLED, "default")
            assert result == "claude_value"
        finally:
            os.chdir(original_cwd)

    def test_get_env_var_default_when_not_found(self):
        """Test get_env_var returns default when variable not found anywhere."""
        # Ensure OS env var is not set
        self.monkeypatch.delenv(MLFLOW_TRACING_ENABLED, raising=False)

        # Create empty settings file in .claude directory
        claude_settings_path = self.temp_dir / ".claude" / "settings.json"
        claude_settings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(claude_settings_path, "w") as f:
            json.dump({}, f)

        # Change to temp directory so .claude/settings.json is found
        original_cwd = os.getcwd()
        try:
            os.chdir(str(self.temp_dir))
            result = get_env_var(MLFLOW_TRACING_ENABLED, "default_value")
            assert result == "default_value"
        finally:
            os.chdir(original_cwd)

    def test_get_tracing_status_enabled(self):
        """Test get_tracing_status returns enabled status when tracing is enabled."""
        # Create settings with tracing enabled
        config_data = {"environment": {MLFLOW_TRACING_ENABLED: "true"}}
        with open(self.settings_path, "w") as f:
            json.dump(config_data, f)

        status = get_tracing_status(self.settings_path)
        assert status["enabled"] is True
        assert "tracking_uri" in status

    def test_get_tracing_status_disabled(self):
        """Test get_tracing_status returns disabled status when tracing is disabled."""
        # Create settings with tracing disabled
        config_data = {"environment": {MLFLOW_TRACING_ENABLED: "false"}}
        with open(self.settings_path, "w") as f:
            json.dump(config_data, f)

        status = get_tracing_status(self.settings_path)
        assert status["enabled"] is False

    def test_get_tracing_status_no_config(self):
        """Test get_tracing_status returns disabled when no configuration exists."""
        non_existent_path = Path(self.temp_dir) / "missing.json"
        status = get_tracing_status(non_existent_path)
        assert status["enabled"] is False
        assert status["reason"] == "No configuration found"

    def test_setup_environment_config_new_file(self):
        """Test setup_environment_config creates new configuration file."""
        tracking_uri = "test://localhost"
        experiment_id = "123"

        setup_environment_config(self.settings_path, tracking_uri, experiment_id)

        # Verify file was created
        assert self.settings_path.exists()

        # Verify configuration contents
        with open(self.settings_path) as f:
            config = json.load(f)

        env_vars = config["environment"]
        assert env_vars[MLFLOW_TRACING_ENABLED] == "true"
        assert env_vars["MLFLOW_TRACKING_URI"] == tracking_uri
        assert env_vars["MLFLOW_EXPERIMENT_ID"] == experiment_id

    def test_setup_environment_config_experiment_id_precedence(self):
        """Test that experiment_id parameter takes precedence over existing config."""
        # Create existing config with different experiment ID
        existing_config = {
            "environment": {
                MLFLOW_TRACING_ENABLED: "true",
                "MLFLOW_EXPERIMENT_ID": "old_id",
                "MLFLOW_TRACKING_URI": "old_uri",
            }
        }
        with open(self.settings_path, "w") as f:
            json.dump(existing_config, f)

        new_tracking_uri = "new://localhost"
        new_experiment_id = "new_id"

        setup_environment_config(self.settings_path, new_tracking_uri, new_experiment_id)

        # Verify configuration was updated
        with open(self.settings_path) as f:
            config = json.load(f)

        env_vars = config["environment"]
        assert env_vars[MLFLOW_TRACING_ENABLED] == "true"
        assert env_vars["MLFLOW_TRACKING_URI"] == new_tracking_uri
        assert env_vars["MLFLOW_EXPERIMENT_ID"] == new_experiment_id
