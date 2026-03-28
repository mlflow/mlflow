import json

import pytest

from mlflow.gemini_cli.config import (
    MLFLOW_TRACING_ENABLED,
    get_env_var,
    get_tracing_status,
    load_gemini_config,
    save_gemini_config,
    setup_environment_config,
)


@pytest.fixture
def temp_settings_path(tmp_path):
    """Provide a temporary settings.json path for tests."""
    return tmp_path / "settings.json"


def test_load_gemini_config_valid_json(temp_settings_path):
    config_data = {"hooks": {"SessionEnd": []}}
    with open(temp_settings_path, "w") as f:
        json.dump(config_data, f)

    result = load_gemini_config(temp_settings_path)
    assert result == config_data


def test_load_gemini_config_missing_file(tmp_path):
    non_existent_path = tmp_path / "non_existent.json"
    result = load_gemini_config(non_existent_path)
    assert result == {}


def test_load_gemini_config_invalid_json(temp_settings_path):
    with open(temp_settings_path, "w") as f:
        f.write("invalid json content")

    result = load_gemini_config(temp_settings_path)
    assert result == {}


def test_save_gemini_config_creates_file(temp_settings_path):
    config_data = {"test": "value"}
    save_gemini_config(temp_settings_path, config_data)

    assert temp_settings_path.exists()
    saved_data = json.loads(temp_settings_path.read_text())
    assert saved_data == config_data


def test_save_gemini_config_creates_directory(tmp_path):
    nested_path = tmp_path / "nested" / "dir" / "settings.json"
    config_data = {"test": "value"}

    save_gemini_config(nested_path, config_data)

    assert nested_path.exists()
    saved_data = json.loads(nested_path.read_text())
    assert saved_data == config_data


def test_get_env_var_from_os_environment(monkeypatch):
    monkeypatch.setenv(MLFLOW_TRACING_ENABLED, "test_os_value")
    result = get_env_var(MLFLOW_TRACING_ENABLED, "default")
    assert result == "test_os_value"


def test_get_env_var_default_when_not_found(monkeypatch):
    monkeypatch.delenv(MLFLOW_TRACING_ENABLED, raising=False)
    result = get_env_var(MLFLOW_TRACING_ENABLED, "default_value")
    assert result == "default_value"


def test_get_tracing_status_enabled(temp_settings_path):
    config_data = {
        "hooks": {
            "SessionEnd": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": 'python -c "from mlflow.gemini_cli.hooks import session_end_hook_handler; session_end_hook_handler()"',
                        }
                    ]
                }
            ]
        }
    }
    with open(temp_settings_path, "w") as f:
        json.dump(config_data, f)

    status = get_tracing_status(temp_settings_path)
    assert status.enabled is True


def test_get_tracing_status_no_hooks(temp_settings_path):
    config_data = {"hooks": {}}
    with open(temp_settings_path, "w") as f:
        json.dump(config_data, f)

    status = get_tracing_status(temp_settings_path)
    assert status.enabled is False


def test_get_tracing_status_no_config(tmp_path):
    non_existent_path = tmp_path / "missing.json"
    status = get_tracing_status(non_existent_path)
    assert status.enabled is False
    assert status.reason == "No configuration found"


def test_setup_environment_config_with_tracking_uri():
    env_vars = setup_environment_config(tracking_uri="http://localhost:5000")
    assert env_vars[MLFLOW_TRACING_ENABLED] == "true"
    assert env_vars["MLFLOW_TRACKING_URI"] == "http://localhost:5000"


def test_setup_environment_config_with_experiment_id():
    env_vars = setup_environment_config(experiment_id="123")
    assert env_vars["MLFLOW_EXPERIMENT_ID"] == "123"
    assert "MLFLOW_EXPERIMENT_NAME" not in env_vars


def test_setup_environment_config_with_experiment_name():
    env_vars = setup_environment_config(experiment_name="my-experiment")
    assert env_vars["MLFLOW_EXPERIMENT_NAME"] == "my-experiment"
    assert "MLFLOW_EXPERIMENT_ID" not in env_vars


def test_setup_environment_config_experiment_id_precedence():
    env_vars = setup_environment_config(
        experiment_id="123", experiment_name="my-experiment"
    )
    assert env_vars["MLFLOW_EXPERIMENT_ID"] == "123"
    assert "MLFLOW_EXPERIMENT_NAME" not in env_vars
