import json

import pytest

from mlflow.kiro_cli.config import (
    MLFLOW_TRACING_ENABLED,
    get_env_var,
    get_tracing_status,
    is_tracing_enabled,
    load_kiro_config,
    save_kiro_config,
    setup_environment_config,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_settings_path(tmp_path):
    """Provide a temporary settings.json path for tests."""
    return tmp_path / "settings.json"


# ---------------------------------------------------------------------------
# load_kiro_config
# ---------------------------------------------------------------------------


def test_load_kiro_config_valid_json(temp_settings_path):
    config_data = {"env": {"MLFLOW_KIRO_CLI_TRACING_ENABLED": "true"}}
    with open(temp_settings_path, "w") as f:
        json.dump(config_data, f)

    result = load_kiro_config(temp_settings_path)
    assert result == config_data


def test_load_kiro_config_missing_file(tmp_path):
    non_existent_path = tmp_path / "non_existent.json"
    result = load_kiro_config(non_existent_path)
    assert result == {}


def test_load_kiro_config_invalid_json(temp_settings_path):
    with open(temp_settings_path, "w") as f:
        f.write("invalid json content")

    result = load_kiro_config(temp_settings_path)
    assert result == {}


# ---------------------------------------------------------------------------
# save_kiro_config
# ---------------------------------------------------------------------------


def test_save_kiro_config_creates_file(temp_settings_path):
    config_data = {"test": "value"}
    save_kiro_config(temp_settings_path, config_data)

    assert temp_settings_path.exists()
    saved_data = json.loads(temp_settings_path.read_text())
    assert saved_data == config_data


def test_save_kiro_config_creates_directory(tmp_path):
    nested_path = tmp_path / "nested" / "dir" / "settings.json"
    config_data = {"test": "value"}

    save_kiro_config(nested_path, config_data)

    assert nested_path.exists()
    saved_data = json.loads(nested_path.read_text())
    assert saved_data == config_data


# ---------------------------------------------------------------------------
# get_env_var
# ---------------------------------------------------------------------------


def test_get_env_var_from_os_environment_when_no_settings(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_TRACING_ENABLED, "test_os_value")
    monkeypatch.chdir(tmp_path)

    result = get_env_var(MLFLOW_TRACING_ENABLED, "default")
    assert result == "test_os_value"


def test_get_env_var_settings_takes_precedence_over_os_env(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_TRACING_ENABLED, "os_value")

    config_data = {"env": {MLFLOW_TRACING_ENABLED: "settings_value"}}
    kiro_settings_path = tmp_path / ".kiro" / "settings.json"
    kiro_settings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(kiro_settings_path, "w") as f:
        json.dump(config_data, f)

    monkeypatch.chdir(tmp_path)
    result = get_env_var(MLFLOW_TRACING_ENABLED, "default")
    assert result == "settings_value"


def test_get_env_var_default_when_not_found(tmp_path, monkeypatch):
    monkeypatch.delenv(MLFLOW_TRACING_ENABLED, raising=False)

    kiro_settings_path = tmp_path / ".kiro" / "settings.json"
    kiro_settings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(kiro_settings_path, "w") as f:
        json.dump({}, f)

    monkeypatch.chdir(tmp_path)
    result = get_env_var(MLFLOW_TRACING_ENABLED, "default_value")
    assert result == "default_value"


def test_get_env_var_graceful_fallthrough_unparseable_settings(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_TRACING_ENABLED, "os_fallback")

    kiro_settings_path = tmp_path / ".kiro" / "settings.json"
    kiro_settings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(kiro_settings_path, "w") as f:
        f.write("not valid json!!!")

    monkeypatch.chdir(tmp_path)
    result = get_env_var(MLFLOW_TRACING_ENABLED, "default")
    assert result == "os_fallback"


# ---------------------------------------------------------------------------
# get_tracing_status
# ---------------------------------------------------------------------------


def test_get_tracing_status_enabled(temp_settings_path):
    config_data = {
        "env": {
            MLFLOW_TRACING_ENABLED: "true",
            "MLFLOW_TRACKING_URI": "sqlite:///mlflow.db",
            "MLFLOW_EXPERIMENT_ID": "42",
            "MLFLOW_EXPERIMENT_NAME": "my-experiment",
        }
    }
    with open(temp_settings_path, "w") as f:
        json.dump(config_data, f)

    status = get_tracing_status(temp_settings_path)
    assert status.enabled is True
    assert status.tracking_uri == "sqlite:///mlflow.db"
    assert status.experiment_id == "42"
    assert status.experiment_name == "my-experiment"


def test_get_tracing_status_disabled(temp_settings_path):
    config_data = {"env": {MLFLOW_TRACING_ENABLED: "false"}}
    with open(temp_settings_path, "w") as f:
        json.dump(config_data, f)

    status = get_tracing_status(temp_settings_path)
    assert status.enabled is False


def test_get_tracing_status_no_config(tmp_path):
    non_existent_path = tmp_path / "missing.json"
    status = get_tracing_status(non_existent_path)
    assert status.enabled is False
    assert status.reason == "No configuration found"


# ---------------------------------------------------------------------------
# setup_environment_config
# ---------------------------------------------------------------------------


def test_setup_environment_config_new_file(temp_settings_path):
    tracking_uri = "test://localhost"
    experiment_id = "123"

    setup_environment_config(temp_settings_path, tracking_uri, experiment_id)

    assert temp_settings_path.exists()
    config = json.loads(temp_settings_path.read_text())
    env_vars = config["env"]
    assert env_vars[MLFLOW_TRACING_ENABLED] == "true"
    assert env_vars["MLFLOW_TRACKING_URI"] == tracking_uri
    assert env_vars["MLFLOW_EXPERIMENT_ID"] == experiment_id


def test_setup_environment_config_id_beats_name(temp_settings_path):
    """When both experiment_id and experiment_name are supplied, ID wins and name is removed."""
    existing_config = {
        "env": {
            MLFLOW_TRACING_ENABLED: "true",
            "MLFLOW_EXPERIMENT_NAME": "old_name",
        }
    }
    with open(temp_settings_path, "w") as f:
        json.dump(existing_config, f)

    setup_environment_config(
        temp_settings_path,
        tracking_uri="uri",
        experiment_id="new_id",
        experiment_name="new_name",
    )

    config = json.loads(temp_settings_path.read_text())
    env_vars = config["env"]
    assert env_vars["MLFLOW_EXPERIMENT_ID"] == "new_id"
    assert "MLFLOW_EXPERIMENT_NAME" not in env_vars


def test_setup_environment_config_name_only(temp_settings_path):
    """When only experiment_name is supplied, name is stored and ID is removed."""
    existing_config = {
        "env": {
            MLFLOW_TRACING_ENABLED: "true",
            "MLFLOW_EXPERIMENT_ID": "old_id",
        }
    }
    with open(temp_settings_path, "w") as f:
        json.dump(existing_config, f)

    setup_environment_config(
        temp_settings_path,
        experiment_name="my_experiment",
    )

    config = json.loads(temp_settings_path.read_text())
    env_vars = config["env"]
    assert env_vars["MLFLOW_EXPERIMENT_NAME"] == "my_experiment"
    assert "MLFLOW_EXPERIMENT_ID" not in env_vars


def test_setup_environment_config_preserves_unrelated_env_keys(temp_settings_path):
    existing_config = {
        "env": {
            "MY_CUSTOM_VAR": "custom_value",
            "ANOTHER_VAR": "another_value",
        }
    }
    with open(temp_settings_path, "w") as f:
        json.dump(existing_config, f)

    setup_environment_config(temp_settings_path, tracking_uri="uri")

    config = json.loads(temp_settings_path.read_text())
    env_vars = config["env"]
    assert env_vars["MY_CUSTOM_VAR"] == "custom_value"
    assert env_vars["ANOTHER_VAR"] == "another_value"
    assert env_vars[MLFLOW_TRACING_ENABLED] == "true"


def test_setup_environment_config_preserves_unrelated_top_level_keys(temp_settings_path):
    existing_config = {
        "some_user_setting": "value",
        "another_setting": [1, 2, 3],
    }
    with open(temp_settings_path, "w") as f:
        json.dump(existing_config, f)

    setup_environment_config(temp_settings_path, tracking_uri="uri")

    config = json.loads(temp_settings_path.read_text())
    assert config["some_user_setting"] == "value"
    assert config["another_setting"] == [1, 2, 3]
    assert config["env"][MLFLOW_TRACING_ENABLED] == "true"


# ---------------------------------------------------------------------------
# is_tracing_enabled
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes", "Yes", "YES"])
def test_is_tracing_enabled_truthy_values(tmp_path, monkeypatch, value):
    monkeypatch.setenv(MLFLOW_TRACING_ENABLED, value)
    monkeypatch.chdir(tmp_path)
    assert is_tracing_enabled() is True


@pytest.mark.parametrize("value", ["false", "False", "0", "no", "No", "", "random", "nope", "tru"])
def test_is_tracing_enabled_falsy_values(tmp_path, monkeypatch, value):
    monkeypatch.setenv(MLFLOW_TRACING_ENABLED, value)
    monkeypatch.chdir(tmp_path)
    assert is_tracing_enabled() is False
