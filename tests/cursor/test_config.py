import json

import pytest

from mlflow.cursor.config import (
    CURSOR_DIR,
    MLFLOW_CURSOR_TRACING_ENABLED,
    get_cursor_hooks_path,
    get_env_var,
    get_tracing_status,
    is_tracing_enabled,
    load_cursor_config,
    save_cursor_config,
    setup_environment_config,
)


@pytest.fixture
def temp_hooks_path(tmp_path):
    cursor_dir = tmp_path / CURSOR_DIR
    cursor_dir.mkdir(parents=True, exist_ok=True)
    return cursor_dir / "hooks.json"


def test_get_cursor_hooks_path(tmp_path):
    result = get_cursor_hooks_path(tmp_path)
    assert result == tmp_path / CURSOR_DIR / "hooks.json"


def test_load_cursor_config_valid_json(temp_hooks_path):
    config_data = {"version": 1, "hooks": {"stop": []}}
    with open(temp_hooks_path, "w") as f:
        json.dump(config_data, f)

    result = load_cursor_config(temp_hooks_path)
    assert result == config_data


def test_load_cursor_config_missing_file(tmp_path):
    non_existent_path = tmp_path / "non_existent.json"
    result = load_cursor_config(non_existent_path)
    assert result == {"version": 1, "hooks": {}}


def test_load_cursor_config_invalid_json(temp_hooks_path):
    with open(temp_hooks_path, "w") as f:
        f.write("invalid json content")

    result = load_cursor_config(temp_hooks_path)
    assert result == {"version": 1, "hooks": {}}


def test_save_cursor_config_creates_file(temp_hooks_path):
    config_data = {"version": 1, "hooks": {"test": []}}
    save_cursor_config(temp_hooks_path, config_data)

    assert temp_hooks_path.exists()
    saved_data = json.loads(temp_hooks_path.read_text())
    assert saved_data == config_data


def test_save_cursor_config_creates_directory(tmp_path):
    nested_path = tmp_path / "nested" / "dir" / "hooks.json"
    config_data = {"version": 1, "hooks": {}}

    save_cursor_config(nested_path, config_data)

    assert nested_path.exists()
    saved_data = json.loads(nested_path.read_text())
    assert saved_data == config_data


def test_get_env_var_from_os_environment_when_no_env_file(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_CURSOR_TRACING_ENABLED, "test_os_value")
    monkeypatch.chdir(tmp_path)

    result = get_env_var(MLFLOW_CURSOR_TRACING_ENABLED, "default")
    assert result == "test_os_value"


def test_get_env_var_env_file_takes_precedence_over_os_env(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_CURSOR_TRACING_ENABLED, "os_value")

    cursor_dir = tmp_path / CURSOR_DIR
    cursor_dir.mkdir(parents=True, exist_ok=True)
    env_file = cursor_dir / ".env"
    env_file.write_text(f"{MLFLOW_CURSOR_TRACING_ENABLED}=env_file_value\n")

    monkeypatch.chdir(tmp_path)
    result = get_env_var(MLFLOW_CURSOR_TRACING_ENABLED, "default")
    assert result == "env_file_value"


def test_get_env_var_default_when_not_found(tmp_path, monkeypatch):
    monkeypatch.delenv(MLFLOW_CURSOR_TRACING_ENABLED, raising=False)
    monkeypatch.chdir(tmp_path)

    result = get_env_var(MLFLOW_CURSOR_TRACING_ENABLED, "default_value")
    assert result == "default_value"


def test_is_tracing_enabled_true(tmp_path, monkeypatch):
    cursor_dir = tmp_path / CURSOR_DIR
    cursor_dir.mkdir(parents=True, exist_ok=True)
    env_file = cursor_dir / ".env"
    env_file.write_text(f"{MLFLOW_CURSOR_TRACING_ENABLED}=true\n")

    monkeypatch.chdir(tmp_path)
    assert is_tracing_enabled() is True


def test_is_tracing_enabled_false(tmp_path, monkeypatch):
    monkeypatch.delenv(MLFLOW_CURSOR_TRACING_ENABLED, raising=False)
    monkeypatch.chdir(tmp_path)
    assert is_tracing_enabled() is False


def test_get_tracing_status_no_config(tmp_path):
    status = get_tracing_status(tmp_path)
    assert status.enabled is False
    assert status.reason == "No configuration found"


def test_get_tracing_status_no_mlflow_hooks(tmp_path):
    hooks_path = get_cursor_hooks_path(tmp_path)
    hooks_path.parent.mkdir(parents=True, exist_ok=True)

    config_data = {"version": 1, "hooks": {"stop": [{"command": "some_other_command"}]}}
    with open(hooks_path, "w") as f:
        json.dump(config_data, f)

    status = get_tracing_status(tmp_path)
    assert status.enabled is False
    assert status.reason == "MLflow hooks not configured"


def test_get_tracing_status_enabled(tmp_path):
    hooks_path = get_cursor_hooks_path(tmp_path)
    hooks_path.parent.mkdir(parents=True, exist_ok=True)

    config_data = {
        "version": 1,
        "hooks": {"stop": [{"command": 'python3 -c "from mlflow.cursor.hooks import..."'}]},
    }
    with open(hooks_path, "w") as f:
        json.dump(config_data, f)

    env_file = tmp_path / CURSOR_DIR / ".env"
    env_file.write_text(f"{MLFLOW_CURSOR_TRACING_ENABLED}=true\n")

    status = get_tracing_status(tmp_path)
    assert status.enabled is True


def test_setup_environment_config_new_file(tmp_path):
    tracking_uri = "test://localhost"
    experiment_id = "123"

    setup_environment_config(tmp_path, tracking_uri, experiment_id)

    env_file = tmp_path / CURSOR_DIR / ".env"
    assert env_file.exists()

    content = env_file.read_text()
    assert f"{MLFLOW_CURSOR_TRACING_ENABLED}=true" in content
    assert "MLFLOW_TRACKING_URI=test://localhost" in content
    assert "MLFLOW_EXPERIMENT_ID=123" in content


def test_setup_environment_config_experiment_name(tmp_path):
    setup_environment_config(tmp_path, experiment_name="test_experiment")

    env_file = tmp_path / CURSOR_DIR / ".env"
    content = env_file.read_text()
    assert "MLFLOW_EXPERIMENT_NAME=test_experiment" in content
    assert "MLFLOW_EXPERIMENT_ID" not in content
