import json

from mlflow.qwen_code.config import (
    get_tracing_status,
    load_qwen_config,
    save_qwen_config,
    setup_environment_config,
)


def test_load_qwen_config_valid_json(tmp_path):
    settings_path = tmp_path / "settings.json"
    config = {"hooks": {"Stop": []}}
    settings_path.write_text(json.dumps(config))

    result = load_qwen_config(settings_path)
    assert result == config


def test_load_qwen_config_missing_file(tmp_path):
    result = load_qwen_config(tmp_path / "nonexistent.json")
    assert result == {}


def test_load_qwen_config_invalid_json(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("not json")

    result = load_qwen_config(settings_path)
    assert result == {}


def test_save_qwen_config_creates_file(tmp_path):
    settings_path = tmp_path / "settings.json"
    config = {"test": "value"}
    save_qwen_config(settings_path, config)

    assert settings_path.exists()
    assert json.loads(settings_path.read_text()) == config


def test_get_tracing_status_no_config(tmp_path):
    status = get_tracing_status(tmp_path / "settings.json")
    assert not status.enabled
    assert status.reason == "No configuration found"


def test_get_tracing_status_enabled(tmp_path):
    settings_path = tmp_path / "settings.json"
    config = {"env": {"MLFLOW_QWEN_TRACING_ENABLED": "true", "MLFLOW_TRACKING_URI": "databricks"}}
    settings_path.write_text(json.dumps(config))

    status = get_tracing_status(settings_path)
    assert status.enabled
    assert status.tracking_uri == "databricks"


def test_get_tracing_status_disabled(tmp_path):
    settings_path = tmp_path / "settings.json"
    config = {"env": {"MLFLOW_QWEN_TRACING_ENABLED": "false"}}
    settings_path.write_text(json.dumps(config))

    status = get_tracing_status(settings_path)
    assert not status.enabled


def test_setup_environment_config(tmp_path):
    settings_path = tmp_path / "settings.json"
    setup_environment_config(settings_path, tracking_uri="databricks", experiment_id="123")

    config = json.loads(settings_path.read_text())
    assert config["env"]["MLFLOW_QWEN_TRACING_ENABLED"] == "true"
    assert config["env"]["MLFLOW_TRACKING_URI"] == "databricks"
    assert config["env"]["MLFLOW_EXPERIMENT_ID"] == "123"
    assert "MLFLOW_EXPERIMENT_NAME" not in config["env"]
