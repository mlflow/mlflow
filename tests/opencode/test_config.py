import json
from pathlib import Path

from mlflow.opencode.config import (
    MLFLOW_PLUGIN_NPM_PACKAGE,
    disable_tracing,
    get_opencode_config_path,
    get_tracing_status,
    load_json_config,
    save_opencode_config,
    setup_hook_config,
)


def test_get_opencode_config_path():
    directory = Path("/test/project")
    config_path = get_opencode_config_path(directory)
    assert config_path == Path("/test/project/opencode.json")


def test_load_json_config_nonexistent(tmp_path):
    config_path = tmp_path / "opencode.json"
    config = load_json_config(config_path)
    assert config == {}


def test_load_json_config_valid(tmp_path):
    config_path = tmp_path / "opencode.json"
    expected = {"model": "anthropic/claude-sonnet-4-20250514", "plugin": []}
    with open(config_path, "w") as f:
        json.dump(expected, f)

    config = load_json_config(config_path)
    assert config == expected


def test_load_json_config_invalid_json(tmp_path):
    config_path = tmp_path / "opencode.json"
    with open(config_path, "w") as f:
        f.write("invalid json {")

    config = load_json_config(config_path)
    assert config == {}


def test_save_opencode_config(tmp_path):
    config_path = tmp_path / "opencode.json"
    config = {"model": "anthropic/claude-sonnet-4-20250514", "plugin": ["test-plugin"]}

    save_opencode_config(config_path, config)

    assert config_path.exists()
    with open(config_path) as f:
        loaded = json.load(f)
    assert loaded == config


def test_save_opencode_config_creates_directory(tmp_path):
    config_path = tmp_path / "subdir" / "opencode.json"
    config = {"test": "value"}

    save_opencode_config(config_path, config)

    assert config_path.exists()


def test_get_tracing_status_no_config(tmp_path):
    config_path = tmp_path / "opencode.json"
    status = get_tracing_status(config_path)

    assert status.enabled is False
    assert status.reason == "No configuration found"


def test_get_tracing_status_no_plugin(tmp_path):
    config_path = tmp_path / "opencode.json"
    with open(config_path, "w") as f:
        json.dump({"model": "anthropic/claude-sonnet-4-20250514"}, f)

    status = get_tracing_status(config_path)
    assert status.enabled is False


def test_get_tracing_status_with_plugin(tmp_path):
    config_path = tmp_path / "opencode.json"
    with open(config_path, "w") as f:
        json.dump({"plugin": [MLFLOW_PLUGIN_NPM_PACKAGE]}, f)

    status = get_tracing_status(config_path)
    assert status.enabled is True


def test_get_tracing_status_with_file_plugin(tmp_path):
    config_path = tmp_path / "opencode.json"
    with open(config_path, "w") as f:
        json.dump({"plugin": ["file:///path/to/mlflow/opencode/plugin"]}, f)

    status = get_tracing_status(config_path)
    assert status.enabled is True


def test_get_tracing_status_reads_mlflow_config(tmp_path):
    config_path = tmp_path / "opencode.json"
    with open(config_path, "w") as f:
        json.dump({"plugin": [MLFLOW_PLUGIN_NPM_PACKAGE]}, f)

    # Create mlflow.json with tracking settings
    opencode_dir = tmp_path / ".opencode"
    opencode_dir.mkdir(parents=True, exist_ok=True)
    mlflow_config = opencode_dir / "mlflow.json"
    mlflow_config.write_text(
        '{"enabled": true, "trackingUri": "http://localhost:5000", "experimentId": "123"}'
    )

    status = get_tracing_status(config_path)
    assert status.enabled is True
    assert status.tracking_uri == "http://localhost:5000"
    assert status.experiment_id == "123"


def test_setup_hook_config_creates_config(tmp_path):
    config_path = tmp_path / "opencode.json"

    plugin_name = setup_hook_config(config_path)

    assert config_path.exists()
    config = load_json_config(config_path)
    assert plugin_name in config.get("plugin", [])
    assert plugin_name == MLFLOW_PLUGIN_NPM_PACKAGE


def test_setup_hook_config_preserves_existing(tmp_path):
    config_path = tmp_path / "opencode.json"
    with open(config_path, "w") as f:
        json.dump({"model": "anthropic/claude-sonnet-4-20250514", "plugin": ["other-plugin"]}, f)

    setup_hook_config(config_path)

    config = load_json_config(config_path)
    assert config["model"] == "anthropic/claude-sonnet-4-20250514"
    assert "other-plugin" in config["plugin"]


def test_setup_hook_config_with_tracking_uri(tmp_path):
    config_path = tmp_path / "opencode.json"

    setup_hook_config(config_path, tracking_uri="databricks")

    # Check that mlflow.json is created with tracking URI
    mlflow_config = tmp_path / ".opencode" / "mlflow.json"
    assert mlflow_config.exists()
    with open(mlflow_config) as f:
        config = json.load(f)
    assert config["enabled"] is True
    assert config["trackingUri"] == "databricks"


def test_setup_hook_config_with_experiment_id(tmp_path):
    config_path = tmp_path / "opencode.json"

    setup_hook_config(config_path, experiment_id="123456")

    # Check that mlflow.json is created with experiment ID
    mlflow_config = tmp_path / ".opencode" / "mlflow.json"
    assert mlflow_config.exists()
    with open(mlflow_config) as f:
        config = json.load(f)
    assert config["experimentId"] == "123456"


def test_setup_hook_config_with_experiment_name(tmp_path):
    config_path = tmp_path / "opencode.json"

    setup_hook_config(config_path, experiment_name="My Experiment")

    # Check that mlflow.json is created with experiment name
    mlflow_config = tmp_path / ".opencode" / "mlflow.json"
    assert mlflow_config.exists()
    with open(mlflow_config) as f:
        config = json.load(f)
    assert config["experimentName"] == "My Experiment"


def test_setup_hook_config_updates_existing_plugin(tmp_path):
    config_path = tmp_path / "opencode.json"
    # Set up initial config with a file:// mlflow plugin
    with open(config_path, "w") as f:
        json.dump({"plugin": ["file:///old/path/to/mlflow/plugin"]}, f)

    # Update configuration
    plugin_name = setup_hook_config(config_path, tracking_uri="new-uri")

    config = load_json_config(config_path)
    # Should have only one MLflow plugin (not duplicated)
    mlflow_plugins = [p for p in config.get("plugin", []) if "mlflow" in p.lower()]
    assert len(mlflow_plugins) == 1
    assert mlflow_plugins[0] == plugin_name


def test_disable_tracing_no_config(tmp_path):
    config_path = tmp_path / "opencode.json"
    result = disable_tracing(config_path)
    assert result is False


def test_disable_tracing_removes_plugin(tmp_path):
    config_path = tmp_path / "opencode.json"
    with open(config_path, "w") as f:
        json.dump({"plugin": [MLFLOW_PLUGIN_NPM_PACKAGE, "other-plugin"]}, f)

    result = disable_tracing(config_path)

    assert result is True
    config = load_json_config(config_path)
    assert MLFLOW_PLUGIN_NPM_PACKAGE not in config.get("plugin", [])
    assert "other-plugin" in config["plugin"]


def test_disable_tracing_removes_file_plugin(tmp_path):
    config_path = tmp_path / "opencode.json"
    with open(config_path, "w") as f:
        json.dump({"plugin": ["file:///path/to/mlflow/opencode/plugin", "other-plugin"]}, f)

    # Create mlflow.json config
    opencode_dir = tmp_path / ".opencode"
    opencode_dir.mkdir(parents=True, exist_ok=True)
    mlflow_config = opencode_dir / "mlflow.json"
    mlflow_config.write_text('{"enabled": true}')

    result = disable_tracing(config_path)

    assert result is True
    config = load_json_config(config_path)
    # Should have removed MLflow plugin
    assert not any("mlflow" in p.lower() for p in config.get("plugin", []))
    assert "other-plugin" in config["plugin"]
    # Should have removed mlflow config
    assert not mlflow_config.exists()


def test_disable_tracing_keeps_other_plugins(tmp_path):
    config_path = tmp_path / "opencode.json"
    with open(config_path, "w") as f:
        json.dump({"plugin": ["file:///path/to/mlflow/opencode/plugin", "some-other-plugin"]}, f)

    result = disable_tracing(config_path)

    assert result is True
    config = load_json_config(config_path)
    assert "some-other-plugin" in config["plugin"]
