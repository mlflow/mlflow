import json

from mlflow.codex.config import (
    get_tracing_status,
    load_codex_hooks,
    save_codex_hooks,
)


def test_load_codex_hooks_valid_json(tmp_path):
    hooks_path = tmp_path / "hooks.json"
    config = {"Stop": [{"hooks": [{"type": "command", "command": "test"}]}]}
    hooks_path.write_text(json.dumps(config))

    result = load_codex_hooks(hooks_path)
    assert result == config


def test_load_codex_hooks_missing_file(tmp_path):
    result = load_codex_hooks(tmp_path / "nonexistent.json")
    assert result == {}


def test_load_codex_hooks_invalid_json(tmp_path):
    hooks_path = tmp_path / "hooks.json"
    hooks_path.write_text("not json")

    result = load_codex_hooks(hooks_path)
    assert result == {}


def test_save_codex_hooks_creates_file(tmp_path):
    hooks_path = tmp_path / "hooks.json"
    config = {"Stop": []}
    save_codex_hooks(hooks_path, config)

    assert hooks_path.exists()
    assert json.loads(hooks_path.read_text()) == config


def test_get_tracing_status_no_config(tmp_path):
    status = get_tracing_status(tmp_path)
    assert not status.enabled
    assert status.reason == "No configuration found"


def test_get_tracing_status_with_mlflow_hook(tmp_path):
    hooks_path = tmp_path / "hooks.json"
    config = {
        "Stop": [{"hooks": [{"type": "command", "command": "mlflow autolog codex stop-hook"}]}]
    }
    hooks_path.write_text(json.dumps(config))

    status = get_tracing_status(tmp_path)
    assert status.enabled


def test_get_tracing_status_without_mlflow_hook(tmp_path):
    hooks_path = tmp_path / "hooks.json"
    config = {"Stop": [{"hooks": [{"type": "command", "command": "some-other-command"}]}]}
    hooks_path.write_text(json.dumps(config))

    status = get_tracing_status(tmp_path)
    assert not status.enabled
    assert status.reason == "MLflow hooks not configured"
