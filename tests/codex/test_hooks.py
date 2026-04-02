import json

from mlflow.codex.config import load_codex_hooks
from mlflow.codex.hooks import disable_tracing_hooks, setup_hooks_config, upsert_hook


def test_upsert_hook_creates_new(tmp_path):
    config = {}
    upsert_hook(config, "Stop", "stop-hook")

    assert "Stop" in config
    assert len(config["Stop"]) == 1
    hooks = config["Stop"][0]["hooks"]
    assert len(hooks) == 1
    assert "mlflow autolog codex stop-hook" in hooks[0]["command"]


def test_upsert_hook_updates_existing(tmp_path):
    config = {
        "Stop": [{"hooks": [{"type": "command", "command": "mlflow autolog codex stop-hook"}]}]
    }
    upsert_hook(config, "Stop", "stop-hook")

    # Should still have exactly one hook group with one hook
    assert len(config["Stop"]) == 1
    assert len(config["Stop"][0]["hooks"]) == 1


def test_setup_hooks_config(tmp_path):
    setup_hooks_config(tmp_path)

    hooks_path = tmp_path / "hooks.json"
    assert hooks_path.exists()

    config = load_codex_hooks(hooks_path)
    assert "Stop" in config


def test_disable_tracing_hooks_removes_hooks(tmp_path):
    # Set up hooks first
    setup_hooks_config(tmp_path)

    result = disable_tracing_hooks(tmp_path)
    assert result is True

    hooks_path = tmp_path / "hooks.json"
    # File should be removed since config is empty after removing the hook
    assert not hooks_path.exists()


def test_disable_tracing_hooks_no_config(tmp_path):
    result = disable_tracing_hooks(tmp_path)
    assert result is False


def test_disable_tracing_hooks_preserves_other_hooks(tmp_path):
    hooks_path = tmp_path / "hooks.json"
    config = {
        "Stop": [
            {"hooks": [{"type": "command", "command": "mlflow autolog codex stop-hook"}]},
            {"hooks": [{"type": "command", "command": "other-command"}]},
        ]
    }
    hooks_path.write_text(json.dumps(config))

    result = disable_tracing_hooks(tmp_path)
    assert result is True

    remaining = json.loads(hooks_path.read_text())
    assert len(remaining["Stop"]) == 1
    assert remaining["Stop"][0]["hooks"][0]["command"] == "other-command"
