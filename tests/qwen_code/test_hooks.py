import json

from mlflow.qwen_code.config import load_qwen_config
from mlflow.qwen_code.hooks import disable_tracing_hooks, setup_hooks_config, upsert_hook


def test_upsert_hook_creates_new():
    config = {}
    upsert_hook(config, "Stop", "stop-hook")

    assert "hooks" in config
    assert "Stop" in config["hooks"]
    assert len(config["hooks"]["Stop"]) == 1
    hooks = config["hooks"]["Stop"][0]["hooks"]
    assert len(hooks) == 1
    assert "mlflow autolog qwen-code stop-hook" in hooks[0]["command"]


def test_upsert_hook_updates_existing():
    config = {
        "hooks": {
            "Stop": [
                {"hooks": [{"type": "command", "command": "mlflow autolog qwen-code stop-hook"}]}
            ]
        }
    }
    upsert_hook(config, "Stop", "stop-hook")

    assert len(config["hooks"]["Stop"]) == 1
    assert len(config["hooks"]["Stop"][0]["hooks"]) == 1


def test_setup_hooks_config(tmp_path):
    setup_hooks_config(tmp_path)

    settings_path = tmp_path / "settings.json"
    assert settings_path.exists()

    config = load_qwen_config(settings_path)
    assert "hooks" in config
    assert "Stop" in config["hooks"]


def test_disable_tracing_hooks_removes_hooks(tmp_path):
    setup_hooks_config(tmp_path)

    result = disable_tracing_hooks(tmp_path)
    assert result is True


def test_disable_tracing_hooks_no_config(tmp_path):
    result = disable_tracing_hooks(tmp_path)
    assert result is False


def test_disable_tracing_hooks_preserves_other_hooks(tmp_path):
    settings_path = tmp_path / "settings.json"
    config = {
        "hooks": {
            "Stop": [
                {"hooks": [{"type": "command", "command": "mlflow autolog qwen-code stop-hook"}]},
                {"hooks": [{"type": "command", "command": "other-command"}]},
            ]
        }
    }
    settings_path.write_text(json.dumps(config))

    result = disable_tracing_hooks(tmp_path)
    assert result is True

    remaining = json.loads(settings_path.read_text())
    assert len(remaining["hooks"]["Stop"]) == 1
    assert remaining["hooks"]["Stop"][0]["hooks"][0]["command"] == "other-command"
