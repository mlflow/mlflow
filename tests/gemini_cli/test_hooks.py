import json

import pytest

from mlflow.gemini_cli.config import HOOK_FIELD_HOOKS, MLFLOW_HOOK_IDENTIFIER
from mlflow.gemini_cli.hooks import disable_tracing_hooks, setup_hooks_config, upsert_hook


@pytest.fixture
def temp_settings_path(tmp_path):
    """Provide a temporary settings.json path for tests."""
    return tmp_path / "settings.json"


def test_setup_hooks_config_creates_new_file(temp_settings_path):
    setup_hooks_config(temp_settings_path)

    assert temp_settings_path.exists()
    config = json.loads(temp_settings_path.read_text())
    assert HOOK_FIELD_HOOKS in config
    assert "SessionEnd" in config[HOOK_FIELD_HOOKS]

    # Verify the hook command contains the MLflow identifier
    session_end_hooks = config[HOOK_FIELD_HOOKS]["SessionEnd"]
    assert len(session_end_hooks) == 1
    hooks_list = session_end_hooks[0][HOOK_FIELD_HOOKS]
    assert len(hooks_list) == 1
    assert MLFLOW_HOOK_IDENTIFIER in hooks_list[0]["command"]


def test_setup_hooks_config_preserves_existing_hooks(temp_settings_path):
    # Create existing config with other hooks
    existing_config = {
        HOOK_FIELD_HOOKS: {
            "BeforeTool": [{"hooks": [{"type": "command", "command": "echo test"}]}],
        }
    }
    with open(temp_settings_path, "w") as f:
        json.dump(existing_config, f)

    setup_hooks_config(temp_settings_path)

    config = json.loads(temp_settings_path.read_text())
    assert "BeforeTool" in config[HOOK_FIELD_HOOKS]
    assert "SessionEnd" in config[HOOK_FIELD_HOOKS]


def test_setup_hooks_config_updates_existing_mlflow_hook(temp_settings_path):
    # Create config with existing MLflow hook
    existing_config = {
        HOOK_FIELD_HOOKS: {
            "SessionEnd": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": 'python -c "from mlflow.gemini_cli.hooks import old_handler; old_handler()"',
                        }
                    ]
                }
            ]
        }
    }
    with open(temp_settings_path, "w") as f:
        json.dump(existing_config, f)

    setup_hooks_config(temp_settings_path)

    config = json.loads(temp_settings_path.read_text())
    session_end_hooks = config[HOOK_FIELD_HOOKS]["SessionEnd"]
    assert len(session_end_hooks) == 1
    hooks_list = session_end_hooks[0][HOOK_FIELD_HOOKS]
    assert len(hooks_list) == 1
    assert "session_end_hook_handler" in hooks_list[0]["command"]


def test_disable_tracing_hooks_removes_mlflow_hooks(temp_settings_path):
    setup_hooks_config(temp_settings_path)

    result = disable_tracing_hooks(temp_settings_path)
    assert result is True

    config = json.loads(temp_settings_path.read_text())
    # The hooks section should be cleaned up
    assert HOOK_FIELD_HOOKS not in config or "SessionEnd" not in config.get(HOOK_FIELD_HOOKS, {})


def test_disable_tracing_hooks_no_config(tmp_path):
    non_existent = tmp_path / "missing.json"
    result = disable_tracing_hooks(non_existent)
    assert result is False


def test_disable_tracing_hooks_preserves_other_hooks(temp_settings_path):
    # Create config with both MLflow and other hooks
    config = {
        HOOK_FIELD_HOOKS: {
            "SessionEnd": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": 'python -c "from mlflow.gemini_cli.hooks import session_end_hook_handler; session_end_hook_handler()"',
                        }
                    ]
                }
            ],
            "BeforeTool": [
                {"hooks": [{"type": "command", "command": "echo other"}]}
            ],
        }
    }
    with open(temp_settings_path, "w") as f:
        json.dump(config, f)

    result = disable_tracing_hooks(temp_settings_path)
    assert result is True

    config = json.loads(temp_settings_path.read_text())
    assert "BeforeTool" in config[HOOK_FIELD_HOOKS]


def test_upsert_hook_adds_new_hook():
    config = {HOOK_FIELD_HOOKS: {}}
    upsert_hook(config, "SessionEnd", "session_end_hook_handler")

    assert "SessionEnd" in config[HOOK_FIELD_HOOKS]
    hooks_list = config[HOOK_FIELD_HOOKS]["SessionEnd"]
    assert len(hooks_list) == 1
    assert MLFLOW_HOOK_IDENTIFIER in hooks_list[0][HOOK_FIELD_HOOKS][0]["command"]
