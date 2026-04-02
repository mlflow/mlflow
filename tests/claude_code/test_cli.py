import json
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from mlflow.claude_code.cli import commands
from mlflow.claude_code.config import HOOK_FIELD_COMMAND, HOOK_FIELD_HOOKS
from mlflow.claude_code.hooks import upsert_hook


@pytest.fixture
def runner():
    """Provide a CLI runner for tests."""
    return CliRunner()


def test_claude_help_command(runner):
    result = runner.invoke(commands, ["--help"])
    assert result.exit_code == 0
    assert "Commands for autologging with MLflow" in result.output
    assert "claude" in result.output


def test_trace_command_help(runner):
    result = runner.invoke(commands, ["claude", "--help"])
    assert result.exit_code == 0
    assert "Set up Claude Code tracing" in result.output
    assert "--tracking-uri" in result.output
    assert "--experiment-id" in result.output
    assert "--disable" in result.output
    assert "--status" in result.output


def test_trace_status_with_no_config(runner):
    with runner.isolated_filesystem():
        result = runner.invoke(commands, ["claude", "--status"])
        assert result.exit_code == 0
        assert "❌ Claude tracing is not enabled" in result.output


def test_trace_disable_with_no_config(runner):
    with runner.isolated_filesystem():
        result = runner.invoke(commands, ["claude", "--disable"])
        assert result.exit_code == 0


def _get_hook_command_from_settings() -> str:
    settings_path = Path(".claude/settings.json")
    with open(settings_path) as f:
        config = json.load(f)

    if hooks := config.get("hooks"):
        for group in hooks.get("Stop", []):
            for hook in group.get("hooks", []):
                if command := hook.get("command"):
                    return command

    raise AssertionError("No hook command found in settings.json")


def test_claude_setup_with_uv_env_var(runner, monkeypatch):
    monkeypatch.setenv("UV", "/path/to/uv")

    with runner.isolated_filesystem():
        result = runner.invoke(commands, ["claude"])
        assert result.exit_code == 0

        hook_command = _get_hook_command_from_settings()
        assert hook_command == "uv run mlflow autolog claude stop-hook"


def test_claude_setup_without_uv_env_var(runner, monkeypatch):
    monkeypatch.delenv("UV", raising=False)

    with runner.isolated_filesystem():
        result = runner.invoke(commands, ["claude"])
        assert result.exit_code == 0

        hook_command = _get_hook_command_from_settings()
        assert hook_command == "mlflow autolog claude stop-hook"


def test_upsert_hook_uses_cli_command():
    config = {HOOK_FIELD_HOOKS: {}}
    upsert_hook(config, "Stop", "stop-hook")

    hook_command = config[HOOK_FIELD_HOOKS]["Stop"][0][HOOK_FIELD_HOOKS][0][HOOK_FIELD_COMMAND]
    assert "mlflow autolog claude stop-hook" in hook_command


def test_upsert_hook_upgrades_legacy_hook():
    legacy_command = (
        'python -I -c "from mlflow.claude_code.hooks import stop_hook_handler; stop_hook_handler()"'
    )
    config = {
        HOOK_FIELD_HOOKS: {
            "Stop": [{HOOK_FIELD_HOOKS: [{"type": "command", HOOK_FIELD_COMMAND: legacy_command}]}]
        }
    }
    upsert_hook(config, "Stop", "stop-hook")

    hook_command = config[HOOK_FIELD_HOOKS]["Stop"][0][HOOK_FIELD_HOOKS][0][HOOK_FIELD_COMMAND]
    assert "mlflow autolog claude stop-hook" in hook_command
    assert "python -I -c" not in hook_command


def test_stop_hook_subcommand_is_routable(runner):
    with mock.patch("mlflow.claude_code.cli.stop_hook_handler") as mock_handler:
        result = runner.invoke(commands, ["claude", "stop-hook"])
        assert result.exit_code == 0
        mock_handler.assert_called_once()
