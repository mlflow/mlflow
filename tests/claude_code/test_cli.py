import json
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from mlflow.claude_code.cli import commands
from mlflow.claude_code.config import HOOK_FIELD_COMMAND, HOOK_FIELD_HOOKS
from mlflow.claude_code.hooks import disable_tracing_hooks, upsert_hook


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


def test_upsert_hook_with_matcher():
    config = {HOOK_FIELD_HOOKS: {}}
    upsert_hook(config, "PostToolUse", "exit-plan-mode-hook", matcher="ExitPlanMode")

    groups = config[HOOK_FIELD_HOOKS]["PostToolUse"]
    assert len(groups) == 1
    group = groups[0]
    assert group.get("matcher") == "ExitPlanMode"
    hook_command = group[HOOK_FIELD_HOOKS][0][HOOK_FIELD_COMMAND]
    assert "mlflow autolog claude exit-plan-mode-hook" in hook_command


def test_upsert_hook_stop_and_post_tool_use_are_independent():
    config = {HOOK_FIELD_HOOKS: {}}
    upsert_hook(config, "Stop", "stop-hook")
    upsert_hook(config, "PostToolUse", "exit-plan-mode-hook", matcher="ExitPlanMode")

    stop_cmd = config[HOOK_FIELD_HOOKS]["Stop"][0][HOOK_FIELD_HOOKS][0][HOOK_FIELD_COMMAND]
    post_cmd = config[HOOK_FIELD_HOOKS]["PostToolUse"][0][HOOK_FIELD_HOOKS][0][HOOK_FIELD_COMMAND]
    assert "stop-hook" in stop_cmd
    assert "exit-plan-mode-hook" in post_cmd


def test_exit_plan_mode_hook_subcommand_is_routable(runner):
    with mock.patch("mlflow.claude_code.cli.exit_plan_mode_hook_handler") as mock_handler:
        result = runner.invoke(commands, ["claude", "exit-plan-mode-hook"])
        assert result.exit_code == 0
        mock_handler.assert_called_once()


def test_disable_tracing_hooks_partial_removal(tmp_path):
    # A group that mixes MLflow and non-MLflow hooks should still report hooks_removed=True.
    settings_path = tmp_path / "settings.json"
    mlflow_cmd = "uv run mlflow autolog claude stop-hook"
    other_cmd = "echo other-hook"
    settings_path.write_text(
        json.dumps({
            "hooks": {
                "Stop": [
                    {
                        "hooks": [
                            {"type": "command", "command": mlflow_cmd},
                            {"type": "command", "command": other_cmd},
                        ]
                    }
                ]
            }
        })
    )

    removed = disable_tracing_hooks(settings_path)

    assert removed
    config = json.loads(settings_path.read_text())
    remaining = config["hooks"]["Stop"][0]["hooks"]
    assert len(remaining) == 1
    assert remaining[0]["command"] == other_cmd
