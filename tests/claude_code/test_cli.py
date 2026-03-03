import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from mlflow.claude_code.cli import commands


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
        assert hook_command == (
            "uv run python -c "
            '"from mlflow.claude_code.hooks import stop_hook_handler; stop_hook_handler()"'
        )


def test_claude_setup_without_uv_env_var(runner, monkeypatch):
    monkeypatch.delenv("UV", raising=False)

    with runner.isolated_filesystem():
        result = runner.invoke(commands, ["claude"])
        assert result.exit_code == 0

        hook_command = _get_hook_command_from_settings()
        assert hook_command == (
            "python -c "
            '"from mlflow.claude_code.hooks import stop_hook_handler; stop_hook_handler()"'
        )
