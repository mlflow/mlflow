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
        # Should handle gracefully even if no config exists


def _get_hook_command_from_settings():
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
    # Set UV environment variable
    monkeypatch.setenv("UV", "/path/to/uv")

    with runner.isolated_filesystem():
        result = runner.invoke(commands, ["claude"])
        assert result.exit_code == 0

        hook_command = _get_hook_command_from_settings()
        assert "uv run python" in hook_command
        assert "from mlflow.claude_code.hooks import stop_hook_handler" in hook_command


def test_claude_setup_without_uv_env_var(runner, monkeypatch):
    # Ensure UV environment variable is not set
    monkeypatch.delenv("UV", raising=False)

    with runner.isolated_filesystem():
        result = runner.invoke(commands, ["claude"])
        assert result.exit_code == 0

        hook_command = _get_hook_command_from_settings()
        # Should start with 'python' but NOT 'uv run python'
        assert hook_command.startswith("python -c")
        assert "uv run python" not in hook_command
        assert "from mlflow.claude_code.hooks import stop_hook_handler" in hook_command
