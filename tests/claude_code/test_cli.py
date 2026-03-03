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


def test_claude_setup_with_uv_env_var(runner, monkeypatch):
    # Set UV environment variable
    monkeypatch.setenv("UV", "/path/to/uv")

    with runner.isolated_filesystem():
        result = runner.invoke(commands, ["claude"])
        assert result.exit_code == 0

        # Read the generated settings.json
        settings_path = Path(".claude/settings.json")
        assert settings_path.exists()

        with open(settings_path) as f:
            config = json.load(f)

        # Check that the hook command uses 'uv run python'
        assert "hooks" in config
        assert "Stop" in config["hooks"]
        hook_groups = config["hooks"]["Stop"]
        assert len(hook_groups) > 0

        # Find the MLflow hook
        hook_command = None
        for group in hook_groups:
            if "hooks" in group:
                for hook in group["hooks"]:
                    if "command" in hook:
                        hook_command = hook["command"]
                        break

        assert hook_command is not None
        assert "uv run python" in hook_command
        assert "from mlflow.claude_code.hooks import stop_hook_handler" in hook_command


def test_claude_setup_without_uv_env_var(runner, monkeypatch):
    # Ensure UV environment variable is not set
    monkeypatch.delenv("UV", raising=False)

    with runner.isolated_filesystem():
        result = runner.invoke(commands, ["claude"])
        assert result.exit_code == 0

        # Read the generated settings.json
        settings_path = Path(".claude/settings.json")
        assert settings_path.exists()

        with open(settings_path) as f:
            config = json.load(f)

        # Check that the hook command uses plain 'python'
        assert "hooks" in config
        assert "Stop" in config["hooks"]
        hook_groups = config["hooks"]["Stop"]
        assert len(hook_groups) > 0

        # Find the MLflow hook
        hook_command = None
        for group in hook_groups:
            if "hooks" in group:
                for hook in group["hooks"]:
                    if "command" in hook:
                        hook_command = hook["command"]
                        break

        assert hook_command is not None
        # Should start with 'python' but NOT 'uv run python'
        assert hook_command.startswith("python -c")
        assert "uv run python" not in hook_command
        assert "from mlflow.claude_code.hooks import stop_hook_handler" in hook_command
