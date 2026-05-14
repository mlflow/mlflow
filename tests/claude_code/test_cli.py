import json
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from mlflow.claude_code.cli import commands


@pytest.fixture
def runner():
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


def test_claude_setup_installs_plugin_and_writes_env(runner):
    with (
        runner.isolated_filesystem(),
        mock.patch("mlflow.claude_code.cli.ensure_plugin_installed") as mock_install,
        mock.patch("mlflow.claude_code.cli.migrate_legacy_hooks") as mock_migrate,
    ):
        result = runner.invoke(commands, ["claude", "-u", "http://localhost:5000", "-e", "123"])
        assert result.exit_code == 0

        mock_install.assert_called_once_with(Path.cwd())
        mock_migrate.assert_called_once_with(Path.cwd() / ".claude" / "settings.json")

        config = json.loads(Path(".claude/settings.json").read_text())
        assert config["env"]["MLFLOW_CLAUDE_TRACING_ENABLED"] == "true"
        assert config["env"]["MLFLOW_TRACKING_URI"] == "http://localhost:5000"
        assert config["env"]["MLFLOW_EXPERIMENT_ID"] == "123"
        assert "hooks" not in config


def test_mlflow_cmd_empty_string_raises_error(runner):
    with runner.isolated_filesystem():
        result = runner.invoke(commands, ["claude", "--mlflow-cmd", ""])
        assert result.exit_code != 0
        assert "must not be empty or whitespace-only" in result.output


def test_claude_setup_surfaces_plugin_install_failure(runner):
    with (
        runner.isolated_filesystem(),
        mock.patch(
            "mlflow.claude_code.cli.ensure_plugin_installed",
            side_effect=RuntimeError("boom"),
        ),
    ):
        result = runner.invoke(commands, ["claude"])
        assert result.exit_code != 0
        assert "boom" in result.output


def test_mlflow_cmd_whitespace_only_raises_error(runner):
    with runner.isolated_filesystem():
        result = runner.invoke(commands, ["claude", "--mlflow-cmd", "   "])
        assert result.exit_code != 0
        assert "must not be empty or whitespace-only" in result.output


def test_stop_hook_subcommand_is_routable(runner):
    with mock.patch("mlflow.claude_code.cli.stop_hook_handler") as mock_handler:
        result = runner.invoke(commands, ["claude", "stop-hook"])
        assert result.exit_code == 0
        mock_handler.assert_called_once()
