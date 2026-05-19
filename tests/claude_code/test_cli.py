import json
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from mlflow.claude_code.cli import commands


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def _clear_mlflow_env(monkeypatch):
    for name in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_ID",
        "MLFLOW_EXPERIMENT_NAME",
    ):
        monkeypatch.delenv(name, raising=False)


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
    assert "--non-interactive" in result.output
    assert "--disable" in result.output
    assert "--status" in result.output


def test_trace_status_with_no_config(runner):
    with runner.isolated_filesystem():
        result = runner.invoke(commands, ["claude", "--status"])
        assert result.exit_code == 0
        assert "Claude tracing is not enabled" in result.output


def test_trace_disable_with_no_config(runner):
    with runner.isolated_filesystem():
        result = runner.invoke(commands, ["claude", "--disable"])
        assert result.exit_code == 0


def test_claude_setup_installs_plugin_and_writes_env(runner):
    with (
        runner.isolated_filesystem(),
        mock.patch("mlflow.claude_code.cli.ensure_plugin_installed") as mock_install,
    ):
        result = runner.invoke(commands, ["claude", "-u", "http://localhost:5000", "-e", "123"])
        assert result.exit_code == 0

        mock_install.assert_called_once_with(Path(".").resolve())

        config = json.loads(Path(".claude/settings.json").read_text())
        assert config["env"]["MLFLOW_CLAUDE_TRACING_ENABLED"] == "true"
        assert config["env"]["MLFLOW_TRACKING_URI"] == "http://localhost:5000"
        assert config["env"]["MLFLOW_EXPERIMENT_ID"] == "123"
        assert "hooks" not in config


def test_claude_setup_prompts_for_missing_values_in_interactive_mode(runner):
    with (
        runner.isolated_filesystem(),
        mock.patch("mlflow.claude_code.cli.ensure_plugin_installed"),
        mock.patch("mlflow.claude_code.cli._is_interactive_shell", return_value=True),
        mock.patch("mlflow.get_tracking_uri", return_value="http://localhost:5000"),
    ):
        result = runner.invoke(commands, ["claude"], input="\n42\n")
        assert result.exit_code == 0

        config = json.loads(Path(".claude/settings.json").read_text())
        assert config["env"]["MLFLOW_TRACKING_URI"] == "http://localhost:5000"
        assert config["env"]["MLFLOW_EXPERIMENT_ID"] == "42"
        assert "interactive mode" in result.output
        assert "MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_ID" in result.output


def test_claude_setup_shows_plugin_install_message(runner):
    with (
        runner.isolated_filesystem(),
        mock.patch("mlflow.claude_code.cli.ensure_plugin_installed"),
    ):
        result = runner.invoke(commands, ["claude", "-u", "http://localhost:5000", "-e", "123"])
        assert result.exit_code == 0
        assert "MLflow Claude plugin for Claude Code" in result.output
        assert "Claude Code plugin installed" in result.output


def test_claude_setup_non_interactive_uses_defaults(runner):
    with (
        runner.isolated_filesystem(),
        mock.patch("mlflow.claude_code.cli.ensure_plugin_installed"),
        mock.patch("mlflow.get_tracking_uri", return_value="file:///tmp/mlruns"),
    ):
        result = runner.invoke(commands, ["claude", "--non-interactive"])
        assert result.exit_code == 0

        config = json.loads(Path(".claude/settings.json").read_text())
        assert config["env"]["MLFLOW_TRACKING_URI"] == "file:///tmp/mlruns"
        assert config["env"]["MLFLOW_EXPERIMENT_ID"] == "0"


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


def test_setup_rejects_experiment_id_and_name_together(runner):
    with runner.isolated_filesystem():
        result = runner.invoke(
            commands,
            ["claude", "--experiment-id", "1", "--experiment-name", "my-exp"],
        )
        assert result.exit_code != 0
        assert "Choose either --experiment-id or --experiment-name" in result.output


def test_stop_hook_subcommand_is_routable(runner):
    with mock.patch("mlflow.claude_code.cli.stop_hook_handler") as mock_handler:
        result = runner.invoke(commands, ["claude", "stop-hook"])
        assert result.exit_code == 0
        mock_handler.assert_called_once()
