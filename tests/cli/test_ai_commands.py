"""CLI integration tests for AI commands feature."""

from unittest import mock

from click.testing import CliRunner

from mlflow.cli import cli


def test_list_commands_cli():
    mock_commands = [
        {
            "key": "genai/analyze_experiment",
            "namespace": "genai",
            "description": "Analyzes an MLflow experiment",
        },
        {
            "key": "ml/train",
            "namespace": "ml",
            "description": "Training helper",
        },
    ]

    with mock.patch("mlflow.ai_commands.list_commands", return_value=mock_commands):
        runner = CliRunner()
        result = runner.invoke(cli, ["ai-commands", "list"])

    assert result.exit_code == 0
    assert "genai/analyze_experiment: Analyzes an MLflow experiment" in result.output
    assert "ml/train: Training helper" in result.output


def test_list_commands_with_namespace_cli():
    mock_commands = [
        {
            "key": "genai/analyze_experiment",
            "namespace": "genai",
            "description": "Analyzes an MLflow experiment",
        },
    ]

    with mock.patch(
        "mlflow.cli.ai_commands.list_commands", return_value=mock_commands
    ) as mock_list:
        runner = CliRunner()
        result = runner.invoke(cli, ["ai-commands", "list", "--namespace", "genai"])

    assert result.exit_code == 0
    mock_list.assert_called_once_with("genai")
    assert "genai/analyze_experiment" in result.output


def test_list_commands_empty_cli():
    with mock.patch("mlflow.ai_commands.list_commands", return_value=[]):
        runner = CliRunner()
        result = runner.invoke(cli, ["ai-commands", "list"])

    assert result.exit_code == 0
    assert "No AI commands found" in result.output


def test_list_commands_empty_namespace_cli():
    with mock.patch("mlflow.ai_commands.list_commands", return_value=[]):
        runner = CliRunner()
        result = runner.invoke(cli, ["ai-commands", "list", "--namespace", "unknown"])

    assert result.exit_code == 0
    assert "No AI commands found in namespace 'unknown'" in result.output


def test_get_command_cli():
    mock_content = """---
namespace: genai
description: Test command
---

Hello! This is test content."""

    with mock.patch("mlflow.ai_commands.get_command", return_value=mock_content):
        runner = CliRunner()
        result = runner.invoke(cli, ["ai-commands", "get", "genai/analyze_experiment"])

    assert result.exit_code == 0
    assert mock_content == result.output.rstrip("\n")


def test_get_invalid_command_cli():
    with mock.patch(
        "mlflow.cli.ai_commands.get_command",
        side_effect=FileNotFoundError("Command 'invalid/cmd' not found"),
    ):
        runner = CliRunner()
        result = runner.invoke(cli, ["ai-commands", "get", "invalid/cmd"])

    assert result.exit_code != 0
    assert "Error: Command 'invalid/cmd' not found" in result.output


def test_ai_commands_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["ai-commands", "--help"])

    assert result.exit_code == 0
    assert "Manage MLflow AI commands for LLMs" in result.output
    assert "list" in result.output
    assert "get" in result.output
    assert "run" in result.output


def test_get_command_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["ai-commands", "get", "--help"])

    assert result.exit_code == 0
    assert "Get a specific AI command by key" in result.output
    assert "KEY" in result.output


def test_list_command_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["ai-commands", "list", "--help"])

    assert result.exit_code == 0
    assert "List all available AI commands" in result.output
    assert "--namespace" in result.output


def test_run_command_cli():
    mock_content = """---
namespace: genai
description: Test command
---

# Test Command
This is test content."""

    with mock.patch("mlflow.ai_commands.get_command", return_value=mock_content):
        runner = CliRunner()
        result = runner.invoke(cli, ["ai-commands", "run", "genai/analyze_experiment"])

    assert result.exit_code == 0
    assert "The user has run an MLflow AI command via CLI" in result.output
    assert "Start executing the workflow immediately without any preamble" in result.output
    assert "# Test Command" in result.output
    assert "This is test content." in result.output
    # Should not have frontmatter
    assert "namespace: genai" not in result.output
    assert "description: Test command" not in result.output
    assert "---" not in result.output


def test_run_invalid_command_cli():
    with mock.patch(
        "mlflow.ai_commands.get_command",
        side_effect=FileNotFoundError("Command 'invalid/cmd' not found"),
    ):
        runner = CliRunner()
        result = runner.invoke(cli, ["ai-commands", "run", "invalid/cmd"])

    assert result.exit_code != 0
    assert "Error: Command 'invalid/cmd' not found" in result.output


def test_run_command_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["ai-commands", "run", "--help"])

    assert result.exit_code == 0
    assert "Get a command formatted for execution by an AI assistant" in result.output
    assert "KEY" in result.output


def test_actual_command_exists():
    runner = CliRunner()

    # Test list includes our command
    result = runner.invoke(cli, ["ai-commands", "list"])
    assert result.exit_code == 0
    assert "genai/analyze_experiment" in result.output

    # Test we can get the command
    result = runner.invoke(cli, ["ai-commands", "get", "genai/analyze_experiment"])
    assert result.exit_code == 0
    assert "# Analyze Experiment" in result.output
    assert "Analyzes traces in an MLflow experiment" in result.output

    # Test we can run the command
    result = runner.invoke(cli, ["ai-commands", "run", "genai/analyze_experiment"])
    assert result.exit_code == 0
    assert "The user has run an MLflow AI command via CLI" in result.output
    assert "Start executing the workflow immediately without any preamble" in result.output
    assert "# Analyze Experiment" in result.output
    # Should not have frontmatter
    assert "namespace: genai" not in result.output
    assert "---" not in result.output

    # Test filtering by namespace
    result = runner.invoke(cli, ["ai-commands", "list", "--namespace", "genai"])
    assert result.exit_code == 0
    assert "genai/analyze_experiment" in result.output

    # Test filtering by wrong namespace excludes it
    result = runner.invoke(cli, ["ai-commands", "list", "--namespace", "ml"])
    assert result.exit_code == 0
    assert "genai/analyze_experiment" not in result.output
