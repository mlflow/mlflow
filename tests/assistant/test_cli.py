import os
from unittest import mock

import pytest
from click.testing import CliRunner

from mlflow.assistant.cli import commands


@pytest.fixture
def runner():
    return CliRunner()


def test_assistant_help(runner):
    result = runner.invoke(commands, ["--help"])
    assert result.exit_code == 0
    assert "AI-powered trace analysis" in result.output
    assert "--configure" in result.output


def test_configure_cli_not_found(runner):
    with mock.patch("mlflow.assistant.cli.shutil.which", return_value=None):
        result = runner.invoke(commands, ["--configure"], input="1\n")
        assert "not installed" in result.output


def test_configure_auth_failure(runner):
    mock_result = mock.Mock()
    mock_result.returncode = 1
    mock_result.stderr = "unauthorized"

    with (
        mock.patch("mlflow.assistant.cli.shutil.which", return_value="/usr/bin/claude"),
        mock.patch(
            "mlflow.assistant.providers.claude_code.subprocess.run",
            return_value=mock_result,
        ),
    ):
        result = runner.invoke(commands, ["--configure"], input="1\n")
        assert result.exit_code == 0
        # Should show error about authentication
        assert "Not authenticated" in result.output or "not installed" in result.output.lower()


def test_configure_experiment_fetch_failure(runner):
    mock_result = mock.Mock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    with (
        mock.patch("mlflow.assistant.cli.shutil.which", return_value="/usr/bin/claude"),
        mock.patch(
            "mlflow.assistant.providers.claude_code.subprocess.run",
            return_value=mock_result,
        ),
        mock.patch(
            "mlflow.assistant.cli._fetch_recent_experiments",
            return_value=[],
        ),
    ):
        # Input: provider=1, connect=y, tracking_uri=default
        result = runner.invoke(
            commands,
            ["--configure"],
            input="1\ny\nhttp://localhost:5000\n",
        )
        assert "Could not fetch experiments" in result.output


def test_configure_success(runner, tmp_path):
    mock_result = mock.Mock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    mock_config = mock.Mock()
    mock_config.providers = {}
    mock_config.projects = {}

    with (
        mock.patch("mlflow.assistant.cli.shutil.which", return_value="/usr/bin/claude"),
        mock.patch(
            "mlflow.assistant.providers.claude_code.subprocess.run",
            return_value=mock_result,
        ),
        mock.patch(
            "mlflow.assistant.cli._fetch_recent_experiments",
            return_value=[("1", "Test Experiment")],
        ),
        mock.patch(
            "mlflow.assistant.cli.AssistantConfig.load",
            return_value=mock_config,
        ),
        mock.patch.object(mock_config, "save"),
        mock.patch.object(mock_config, "set_provider"),
        runner.isolated_filesystem(temp_dir=tmp_path),
    ):
        # Input: provider=1, connect=y, tracking_uri, experiment=1, project_path, model=default
        result = runner.invoke(
            commands,
            ["--configure"],
            input=f"1\ny\nhttp://localhost:5000\n1\n{tmp_path}\ndefault\n",
        )
        assert "Setup Complete" in result.output


def test_configure_tilde_expansion(runner):
    mock_result = mock.Mock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    mock_config = mock.Mock()
    mock_config.providers = {}
    projects_dict = {}
    mock_config.projects = projects_dict

    home_dir = os.path.expanduser("~")

    with (
        mock.patch("mlflow.assistant.cli.shutil.which", return_value="/usr/bin/claude"),
        mock.patch(
            "mlflow.assistant.providers.claude_code.subprocess.run",
            return_value=mock_result,
        ),
        mock.patch(
            "mlflow.assistant.cli._fetch_recent_experiments",
            return_value=[("1", "Test Experiment")],
        ),
        mock.patch(
            "mlflow.assistant.cli.AssistantConfig.load",
            return_value=mock_config,
        ),
        mock.patch.object(mock_config, "save"),
        mock.patch.object(mock_config, "set_provider"),
    ):
        # Input: provider=1, connect=y, tracking_uri, experiment=1, project_path=~, model=default
        result = runner.invoke(
            commands,
            ["--configure"],
            input="1\ny\nhttp://localhost:5000\n1\n~\ndefault\n",
        )
        # Should succeed because ~ expands to home dir which exists
        assert "Setup Complete" in result.output
        # Verify the saved path is the expanded path, not ~
        assert "1" in projects_dict
        assert projects_dict["1"].location == home_dir


def test_configure_relative_path(runner):
    mock_result = mock.Mock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    mock_config = mock.Mock()
    mock_config.providers = {}
    projects_dict = {}
    mock_config.projects = projects_dict

    with (
        mock.patch("mlflow.assistant.cli.shutil.which", return_value="/usr/bin/claude"),
        mock.patch(
            "mlflow.assistant.providers.claude_code.subprocess.run",
            return_value=mock_result,
        ),
        mock.patch(
            "mlflow.assistant.cli._fetch_recent_experiments",
            return_value=[("1", "Test Experiment")],
        ),
        mock.patch(
            "mlflow.assistant.cli.AssistantConfig.load",
            return_value=mock_config,
        ),
        mock.patch.object(mock_config, "save"),
        mock.patch.object(mock_config, "set_provider"),
    ):
        # Use "." which should resolve to current directory
        result = runner.invoke(
            commands,
            ["--configure"],
            input="1\ny\nhttp://localhost:5000\n1\n.\ndefault\n",
        )
        assert "Setup Complete" in result.output
        # Verify the saved path is absolute, not "."
        assert "1" in projects_dict
        assert os.path.isabs(projects_dict["1"].location)
        assert projects_dict["1"].location != "."
