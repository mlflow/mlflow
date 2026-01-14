import os
from unittest import mock

import pytest
from click.testing import CliRunner

from mlflow.assistant.cli import commands, resolve_cli_options


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


def test_resolve_cli_options_experiment_requires_project_path():
    with mock.patch("mlflow.get_tracking_uri", return_value="http://localhost:5000"):
        error, _, _, _ = resolve_cli_options(
            tracking_uri=None, experiment_id="123", project_path=None
        )
        assert error == "--project-path is required when --experiment-id is specified."


def test_resolve_cli_options_project_path_requires_experiment():
    with (
        mock.patch("mlflow.get_tracking_uri", return_value="http://localhost:5000"),
        mock.patch.dict(os.environ, {}, clear=True),
    ):
        error, _, _, _ = resolve_cli_options(
            tracking_uri=None, experiment_id=None, project_path="/tmp"
        )
        assert "--experiment-id is required when --project-path is specified" in error


def test_resolve_cli_options_project_path_requires_tracking_uri():
    with (
        mock.patch("mlflow.get_tracking_uri", return_value=None),
        mock.patch.dict(os.environ, {"MLFLOW_EXPERIMENT_ID": "123"}, clear=True),
    ):
        error, _, _, _ = resolve_cli_options(
            tracking_uri=None, experiment_id=None, project_path="/tmp"
        )
        assert "--tracking-uri is required" in error


def test_resolve_cli_options_valid_with_all_options():
    error, uri, exp_id, path = resolve_cli_options(
        tracking_uri="http://localhost:5000", experiment_id="123", project_path="/tmp"
    )
    assert error is None
    assert uri == "http://localhost:5000"
    assert exp_id == "123"
    assert path == "/tmp"


def test_resolve_cli_options_no_experiment():
    with mock.patch("mlflow.get_tracking_uri", return_value="http://default:5000"):
        error, uri, exp_id, path = resolve_cli_options(
            tracking_uri=None, experiment_id=None, project_path=None
        )
        assert error is None
        assert uri == "http://default:5000"
        assert exp_id is None
        assert path is None


def test_resolve_cli_options_resolves_tracking_uri_from_env():
    with mock.patch("mlflow.get_tracking_uri", return_value="http://env-server:5000"):
        error, uri, exp_id, _ = resolve_cli_options(
            tracking_uri=None, experiment_id="123", project_path="/tmp"
        )
        assert error is None
        assert uri == "http://env-server:5000"
        assert exp_id == "123"


def test_resolve_cli_options_resolves_experiment_id_from_env():
    with (
        mock.patch("mlflow.get_tracking_uri", return_value="http://localhost:5000"),
        mock.patch.dict(os.environ, {"MLFLOW_EXPERIMENT_ID": "456"}, clear=True),
    ):
        error, uri, exp_id, path = resolve_cli_options(
            tracking_uri=None, experiment_id=None, project_path="/tmp"
        )
        assert error is None
        assert uri == "http://localhost:5000"
        assert exp_id == "456"
        assert path == "/tmp"


def test_experiment_id_requires_project_path(runner):
    with mock.patch("mlflow.get_tracking_uri", return_value="http://localhost:5000"):
        result = runner.invoke(commands, ["--configure", "--experiment-id", "123"])
        assert result.exit_code == 0
        assert "--project-path is required" in result.output


def test_project_path_requires_experiment_id(runner):
    with (
        mock.patch("mlflow.get_tracking_uri", return_value="http://localhost:5000"),
        mock.patch.dict(os.environ, {}, clear=True),
    ):
        result = runner.invoke(commands, ["--configure", "--project-path", "/tmp"])
        assert result.exit_code == 0
        assert "--experiment-id is required" in result.output


def test_non_interactive_with_provider(runner):
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
            "mlflow.assistant.cli.AssistantConfig.load",
            return_value=mock_config,
        ),
        mock.patch.object(mock_config, "save"),
        mock.patch.object(mock_config, "set_provider"),
    ):
        result = runner.invoke(
            commands,
            ["--configure", "--provider", "claude_code", "--model", "default"],
        )
        assert "configured successfully" in result.output
        mock_config.set_provider.assert_called_once_with("claude_code", "default")


def test_non_interactive_with_experiment(runner, tmp_path):
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
            "mlflow.assistant.cli.AssistantConfig.load",
            return_value=mock_config,
        ),
        mock.patch.object(mock_config, "save"),
        mock.patch.object(mock_config, "set_provider"),
    ):
        result = runner.invoke(
            commands,
            [
                "--configure",
                "--provider",
                "claude_code",
                "--experiment-id",
                "123",
                "--project-path",
                str(tmp_path),
            ],
        )
        assert "configured successfully" in result.output
        assert "123" in projects_dict
        assert projects_dict["123"].location == str(tmp_path)


def test_non_interactive_tracking_uri_from_env(runner, tmp_path):
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
            "mlflow.assistant.cli.AssistantConfig.load",
            return_value=mock_config,
        ),
        mock.patch.object(mock_config, "save"),
        mock.patch.object(mock_config, "set_provider"),
        mock.patch("mlflow.get_tracking_uri", return_value="http://env-tracking:5000"),
    ):
        # Without --tracking-uri, should use mlflow.get_tracking_uri()
        result = runner.invoke(
            commands,
            [
                "--configure",
                "--provider",
                "claude_code",
                "--experiment-id",
                "123",
                "--project-path",
                str(tmp_path),
            ],
        )
        assert "configured successfully" in result.output


def test_non_interactive_invalid_project_path(runner):
    mock_result = mock.Mock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    with (
        mock.patch("mlflow.assistant.cli.shutil.which", return_value="/usr/bin/claude"),
        mock.patch(
            "mlflow.assistant.providers.claude_code.subprocess.run",
            return_value=mock_result,
        ),
    ):
        result = runner.invoke(
            commands,
            [
                "--configure",
                "--provider",
                "claude_code",
                "--experiment-id",
                "123",
                "--project-path",
                "/nonexistent/path/does/not/exist",
            ],
        )
        assert "does not exist" in result.output
