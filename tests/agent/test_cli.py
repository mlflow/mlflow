from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from mlflow.agent.setup.cli import setup


@pytest.fixture
def tmp_git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_setup_local_server_path(tmp_git_repo: Path):
    with (
        mock.patch(
            "mlflow.agent.agents.shutil.which", return_value="/usr/local/bin/claude"
        ) as mock_which,
        mock.patch("mlflow.agent.setup.cli._find_available_port", return_value=5050) as mock_port,
    ):
        result = CliRunner().invoke(setup, ["--agent", "claude", "--print"], input="y\n\n")
    assert result.exit_code == 0, result.stderr
    assert "Picked local tracking URI: http://127.0.0.1:5050" in result.stderr
    assert "mlflow server --host 127.0.0.1 --port 5050" in result.stdout
    assert "Start a local MLflow tracking server" in result.stdout
    mock_which.assert_called()
    mock_port.assert_called_once()


def test_setup_user_provided_uri(tmp_git_repo: Path):
    with mock.patch(
        "mlflow.agent.agents.shutil.which", return_value="/usr/local/bin/claude"
    ) as mock_which:
        result = CliRunner().invoke(
            setup, ["--agent", "claude", "--print"], input="y\nhttp://localhost:5001\n"
        )
    assert result.exit_code == 0, result.stderr
    assert "Start a local MLflow tracking server" not in result.stdout
    assert "MLFLOW_TRACKING_URI=http://localhost:5001" in result.stdout
    mock_which.assert_called()


def test_setup_declined_skills_uses_bundled_path(tmp_git_repo: Path):
    with mock.patch(
        "mlflow.agent.agents.shutil.which", return_value="/usr/local/bin/claude"
    ) as mock_which:
        result = CliRunner().invoke(
            setup, ["--agent", "claude", "--print"], input="n\nhttp://localhost:5001\n"
        )
    assert result.exit_code == 0, result.stderr
    assert "Skipping skill installation." in result.stderr
    assert "MLflow skills are bundled at" in result.stdout
    assert "/mlflow/assistant/skills" in result.stdout
    assert not (tmp_git_repo / ".claude").exists()
    mock_which.assert_called()


def test_setup_requested_agent_not_installed(tmp_git_repo: Path):
    with mock.patch("mlflow.agent.agents.shutil.which", return_value=None) as mock_which:
        result = CliRunner().invoke(setup, ["--agent", "claude", "--print"])
    assert result.exit_code != 0
    assert "not found on PATH" in result.stderr
    mock_which.assert_called()


def test_setup_databricks_prompts_for_workspace_path(tmp_git_repo: Path):
    with mock.patch(
        "mlflow.agent.agents.shutil.which", return_value="/usr/local/bin/claude"
    ) as mock_which:
        result = CliRunner().invoke(
            setup,
            ["--agent", "claude", "--print"],
            input="y\ndatabricks\n/Users/me@example.com/my-app\n",
        )
    assert result.exit_code == 0, result.stderr
    assert "Workspace experiment path" in result.stderr
    assert "Configure the Databricks workspace" in result.stdout
    assert "MLFLOW_TRACKING_URI=databricks" in result.stdout
    assert "MLFLOW_REGISTRY_URI=databricks-uc" in result.stdout
    assert 'mlflow.set_experiment("/Users/me@example.com/my-app")' in result.stdout
    assert "Start a local MLflow tracking server" not in result.stdout
    mock_which.assert_called()


def test_setup_databricks_rejects_non_absolute_experiment_path(tmp_git_repo: Path):
    with mock.patch(
        "mlflow.agent.agents.shutil.which", return_value="/usr/local/bin/claude"
    ) as mock_which:
        result = CliRunner().invoke(
            setup,
            ["--agent", "claude", "--print"],
            input="y\ndatabricks\nmy-app\n",
        )
    assert result.exit_code != 0
    assert "must start with '/'" in result.stderr
    mock_which.assert_called()
