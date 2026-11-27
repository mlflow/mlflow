"""Tests for mlflow.kiro.cli."""

import json
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from mlflow.kiro.cli import commands
from mlflow.kiro.config import (
    KIRO_ENV_FILE,
    KIRO_HOOKS_DIR,
    MLFLOW_HOOK_FILE,
    MLFLOW_TRACING_ENABLED,
    load_kiro_env,
)


@pytest.fixture()
def runner():
    return CliRunner()


def _hook_command_from_dir(directory: Path) -> str:
    hook_file = directory / KIRO_HOOKS_DIR / MLFLOW_HOOK_FILE
    payload = json.loads(hook_file.read_text())
    return payload["hooks"][0]["actions"][0]["command"]


# ---------------------------------------------------------------------------
# Top-level group
# ---------------------------------------------------------------------------


def test_autolog_help(runner):
    result = runner.invoke(commands, ["--help"])
    assert result.exit_code == 0
    assert "Commands for autologging with MLflow" in result.output


def test_kiro_help(runner):
    result = runner.invoke(commands, ["kiro", "--help"])
    assert result.exit_code == 0
    assert "--tracking-uri" in result.output
    assert "--experiment-id" in result.output
    assert "--disable" in result.output
    assert "--status" in result.output


# ---------------------------------------------------------------------------
# Setup (default invocation)
# ---------------------------------------------------------------------------


def test_kiro_setup_creates_hook_file(runner, monkeypatch):
    monkeypatch.delenv("UV", raising=False)
    with runner.isolated_filesystem():
        result = runner.invoke(commands, ["kiro"])
        assert result.exit_code == 0, result.output
        command = _hook_command_from_dir(Path("."))
        assert "mlflow autolog kiro stop-hook" in command


def test_kiro_setup_uses_uv_when_env_set(runner, monkeypatch):
    monkeypatch.setenv("UV", "/usr/local/bin/uv")
    with runner.isolated_filesystem():
        result = runner.invoke(commands, ["kiro"])
        assert result.exit_code == 0
        command = _hook_command_from_dir(Path("."))
        assert command.startswith("uv run mlflow")


def test_kiro_setup_with_tracking_uri(runner, monkeypatch):
    monkeypatch.delenv("UV", raising=False)
    with runner.isolated_filesystem():
        result = runner.invoke(commands, ["kiro", "--tracking-uri", "sqlite:///test.db"])
        assert result.exit_code == 0
        env = load_kiro_env(Path(KIRO_ENV_FILE))
        assert env["MLFLOW_TRACKING_URI"] == "sqlite:///test.db"


def test_kiro_setup_with_experiment_name(runner, monkeypatch):
    monkeypatch.delenv("UV", raising=False)
    with runner.isolated_filesystem():
        result = runner.invoke(commands, ["kiro", "--experiment-name", "My Kiro Exp"])
        assert result.exit_code == 0
        env = load_kiro_env(Path(KIRO_ENV_FILE))
        assert env["MLFLOW_EXPERIMENT_NAME"] == "My Kiro Exp"


def test_kiro_setup_with_experiment_id(runner, monkeypatch):
    monkeypatch.delenv("UV", raising=False)
    with runner.isolated_filesystem():
        result = runner.invoke(commands, ["kiro", "--experiment-id", "99999"])
        assert result.exit_code == 0
        env = load_kiro_env(Path(KIRO_ENV_FILE))
        assert env["MLFLOW_EXPERIMENT_ID"] == "99999"
        assert "MLFLOW_EXPERIMENT_NAME" not in env


def test_kiro_setup_with_directory_flag(runner, tmp_path, monkeypatch):
    monkeypatch.delenv("UV", raising=False)
    result = runner.invoke(commands, ["kiro", "--directory", str(tmp_path)])
    assert result.exit_code == 0
    hook_file = tmp_path / KIRO_HOOKS_DIR / MLFLOW_HOOK_FILE
    assert hook_file.exists()


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def test_kiro_status_not_configured(runner):
    with runner.isolated_filesystem():
        result = runner.invoke(commands, ["kiro", "--status"])
        assert result.exit_code == 0
        assert "❌" in result.output


def test_kiro_status_configured(runner, monkeypatch):
    monkeypatch.delenv("UV", raising=False)
    with runner.isolated_filesystem():
        runner.invoke(commands, ["kiro", "--tracking-uri", "sqlite:///test.db"])
        result = runner.invoke(commands, ["kiro", "--status"])
        assert result.exit_code == 0
        assert "✅" in result.output
        assert "sqlite:///test.db" in result.output


# ---------------------------------------------------------------------------
# Disable
# ---------------------------------------------------------------------------


def test_kiro_disable_when_not_configured(runner):
    with runner.isolated_filesystem():
        result = runner.invoke(commands, ["kiro", "--disable"])
        assert result.exit_code == 0
        assert "❌" in result.output


def test_kiro_disable_when_configured(runner, monkeypatch):
    monkeypatch.delenv("UV", raising=False)
    with runner.isolated_filesystem():
        runner.invoke(commands, ["kiro"])
        result = runner.invoke(commands, ["kiro", "--disable"])
        assert result.exit_code == 0
        assert "✅" in result.output
        # Hook file should be gone
        assert not (Path(".") / KIRO_HOOKS_DIR / MLFLOW_HOOK_FILE).exists()


# ---------------------------------------------------------------------------
# stop-hook subcommand
# ---------------------------------------------------------------------------


def test_stop_hook_subcommand_is_routable(runner):
    with mock.patch("mlflow.kiro.cli.stop_hook_handler") as mock_handler:
        result = runner.invoke(commands, ["kiro", "stop-hook"])
        assert result.exit_code == 0
        mock_handler.assert_called_once()
