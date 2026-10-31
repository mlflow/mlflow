"""Tests for mlflow.kiro.hooks."""

import json
import os
from unittest import mock

from mlflow.kiro.hooks import stop_hook_handler


def test_stop_hook_handler_uses_session_cwd_for_config_lookup(tmp_path, monkeypatch, capsys):
    env_dir = tmp_path / ".kiro"
    env_dir.mkdir()
    (env_dir / "mlflow_env.json").write_text(
        json.dumps({"MLFLOW_KIRO_TRACING_ENABLED": "true"}),
        encoding="utf-8",
    )

    original_cwd = os.getcwd()
    monkeypatch.chdir(tmp_path.parent)

    with (
        mock.patch("mlflow.kiro.hooks.read_hook_input", return_value={"cwd": str(tmp_path)}),
        mock.patch("mlflow.kiro.hooks.setup_mlflow") as mock_setup_mlflow,
        mock.patch("mlflow.kiro.hooks.process_session", return_value=object()) as mock_process,
    ):
        stop_hook_handler()

    assert os.getcwd() == str(tmp_path)
    assert capsys.readouterr().out == '{"continue": true}\n'
    mock_setup_mlflow.assert_called_once()
    mock_process.assert_called_once_with({"cwd": str(tmp_path)})
    os.chdir(original_cwd)
