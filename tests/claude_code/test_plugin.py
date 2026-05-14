import json
import subprocess
from pathlib import Path
from unittest import mock

import click
import pytest

from mlflow.claude_code.plugin import disable_tracing_plugin, ensure_plugin_installed, migrate_legacy_hooks


def test_migrate_legacy_hooks_removes_only_mlflow_hook(tmp_path):
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "hooks": {
                    "Stop": [
                        {
                            "hooks": [
                                {"type": "command", "command": "mlflow autolog claude stop-hook"},
                                {"type": "command", "command": "echo keep-me"},
                            ]
                        }
                    ]
                }
            }
        )
    )

    assert migrate_legacy_hooks(settings_path) is True
    config = json.loads(settings_path.read_text())
    hooks = config["hooks"]["Stop"][0]["hooks"]
    assert hooks == [{"type": "command", "command": "echo keep-me"}]


def test_disable_tracing_plugin_removes_env_and_legacy_hooks(tmp_path):
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "env": {
                    "MLFLOW_CLAUDE_TRACING_ENABLED": "true",
                    "MLFLOW_TRACKING_URI": "http://localhost:5000",
                    "MLFLOW_EXPERIMENT_ID": "123",
                },
                "hooks": {
                    "Stop": [
                        {
                            "hooks": [
                                {"type": "command", "command": "mlflow autolog claude stop-hook"}
                            ]
                        }
                    ]
                },
            }
        )
    )

    assert disable_tracing_plugin(settings_path) is True
    assert not settings_path.exists()


def test_ensure_plugin_installed_runs_marketplace_add_and_install(tmp_path):
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    with (
        mock.patch("shutil.which", return_value="/usr/local/bin/claude"),
        mock.patch("subprocess.run", return_value=completed) as mock_run,
    ):
        ensure_plugin_installed(tmp_path)

    assert mock_run.call_count == 2
    first_command = mock_run.call_args_list[0].args[0]
    second_command = mock_run.call_args_list[1].args[0]
    assert first_command[:5] == ["claude", "plugin", "marketplace", "add", "mlflow/mlflow"]
    assert second_command[:4] == ["claude", "plugin", "install", "mlflow-tracing@mlflow-plugins"]


def test_ensure_plugin_installed_requires_claude_binary(tmp_path):
    with mock.patch("shutil.which", return_value=None):
        with pytest.raises(click.ClickException, match="Claude Code CLI"):
            ensure_plugin_installed(tmp_path)
