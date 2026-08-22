"""Unit tests for mlflow.kiro_cli.cli — Click plumbing, status, disable, enable, UV env var."""

import json
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from mlflow.kiro_cli.cli import kiro_cli
from mlflow.kiro_cli.config import (
    HOOK_FIELD_COMMAND,
    HOOK_FIELD_HOOKS,
    MLFLOW_HOOK_IDENTIFIER,
    MLFLOW_TRACING_ENABLED,
)


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# 10.1 Click plumbing — --help for autolog and autolog kiro-cli
# ---------------------------------------------------------------------------


def test_autolog_help_lists_kiro_cli(runner):
    """autolog --help lists the kiro-cli subcommand."""
    from mlflow.autolog import autolog

    result = runner.invoke(autolog, ["--help"])
    assert result.exit_code == 0
    assert "kiro-cli" in result.output


def test_kiro_cli_help_lists_expected_flags(runner):
    """kiro-cli --help shows all documented flags."""
    result = runner.invoke(kiro_cli, ["--help"])
    assert result.exit_code == 0
    assert "--directory" in result.output
    assert "--tracking-uri" in result.output
    assert "--experiment-id" in result.output
    assert "--experiment-name" in result.output
    assert "--disable" in result.output
    assert "--status" in result.output


def test_kiro_cli_help_does_not_show_hidden_subcommands(runner):
    """Hidden subcommands should not appear in --help output."""
    result = runner.invoke(kiro_cli, ["--help"])
    assert result.exit_code == 0
    assert "stop-hook" not in result.output
    assert "agent-spawn-hook" not in result.output


# ---------------------------------------------------------------------------
# 10.2 --status with no config
# ---------------------------------------------------------------------------


def test_status_no_config_prints_not_enabled(runner, tmp_path):
    """--status with no config prints the not-enabled message and exits 0."""
    result = runner.invoke(kiro_cli, ["--status", "-d", str(tmp_path)])
    assert result.exit_code == 0
    assert "❌ Kiro CLI tracing is not enabled" in result.output


def test_status_with_enabled_config(runner, tmp_path):
    """--status with enabled config prints the enabled message."""
    settings_path = tmp_path / ".kiro" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps({
            "env": {
                MLFLOW_TRACING_ENABLED: "true",
                "MLFLOW_TRACKING_URI": "sqlite:///mlflow.db",
                "MLFLOW_EXPERIMENT_ID": "42",
            }
        })
    )
    result = runner.invoke(kiro_cli, ["--status", "-d", str(tmp_path)])
    assert result.exit_code == 0
    assert "✅ Kiro CLI tracing is ENABLED" in result.output
    assert "sqlite:///mlflow.db" in result.output
    assert "42" in result.output


# ---------------------------------------------------------------------------
# 10.3 --disable with no config
# ---------------------------------------------------------------------------


def test_disable_no_config_prints_not_enabled(runner, tmp_path):
    """--disable with no config prints the not-enabled message and exits 0."""
    result = runner.invoke(kiro_cli, ["--disable", "-d", str(tmp_path)])
    assert result.exit_code == 0
    assert (
        "not enabled" in result.output.lower() or "No Kiro CLI configuration found" in result.output
    )


def test_disable_with_existing_config(runner, tmp_path):
    """--disable with existing config removes MLflow entries."""
    agent_path = tmp_path / ".kiro" / "agents" / "kiro_default.json"
    agent_path.parent.mkdir(parents=True)
    agent_path.write_text(
        json.dumps({
            "name": "kiro_default",
            "hooks": {
                "stop": [{"type": "command", "command": "mlflow autolog kiro-cli stop-hook"}]
            },
        })
    )
    settings_path = tmp_path / ".kiro" / "settings.json"
    settings_path.write_text(json.dumps({"env": {MLFLOW_TRACING_ENABLED: "true"}}))

    result = runner.invoke(kiro_cli, ["--disable", "-d", str(tmp_path)])
    assert result.exit_code == 0
    assert "✅ Kiro CLI tracing disabled" in result.output


# ---------------------------------------------------------------------------
# 10.4 Enable with/without UV env var
# ---------------------------------------------------------------------------


def _read_agent_config(tmp_path: Path) -> dict:
    agent_path = tmp_path / ".kiro" / "agents" / "kiro_default.json"
    return json.loads(agent_path.read_text())


def test_enable_with_uv_env_var(runner, tmp_path, monkeypatch):
    """When UV env var is set, hook commands use 'uv run mlflow'."""
    monkeypatch.setenv("UV", "/path/to/uv")
    result = runner.invoke(kiro_cli, ["-d", str(tmp_path)])
    assert result.exit_code == 0

    config = _read_agent_config(tmp_path)
    hooks = config[HOOK_FIELD_HOOKS]
    for event_name, entries in hooks.items():
        for entry in entries:
            cmd = entry.get(HOOK_FIELD_COMMAND, "")
            if MLFLOW_HOOK_IDENTIFIER in cmd:
                assert cmd.startswith("uv run mlflow"), f"Expected uv prefix for {event_name}"


def test_enable_without_uv_env_var(runner, tmp_path, monkeypatch):
    """Without UV env var, hook commands use plain 'mlflow'."""
    monkeypatch.delenv("UV", raising=False)
    result = runner.invoke(kiro_cli, ["-d", str(tmp_path)])
    assert result.exit_code == 0

    config = _read_agent_config(tmp_path)
    hooks = config[HOOK_FIELD_HOOKS]
    for event_name, entries in hooks.items():
        for entry in entries:
            cmd = entry.get(HOOK_FIELD_COMMAND, "")
            if MLFLOW_HOOK_IDENTIFIER in cmd:
                assert cmd.startswith("mlflow autolog"), f"Expected plain mlflow for {event_name}"
                assert not cmd.startswith("uv run")


def test_enable_writes_all_five_events(runner, tmp_path, monkeypatch):
    """Enable writes hooks for all five events."""
    monkeypatch.delenv("UV", raising=False)
    result = runner.invoke(kiro_cli, ["-d", str(tmp_path)])
    assert result.exit_code == 0

    config = _read_agent_config(tmp_path)
    hooks = config[HOOK_FIELD_HOOKS]
    expected_events = {"agentSpawn", "userPromptSubmit", "preToolUse", "postToolUse", "stop"}
    assert set(hooks.keys()) == expected_events


def test_enable_writes_settings_file(runner, tmp_path, monkeypatch):
    """Enable creates settings.json with tracing enabled."""
    monkeypatch.delenv("UV", raising=False)
    result = runner.invoke(kiro_cli, ["-d", str(tmp_path)])
    assert result.exit_code == 0

    settings_path = tmp_path / ".kiro" / "settings.json"
    assert settings_path.exists()
    config = json.loads(settings_path.read_text())
    assert config["env"][MLFLOW_TRACING_ENABLED] == "true"


# ---------------------------------------------------------------------------
# 10.5 Hidden subcommands route to correct handler
# ---------------------------------------------------------------------------


def test_stop_hook_subcommand_routes_to_handler(runner):
    """stop-hook subcommand routes to stop_hook_handler."""
    with mock.patch("mlflow.kiro_cli.cli.stop_hook_handler") as mock_handler:
        result = runner.invoke(kiro_cli, ["stop-hook"])
        assert result.exit_code == 0
        mock_handler.assert_called_once()


def test_agent_spawn_hook_subcommand_routes_to_handler(runner):
    """agent-spawn-hook subcommand routes to agent_spawn_hook_handler."""
    with mock.patch("mlflow.kiro_cli.cli.agent_spawn_hook_handler") as mock_handler:
        result = runner.invoke(kiro_cli, ["agent-spawn-hook"])
        assert result.exit_code == 0
        mock_handler.assert_called_once()


def test_user_prompt_submit_hook_subcommand_routes_to_handler(runner):
    """user-prompt-submit-hook subcommand routes to user_prompt_submit_hook_handler."""
    with mock.patch("mlflow.kiro_cli.cli.user_prompt_submit_hook_handler") as mock_handler:
        result = runner.invoke(kiro_cli, ["user-prompt-submit-hook"])
        assert result.exit_code == 0
        mock_handler.assert_called_once()


def test_pre_tool_use_hook_subcommand_routes_to_handler(runner):
    """pre-tool-use-hook subcommand routes to pre_tool_use_hook_handler."""
    with mock.patch("mlflow.kiro_cli.cli.pre_tool_use_hook_handler") as mock_handler:
        result = runner.invoke(kiro_cli, ["pre-tool-use-hook"])
        assert result.exit_code == 0
        mock_handler.assert_called_once()


def test_post_tool_use_hook_subcommand_routes_to_handler(runner):
    """post-tool-use-hook subcommand routes to post_tool_use_hook_handler."""
    with mock.patch("mlflow.kiro_cli.cli.post_tool_use_hook_handler") as mock_handler:
        result = runner.invoke(kiro_cli, ["post-tool-use-hook"])
        assert result.exit_code == 0
        mock_handler.assert_called_once()


# ---------------------------------------------------------------------------
# 10.6 Conflict resolution
# ---------------------------------------------------------------------------


def test_status_and_disable_together_leaves_disk_untouched(runner, tmp_path):
    """--status --disable: status wins, disk is not modified."""
    agent_path = tmp_path / ".kiro" / "agents" / "kiro_default.json"
    agent_path.parent.mkdir(parents=True)
    agent_config = {
        "name": "kiro_default",
        "hooks": {"stop": [{"type": "command", "command": "mlflow autolog kiro-cli stop-hook"}]},
    }
    agent_path.write_text(json.dumps(agent_config))
    settings_path = tmp_path / ".kiro" / "settings.json"
    settings_path.write_text(json.dumps({"env": {MLFLOW_TRACING_ENABLED: "true"}}))

    # Capture file contents before
    agent_before = agent_path.read_text()
    settings_before = settings_path.read_text()

    result = runner.invoke(kiro_cli, ["--status", "--disable", "-d", str(tmp_path)])
    assert result.exit_code == 0
    # Status output should appear
    assert "tracing" in result.output.lower()

    # Files should be untouched
    assert agent_path.read_text() == agent_before
    assert settings_path.read_text() == settings_before


def test_experiment_id_wins_over_name(runner, tmp_path, monkeypatch):
    """-e wins over -n: experiment_name is not stored when both are given."""
    monkeypatch.delenv("UV", raising=False)
    result = runner.invoke(kiro_cli, ["-d", str(tmp_path), "-e", "42", "-n", "my-experiment"])
    assert result.exit_code == 0

    settings_path = tmp_path / ".kiro" / "settings.json"
    config = json.loads(settings_path.read_text())
    env = config["env"]
    assert env.get("MLFLOW_EXPERIMENT_ID") == "42"
    assert "MLFLOW_EXPERIMENT_NAME" not in env
