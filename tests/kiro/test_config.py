"""Tests for mlflow.kiro.config."""

import json
from pathlib import Path

import pytest

from mlflow.kiro.config import (
    KIRO_HOOK_EVENT_AGENT_STOP,
    MLFLOW_HOOK_FILE,
    MLFLOW_HOOK_NAME,
    MLFLOW_TRACING_ENABLED,
    TracingStatus,
    disable_tracing_hooks,
    get_tracing_status,
    load_kiro_env,
    save_kiro_env,
    setup_environment_config,
    setup_hooks_config,
)


# ---------------------------------------------------------------------------
# load_kiro_env / save_kiro_env
# ---------------------------------------------------------------------------


def test_load_kiro_env_missing_file(tmp_path):
    env = load_kiro_env(tmp_path / "missing.json")
    assert env == {}


def test_save_and_load_kiro_env(tmp_path):
    env_path = tmp_path / "mlflow_env.json"
    save_kiro_env(env_path, {"MLFLOW_TRACKING_URI": "sqlite:///test.db"})
    loaded = load_kiro_env(env_path)
    assert loaded["MLFLOW_TRACKING_URI"] == "sqlite:///test.db"


def test_load_kiro_env_invalid_json(tmp_path):
    env_path = tmp_path / "bad.json"
    env_path.write_text("not valid json")
    result = load_kiro_env(env_path)
    assert result == {}


# ---------------------------------------------------------------------------
# setup_hooks_config
# ---------------------------------------------------------------------------


def test_setup_hooks_config_creates_file(tmp_path):
    hooks_dir = tmp_path / ".kiro" / "hooks"
    setup_hooks_config(hooks_dir)
    hook_file = hooks_dir / MLFLOW_HOOK_FILE
    assert hook_file.exists()
    payload = json.loads(hook_file.read_text())
    assert payload["version"] == "1.0"
    hooks = payload["hooks"]
    assert len(hooks) == 1
    assert hooks[0]["name"] == MLFLOW_HOOK_NAME
    assert hooks[0]["event"] == KIRO_HOOK_EVENT_AGENT_STOP
    assert hooks[0]["enabled"] is True
    command = hooks[0]["actions"][0]["command"]
    assert "mlflow autolog kiro stop-hook" in command


def test_setup_hooks_config_uses_uv_when_env_set(tmp_path, monkeypatch):
    monkeypatch.setenv("UV", "/usr/local/bin/uv")
    hooks_dir = tmp_path / ".kiro" / "hooks"
    setup_hooks_config(hooks_dir)
    hook_file = hooks_dir / MLFLOW_HOOK_FILE
    payload = json.loads(hook_file.read_text())
    command = payload["hooks"][0]["actions"][0]["command"]
    assert command.startswith("uv run mlflow")


def test_setup_hooks_config_without_uv(tmp_path, monkeypatch):
    monkeypatch.delenv("UV", raising=False)
    hooks_dir = tmp_path / ".kiro" / "hooks"
    setup_hooks_config(hooks_dir)
    hook_file = hooks_dir / MLFLOW_HOOK_FILE
    payload = json.loads(hook_file.read_text())
    command = payload["hooks"][0]["actions"][0]["command"]
    assert command.startswith("mlflow")
    assert "uv" not in command


def test_setup_hooks_config_overwrites_existing(tmp_path):
    hooks_dir = tmp_path / ".kiro" / "hooks"
    hooks_dir.mkdir(parents=True)
    hook_file = hooks_dir / MLFLOW_HOOK_FILE
    hook_file.write_text("stale content")
    setup_hooks_config(hooks_dir)
    payload = json.loads(hook_file.read_text())
    assert payload["version"] == "1.0"  # valid JSON, not stale


# ---------------------------------------------------------------------------
# setup_environment_config
# ---------------------------------------------------------------------------


def test_setup_environment_config_all_params(tmp_path):
    env_path = tmp_path / "mlflow_env.json"
    setup_environment_config(
        env_path,
        tracking_uri="sqlite:///test.db",
        experiment_name="My Experiment",
    )
    env = load_kiro_env(env_path)
    assert env[MLFLOW_TRACING_ENABLED] == "true"
    assert env["MLFLOW_TRACKING_URI"] == "sqlite:///test.db"
    assert env["MLFLOW_EXPERIMENT_NAME"] == "My Experiment"


def test_setup_environment_config_experiment_id_overrides_name(tmp_path):
    env_path = tmp_path / "mlflow_env.json"
    setup_environment_config(
        env_path,
        experiment_id="123456",
        experiment_name="should_be_removed",
    )
    env = load_kiro_env(env_path)
    assert env["MLFLOW_EXPERIMENT_ID"] == "123456"
    assert "MLFLOW_EXPERIMENT_NAME" not in env


def test_setup_environment_config_no_tracking_uri(tmp_path):
    env_path = tmp_path / "mlflow_env.json"
    setup_environment_config(env_path)
    env = load_kiro_env(env_path)
    assert env[MLFLOW_TRACING_ENABLED] == "true"
    assert "MLFLOW_TRACKING_URI" not in env


# ---------------------------------------------------------------------------
# disable_tracing_hooks
# ---------------------------------------------------------------------------


def test_disable_tracing_hooks_removes_files(tmp_path):
    hooks_dir = tmp_path / ".kiro" / "hooks"
    hooks_dir.mkdir(parents=True)
    hook_file = hooks_dir / MLFLOW_HOOK_FILE
    hook_file.write_text("{}")
    env_path = tmp_path / "mlflow_env.json"
    env_path.write_text("{}")

    removed = disable_tracing_hooks(hooks_dir, env_path)
    assert removed is True
    assert not hook_file.exists()
    assert not env_path.exists()


def test_disable_tracing_hooks_when_not_configured(tmp_path):
    hooks_dir = tmp_path / ".kiro" / "hooks"
    env_path = tmp_path / "mlflow_env.json"
    removed = disable_tracing_hooks(hooks_dir, env_path)
    assert removed is False


# ---------------------------------------------------------------------------
# get_tracing_status
# ---------------------------------------------------------------------------


def test_get_tracing_status_not_configured(tmp_path):
    hooks_dir = tmp_path / ".kiro" / "hooks"
    env_path = tmp_path / "mlflow_env.json"
    status = get_tracing_status(hooks_dir, env_path)
    assert status.enabled is False


def test_get_tracing_status_enabled(tmp_path):
    hooks_dir = tmp_path / ".kiro" / "hooks"
    setup_hooks_config(hooks_dir)
    env_path = tmp_path / "mlflow_env.json"
    setup_environment_config(
        env_path, tracking_uri="sqlite:///test.db", experiment_name="My Exp"
    )
    status = get_tracing_status(hooks_dir, env_path)
    assert status.enabled is True
    assert status.tracking_uri == "sqlite:///test.db"
    assert status.experiment_name == "My Exp"
