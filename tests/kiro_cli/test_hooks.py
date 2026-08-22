"""Unit tests for mlflow.kiro_cli.hooks — upsert, setup, disable, and hook handlers."""

import json
from pathlib import Path
from unittest import mock

import pytest

from mlflow.kiro_cli.config import (
    HOOK_FIELD_COMMAND,
    HOOK_FIELD_HOOKS,
    MLFLOW_HOOK_IDENTIFIER,
    MLFLOW_TRACING_ENABLED,
)
from mlflow.kiro_cli.hooks import (
    HOOK_EVENTS,
    agent_spawn_hook_handler,
    disable_tracing_hooks,
    post_tool_use_hook_handler,
    pre_tool_use_hook_handler,
    setup_hooks_config,
    stop_hook_handler,
    upsert_hook,
    user_prompt_submit_hook_handler,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


@pytest.fixture
def agent_config_path(tmp_path):
    path = tmp_path / ".kiro" / "agents" / "kiro_default.json"
    path.parent.mkdir(parents=True)
    return path


@pytest.fixture
def settings_path(tmp_path):
    path = tmp_path / ".kiro" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 11.1 upsert_hook on flat Kiro shape
# ---------------------------------------------------------------------------


def test_upsert_hook_adds_new_entry(monkeypatch):
    """upsert_hook adds a new entry when none exists."""
    monkeypatch.delenv("UV", raising=False)
    config = {HOOK_FIELD_HOOKS: {}}
    upsert_hook(config, "stop", "stop-hook")

    entries = config[HOOK_FIELD_HOOKS]["stop"]
    assert len(entries) == 1
    assert "mlflow autolog kiro-cli stop-hook" in entries[0][HOOK_FIELD_COMMAND]


def test_upsert_hook_updates_existing_mlflow_entry(monkeypatch):
    """upsert_hook updates an existing MLflow entry in place."""
    monkeypatch.delenv("UV", raising=False)
    config = {
        HOOK_FIELD_HOOKS: {
            "stop": [
                {"type": "command", HOOK_FIELD_COMMAND: "mlflow autolog kiro-cli old-stop-hook"},
            ]
        }
    }
    upsert_hook(config, "stop", "stop-hook")

    entries = config[HOOK_FIELD_HOOKS]["stop"]
    assert len(entries) == 1
    assert entries[0][HOOK_FIELD_COMMAND] == "mlflow autolog kiro-cli stop-hook"


def test_upsert_hook_preserves_user_entries(monkeypatch):
    """upsert_hook preserves user entries and their order."""
    monkeypatch.delenv("UV", raising=False)
    user_entry = {"type": "command", HOOK_FIELD_COMMAND: "~/my-custom-hook.sh"}
    config = {
        HOOK_FIELD_HOOKS: {
            "preToolUse": [user_entry.copy()],
        }
    }
    upsert_hook(config, "preToolUse", "pre-tool-use-hook")

    entries = config[HOOK_FIELD_HOOKS]["preToolUse"]
    assert len(entries) == 2
    assert entries[0][HOOK_FIELD_COMMAND] == "~/my-custom-hook.sh"
    assert MLFLOW_HOOK_IDENTIFIER in entries[1][HOOK_FIELD_COMMAND]


def test_upsert_hook_creates_hooks_key_if_missing(monkeypatch):
    """upsert_hook creates the hooks key if it doesn't exist."""
    monkeypatch.delenv("UV", raising=False)
    config = {}
    upsert_hook(config, "stop", "stop-hook")
    assert HOOK_FIELD_HOOKS in config
    assert "stop" in config[HOOK_FIELD_HOOKS]


# ---------------------------------------------------------------------------
# 11.2 setup_hooks_config is idempotent
# ---------------------------------------------------------------------------


def test_setup_hooks_config_idempotent(agent_config_path, monkeypatch):
    """Two calls to setup_hooks_config produce byte-identical files."""
    monkeypatch.delenv("UV", raising=False)
    setup_hooks_config(agent_config_path)
    first_content = agent_config_path.read_text()

    setup_hooks_config(agent_config_path)
    second_content = agent_config_path.read_text()

    assert first_content == second_content


def test_setup_hooks_config_creates_all_five_events(agent_config_path, monkeypatch):
    """setup_hooks_config creates entries for all five hook events."""
    monkeypatch.delenv("UV", raising=False)
    setup_hooks_config(agent_config_path)

    config = json.loads(agent_config_path.read_text())
    hooks = config[HOOK_FIELD_HOOKS]
    for event in HOOK_EVENTS:
        assert event in hooks, f"Missing event: {event}"
        mlflow_entries = [
            e for e in hooks[event] if MLFLOW_HOOK_IDENTIFIER in e.get(HOOK_FIELD_COMMAND, "")
        ]
        assert len(mlflow_entries) == 1, f"Expected exactly 1 MLflow entry for {event}"


# ---------------------------------------------------------------------------
# 11.3 setup_hooks_config preserves unrelated keys
# ---------------------------------------------------------------------------


def test_setup_hooks_config_preserves_top_level_keys(agent_config_path, monkeypatch):
    """setup_hooks_config preserves unrelated top-level keys like 'description'."""
    monkeypatch.delenv("UV", raising=False)
    existing = {
        "name": "kiro_default",
        "description": "my custom agent",
        "some_other_key": [1, 2, 3],
    }
    agent_config_path.write_text(json.dumps(existing))

    setup_hooks_config(agent_config_path)

    config = json.loads(agent_config_path.read_text())
    assert config["description"] == "my custom agent"
    assert config["some_other_key"] == [1, 2, 3]


def test_setup_hooks_config_preserves_unrelated_hook_events(agent_config_path, monkeypatch):
    """setup_hooks_config preserves user hooks under unrelated event keys."""
    monkeypatch.delenv("UV", raising=False)
    existing = {
        "name": "kiro_default",
        HOOK_FIELD_HOOKS: {
            "customEvent": [{"type": "command", HOOK_FIELD_COMMAND: "echo custom"}],
            "preToolUse": [{"type": "command", HOOK_FIELD_COMMAND: "~/my-check.sh"}],
        },
    }
    agent_config_path.write_text(json.dumps(existing))

    setup_hooks_config(agent_config_path)

    config = json.loads(agent_config_path.read_text())
    hooks = config[HOOK_FIELD_HOOKS]
    # Custom event preserved
    assert "customEvent" in hooks
    assert hooks["customEvent"][0][HOOK_FIELD_COMMAND] == "echo custom"
    # User hook in preToolUse preserved alongside MLflow entry
    pre_tool_cmds = [e[HOOK_FIELD_COMMAND] for e in hooks["preToolUse"]]
    assert "~/my-check.sh" in pre_tool_cmds


# ---------------------------------------------------------------------------
# 11.4 disable_tracing_hooks
# ---------------------------------------------------------------------------


def test_disable_removes_only_mlflow_entries(agent_config_path, settings_path, monkeypatch):
    """disable_tracing_hooks removes only MLflow entries, preserves user entries."""
    monkeypatch.delenv("UV", raising=False)
    agent_config = {
        "name": "kiro_default",
        "description": "my agent",
        HOOK_FIELD_HOOKS: {
            "preToolUse": [
                {"type": "command", HOOK_FIELD_COMMAND: "~/my-check.sh"},
                {
                    "type": "command",
                    HOOK_FIELD_COMMAND: "mlflow autolog kiro-cli pre-tool-use-hook",
                },
            ],
            "stop": [
                {"type": "command", HOOK_FIELD_COMMAND: "mlflow autolog kiro-cli stop-hook"},
            ],
        },
    }
    agent_config_path.write_text(json.dumps(agent_config))
    settings_path.write_text(json.dumps({"env": {MLFLOW_TRACING_ENABLED: "true"}}))

    result = disable_tracing_hooks(agent_config_path, settings_path)
    assert result is True

    config = json.loads(agent_config_path.read_text())
    hooks = config[HOOK_FIELD_HOOKS]
    # User hook preserved
    assert len(hooks["preToolUse"]) == 1
    assert hooks["preToolUse"][0][HOOK_FIELD_COMMAND] == "~/my-check.sh"
    # stop event removed (was only MLflow)
    assert "stop" not in hooks


def test_disable_cleans_up_empty_hooks(agent_config_path, settings_path, monkeypatch):
    """disable_tracing_hooks deletes empty event lists and empty hooks object."""
    monkeypatch.delenv("UV", raising=False)
    agent_config = {
        "name": "kiro_default",
        HOOK_FIELD_HOOKS: {
            "stop": [
                {"type": "command", HOOK_FIELD_COMMAND: "mlflow autolog kiro-cli stop-hook"},
            ],
        },
    }
    agent_config_path.write_text(json.dumps(agent_config))
    settings_path.write_text(json.dumps({"env": {MLFLOW_TRACING_ENABLED: "true"}}))

    disable_tracing_hooks(agent_config_path, settings_path)

    # Agent config should be unlinked (only name remains)
    assert not agent_config_path.exists()


def test_disable_unlinks_settings_when_empty(agent_config_path, settings_path, monkeypatch):
    """disable_tracing_hooks unlinks settings file when no keys remain."""
    monkeypatch.delenv("UV", raising=False)
    settings_path.write_text(json.dumps({"env": {MLFLOW_TRACING_ENABLED: "true"}}))

    disable_tracing_hooks(agent_config_path, settings_path)

    assert not settings_path.exists()


def test_disable_preserves_user_settings_keys(agent_config_path, settings_path, monkeypatch):
    """disable_tracing_hooks preserves user keys in settings.json."""
    monkeypatch.delenv("UV", raising=False)
    settings_path.write_text(
        json.dumps({
            "some_user_setting": "value",
            "env": {
                MLFLOW_TRACING_ENABLED: "true",
                "MY_CUSTOM_VAR": "keep_me",
            },
        })
    )

    disable_tracing_hooks(agent_config_path, settings_path)

    config = json.loads(settings_path.read_text())
    assert config["some_user_setting"] == "value"
    assert config["env"]["MY_CUSTOM_VAR"] == "keep_me"
    assert MLFLOW_TRACING_ENABLED not in config["env"]


def test_disable_returns_false_when_nothing_to_remove(agent_config_path, settings_path):
    """disable_tracing_hooks returns False when no MLflow state exists."""
    result = disable_tracing_hooks(agent_config_path, settings_path)
    assert result is False


# ---------------------------------------------------------------------------
# 11.5 Hook handlers
# ---------------------------------------------------------------------------


class TestTransientHandlerTracingOff:
    """Tracing-off fast-path: emits {"continue": true}, exit 0, no MLflow import."""

    def _run_handler(self, handler_fn, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv(MLFLOW_TRACING_ENABLED, "false")
        monkeypatch.chdir(tmp_path)
        handler_fn()
        captured = capsys.readouterr()
        output = json.loads(captured.out.strip())
        assert output == {"continue": True}

    def test_agent_spawn_tracing_off(self, monkeypatch, tmp_path, capsys):
        self._run_handler(agent_spawn_hook_handler, monkeypatch, tmp_path, capsys)

    def test_user_prompt_submit_tracing_off(self, monkeypatch, tmp_path, capsys):
        self._run_handler(user_prompt_submit_hook_handler, monkeypatch, tmp_path, capsys)

    def test_pre_tool_use_tracing_off(self, monkeypatch, tmp_path, capsys):
        self._run_handler(pre_tool_use_hook_handler, monkeypatch, tmp_path, capsys)

    def test_post_tool_use_tracing_off(self, monkeypatch, tmp_path, capsys):
        self._run_handler(post_tool_use_hook_handler, monkeypatch, tmp_path, capsys)

    def test_stop_tracing_off(self, monkeypatch, tmp_path, capsys):
        self._run_handler(stop_hook_handler, monkeypatch, tmp_path, capsys)


class TestTransientHandlerHappyPath:
    """Tracing-on happy path with fixture payload: emits {"continue": true}."""

    def _run_handler(self, handler_fn, payload_fixture, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv(MLFLOW_TRACING_ENABLED, "true")
        # Create settings.json so is_tracing_enabled reads from it
        settings_dir = tmp_path / ".kiro"
        settings_dir.mkdir(parents=True, exist_ok=True)
        (settings_dir / "settings.json").write_text(
            json.dumps({"env": {MLFLOW_TRACING_ENABLED: "true"}})
        )
        monkeypatch.chdir(tmp_path)

        payload = _load_fixture(payload_fixture)
        with mock.patch("sys.stdin") as mock_stdin:
            mock_stdin.read.return_value = json.dumps(payload)
            handler_fn()

        captured = capsys.readouterr()
        output = json.loads(captured.out.strip())
        assert output["continue"] is True

    def test_agent_spawn_happy_path(self, monkeypatch, tmp_path, capsys):
        self._run_handler(
            agent_spawn_hook_handler,
            "hook_payload_agent_spawn.json",
            monkeypatch,
            tmp_path,
            capsys,
        )

    def test_user_prompt_submit_happy_path(self, monkeypatch, tmp_path, capsys):
        self._run_handler(
            user_prompt_submit_hook_handler,
            "hook_payload_user_prompt_submit.json",
            monkeypatch,
            tmp_path,
            capsys,
        )

    def test_pre_tool_use_happy_path(self, monkeypatch, tmp_path, capsys):
        self._run_handler(
            pre_tool_use_hook_handler,
            "hook_payload_pre_tool_use.json",
            monkeypatch,
            tmp_path,
            capsys,
        )

    def test_post_tool_use_happy_path(self, monkeypatch, tmp_path, capsys):
        self._run_handler(
            post_tool_use_hook_handler,
            "hook_payload_post_tool_use.json",
            monkeypatch,
            tmp_path,
            capsys,
        )


class TestTransientHandlerMissingStdin:
    """Missing stdin / empty stdin: transient handlers emit {"continue": true} and exit 0."""

    def _run_handler(self, handler_fn, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv(MLFLOW_TRACING_ENABLED, "true")
        settings_dir = tmp_path / ".kiro"
        settings_dir.mkdir(parents=True, exist_ok=True)
        (settings_dir / "settings.json").write_text(
            json.dumps({"env": {MLFLOW_TRACING_ENABLED: "true"}})
        )
        monkeypatch.chdir(tmp_path)

        with mock.patch("sys.stdin") as mock_stdin:
            mock_stdin.read.return_value = ""
            handler_fn()

        captured = capsys.readouterr()
        output = json.loads(captured.out.strip())
        assert output["continue"] is True

    def test_agent_spawn_empty_stdin(self, monkeypatch, tmp_path, capsys):
        self._run_handler(agent_spawn_hook_handler, monkeypatch, tmp_path, capsys)

    def test_user_prompt_submit_empty_stdin(self, monkeypatch, tmp_path, capsys):
        self._run_handler(user_prompt_submit_hook_handler, monkeypatch, tmp_path, capsys)

    def test_pre_tool_use_empty_stdin(self, monkeypatch, tmp_path, capsys):
        self._run_handler(pre_tool_use_hook_handler, monkeypatch, tmp_path, capsys)

    def test_post_tool_use_empty_stdin(self, monkeypatch, tmp_path, capsys):
        self._run_handler(post_tool_use_hook_handler, monkeypatch, tmp_path, capsys)


class TestTransientHandlerExceptionHandling:
    """Exception inside handler body: transient handlers still emit {"continue": true}."""

    def _run_handler_with_exception(self, handler_fn, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv(MLFLOW_TRACING_ENABLED, "true")
        settings_dir = tmp_path / ".kiro"
        settings_dir.mkdir(parents=True, exist_ok=True)
        (settings_dir / "settings.json").write_text(
            json.dumps({"env": {MLFLOW_TRACING_ENABLED: "true"}})
        )
        monkeypatch.chdir(tmp_path)

        with mock.patch("sys.stdin") as mock_stdin:
            mock_stdin.read.side_effect = RuntimeError("stdin exploded")
            handler_fn()

        captured = capsys.readouterr()
        output = json.loads(captured.out.strip())
        assert output["continue"] is True

    def test_agent_spawn_exception(self, monkeypatch, tmp_path, capsys):
        self._run_handler_with_exception(agent_spawn_hook_handler, monkeypatch, tmp_path, capsys)

    def test_user_prompt_submit_exception(self, monkeypatch, tmp_path, capsys):
        self._run_handler_with_exception(
            user_prompt_submit_hook_handler, monkeypatch, tmp_path, capsys
        )

    def test_pre_tool_use_exception(self, monkeypatch, tmp_path, capsys):
        self._run_handler_with_exception(pre_tool_use_hook_handler, monkeypatch, tmp_path, capsys)

    def test_post_tool_use_exception(self, monkeypatch, tmp_path, capsys):
        self._run_handler_with_exception(post_tool_use_hook_handler, monkeypatch, tmp_path, capsys)


class TestStopHandlerFailure:
    """Stop handler with MLflow API raising: emits {"continue": false} and exit 1."""

    def test_stop_handler_trace_emission_failure(self, monkeypatch, tmp_path, capsys):
        """Stop handler exits 1 when trace emission fails."""
        monkeypatch.setenv(MLFLOW_TRACING_ENABLED, "true")
        settings_dir = tmp_path / ".kiro"
        settings_dir.mkdir(parents=True, exist_ok=True)
        (settings_dir / "settings.json").write_text(
            json.dumps({"env": {MLFLOW_TRACING_ENABLED: "true"}})
        )
        monkeypatch.chdir(tmp_path)

        # Create transcript files so we get past the short-circuit
        session_dir = tmp_path / ".kiro" / "sessions" / "cli"
        session_dir.mkdir(parents=True)
        (session_dir / "test-session-001.jsonl").write_text(
            '{"kind": "Prompt", "message_id": "msg-001", '
            '"content": [{"kind": "text", "data": "hi"}], "meta": {"timestamp": 1736946000}}\n'
        )
        (session_dir / "test-session-001.json").write_text("{}")

        payload = {
            "hook_event_name": "stop",
            "cwd": str(tmp_path),
            "session_id": "test-session-001",
        }

        with (
            mock.patch("sys.stdin") as mock_stdin,
            mock.patch(
                "mlflow.kiro_cli.tracing.process_turn",
                side_effect=RuntimeError("boom"),
            ),
            mock.patch.object(Path, "home", return_value=tmp_path),
        ):
            mock_stdin.read.return_value = json.dumps(payload)
            with pytest.raises(SystemExit, match="1"):
                stop_hook_handler()

        captured = capsys.readouterr()
        output = json.loads(captured.out.strip())
        assert output["continue"] is False
        assert "boom" in output.get("stopReason", "")


class TestStopHandlerMissingSessionId:
    """Stop handler with missing session_id: emits {"continue": true}, exit 0."""

    def test_stop_handler_missing_session_id(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv(MLFLOW_TRACING_ENABLED, "true")
        settings_dir = tmp_path / ".kiro"
        settings_dir.mkdir(parents=True, exist_ok=True)
        (settings_dir / "settings.json").write_text(
            json.dumps({"env": {MLFLOW_TRACING_ENABLED: "true"}})
        )
        monkeypatch.chdir(tmp_path)

        payload = {"hook_event_name": "stop", "cwd": str(tmp_path)}

        with mock.patch("sys.stdin") as mock_stdin:
            mock_stdin.read.return_value = json.dumps(payload)
            stop_hook_handler()

        captured = capsys.readouterr()
        output = json.loads(captured.out.strip())
        assert output["continue"] is True


# ---------------------------------------------------------------------------
# 11.6 UV env var behavior in hook command strings
# ---------------------------------------------------------------------------


def test_upsert_hook_with_uv_env_var(monkeypatch):
    """When UV is set, hook commands use 'uv run mlflow'."""
    monkeypatch.setenv("UV", "/path/to/uv")
    config = {HOOK_FIELD_HOOKS: {}}
    upsert_hook(config, "stop", "stop-hook")

    cmd = config[HOOK_FIELD_HOOKS]["stop"][0][HOOK_FIELD_COMMAND]
    assert cmd == "uv run mlflow autolog kiro-cli stop-hook"


def test_upsert_hook_without_uv_env_var(monkeypatch):
    """Without UV, hook commands use plain 'mlflow'."""
    monkeypatch.delenv("UV", raising=False)
    config = {HOOK_FIELD_HOOKS: {}}
    upsert_hook(config, "stop", "stop-hook")

    cmd = config[HOOK_FIELD_HOOKS]["stop"][0][HOOK_FIELD_COMMAND]
    assert cmd == "mlflow autolog kiro-cli stop-hook"


def test_setup_hooks_config_with_uv(agent_config_path, monkeypatch):
    """setup_hooks_config uses 'uv run mlflow' when UV is set."""
    monkeypatch.setenv("UV", "/path/to/uv")
    setup_hooks_config(agent_config_path)

    config = json.loads(agent_config_path.read_text())
    for event, entries in config[HOOK_FIELD_HOOKS].items():
        for entry in entries:
            cmd = entry.get(HOOK_FIELD_COMMAND, "")
            if MLFLOW_HOOK_IDENTIFIER in cmd:
                assert cmd.startswith("uv run mlflow"), f"Expected uv prefix for {event}"
