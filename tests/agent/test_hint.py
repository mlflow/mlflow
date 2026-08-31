from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from mlflow.agent import hint
from mlflow.agent.agents import AGENTS

# Captured before the autouse fixture stubs it out.
_REAL_BUNDLED_LOOKUP = hint._bundled_skill_manifest


@pytest.fixture
def clean_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A directory with no agent markers and an isolated home."""
    for marker in (
        *hint._AGENT_ENV_MARKERS,
        *hint._AGENT_ENV_VALUES,
        *hint._AGENT_ENV_VALUES_WITHOUT_TTY,
    ):
        monkeypatch.delenv(marker, raising=False)
    monkeypatch.delenv("MLFLOW_DISABLE_AGENT_HINT", raising=False)
    hint._EMITTED_HINTS.clear()

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)

    cwd = tmp_path / "repo"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


@pytest.fixture(autouse=True)
def bundled_skill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Stand in for the skill copy that released MLflow packages ship."""
    manifest = tmp_path / "bundled" / hint.TRACING_SKILL / "SKILL.md"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("---\nname: tracing\n---\n")
    monkeypatch.setattr(hint, "_bundled_skill_manifest", lambda: manifest)
    return manifest


def hint_message() -> str | None:
    """Run the hint, returning the logged message or None when it stayed silent."""
    with mock.patch.object(hint._logger, "info") as info:
        hint.maybe_hint_tracing_skill()
    return info.call_args[0][0] if info.call_args else None


def test_silent_without_a_coding_agent(clean_env: Path):
    assert hint_message() is None


@pytest.mark.parametrize("marker", hint._AGENT_ENV_MARKERS)
def test_hints_under_each_supported_agent(
    clean_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker: str,
    bundled_skill: Path,
):
    monkeypatch.setenv(marker, "1")
    message = hint_message()
    assert message is not None
    # The hint points at the skill rather than restating its contents.
    assert hint.TRACING_SKILL in message
    assert "before writing any tracing" in message
    # One line, so any `| tail -N` or `| head -N` an agent appends keeps all of it.
    assert len(message.splitlines()) == 1
    assert str(bundled_skill) in message


def test_empty_marker_is_not_a_detection(clean_env: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLAUDECODE", "")
    assert hint_message() is None


def test_disable_env_var_silences_the_hint(clean_env: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    assert hint_message() is None


def test_hints_even_when_the_skill_is_installed(clean_env: Path, monkeypatch: pytest.MonkeyPatch):
    # Installation is deliberately not probed: skills end up in too many places
    # for the check to be accurate, and the pointer stays useful either way.
    monkeypatch.setenv("CLAUDECODE", "1")
    (clean_env / ".claude" / "skills" / hint.TRACING_SKILL).mkdir(parents=True)
    assert hint_message() is not None


def test_points_at_the_bundled_skill_rather_than_the_network(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch, bundled_skill: Path
):
    monkeypatch.setenv("CLAUDECODE", "1")
    message = hint_message()
    assert str(bundled_skill) in message
    assert "github.com" not in message
    assert "http" not in message


def test_silent_when_the_install_ships_no_skill(clean_env: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setattr(hint, "_bundled_skill_manifest", lambda: None)
    # Nothing local to point at, so say nothing rather than send the agent elsewhere.
    assert hint_message() is None


def test_bundled_lookup_survives_a_missing_skills_package(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        hint.resources, "files", mock.Mock(side_effect=ModuleNotFoundError("no skills"))
    )
    assert _REAL_BUNDLED_LOOKUP() is None


@pytest.mark.parametrize(("name", "value"), sorted(hint._AGENT_ENV_VALUES.items()))
def test_hints_for_value_specific_markers(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch, name: str, value: str, bundled_skill: Path
):
    monkeypatch.setenv(name, value)
    assert hint_message() is not None
    # A different value for the same variable is an ordinary human environment.
    monkeypatch.setenv(name, "something-else")
    assert hint_message() is None


@pytest.mark.parametrize(("name", "value"), sorted(hint._AGENT_ENV_VALUES_WITHOUT_TTY.items()))
def test_tty_gated_markers_only_count_off_a_terminal(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch, name: str, value: str, bundled_skill: Path
):
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(hint.sys.stdout, "isatty", lambda: False, raising=False)
    assert hint_message() is not None
    # The same variable in a human's terminal is not a detection.
    monkeypatch.setattr(hint.sys.stdout, "isatty", lambda: True, raising=False)
    assert hint_message() is None


def test_detects_agents_absent_from_the_setup_registry(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch, bundled_skill: Path
):
    # Cursor has no `mlflow agent setup` entry but still drives MLflow.
    assert "cursor" not in AGENTS
    monkeypatch.setenv("CURSOR_AGENT", "1")
    assert hint_message() is not None


def test_antipattern_warning_names_issue_and_is_emitted_once(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("CLAUDECODE", "1")

    with mock.patch.object(hint._logger, "warning") as warning:
        hint.maybe_warn_agent("missing-inputs", "A tool span has no inputs.")
        hint.maybe_warn_agent("missing-inputs", "A different instance has no inputs.")

    warning.assert_called_once()
    assert warning.call_args.args[1] == "A tool span has no inputs."
    assert "different instance" not in str(warning.call_args)


def test_antipattern_warning_is_silent_for_humans(clean_env: Path, caplog):
    hint.maybe_warn_agent("missing-inputs", "A tool span has no inputs.")
    assert not caplog.records


@pytest.mark.parametrize("intent_var", ["DATABRICKS_HOST", "DATABRICKS_CONFIG_PROFILE"])
def test_local_tracking_warns_when_databricks_is_configured(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch, intent_var: str
):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv(intent_var, "configured")

    with (
        mock.patch("mlflow.get_tracking_uri", return_value="file:///tmp/mlruns"),
        mock.patch("mlflow.agent.hint.maybe_warn_agent") as warn,
    ):
        hint.maybe_warn_local_tracking_for_databricks()

    warn.assert_called_once_with(
        "databricks-intent-with-local-tracking",
        "A GenAI trace is being written to the local tracking URI 'file:///tmp/mlruns' even "
        "though a Databricks environment or profile is configured.",
    )


def test_local_tracking_check_ignores_human_and_remote_tracking(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("DATABRICKS_HOST", "https://example.databricks.com")
    with mock.patch("mlflow.agent.hint.maybe_warn_agent") as warn:
        hint.maybe_warn_local_tracking_for_databricks()
        monkeypatch.setenv("CLAUDECODE", "1")
        with mock.patch("mlflow.get_tracking_uri", return_value="databricks"):
            hint.maybe_warn_local_tracking_for_databricks()
    warn.assert_not_called()
