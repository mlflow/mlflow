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
    """A directory with no agent markers, no skills, and an isolated home."""
    for marker in (
        *hint._AGENT_ENV_MARKERS,
        *hint._AGENT_ENV_VALUES,
        *hint._AGENT_ENV_VALUES_WITHOUT_TTY,
    ):
        monkeypatch.delenv(marker, raising=False)
    monkeypatch.delenv("MLFLOW_DISABLE_AGENT_HINT", raising=False)

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


@pytest.mark.parametrize("scope", ["project", "global"])
def test_silent_once_the_skill_is_installed(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch, scope: str
):
    monkeypatch.setenv("CLAUDECODE", "1")
    root = clean_env if scope == "project" else Path.home()
    (root / ".claude" / "skills" / hint.TRACING_SKILL).mkdir(parents=True)
    assert hint_message() is None


def test_silent_from_a_subdirectory_of_the_project(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("CLAUDECODE", "1")
    (clean_env / ".claude" / "skills" / hint.TRACING_SKILL).mkdir(parents=True)
    # Agents often run from a subdirectory while the skill sits at the repo root.
    nested = clean_env / "services" / "api"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert hint_message() is None


def test_silent_for_codex_project_skills_from_the_assistant_flow(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("CODEX_SANDBOX", "seatbelt")
    # `mlflow agent setup`'s assistant flow writes Codex project skills under
    # `.codex/skills`, not the `.agents/skills` Codex reads natively.
    (clean_env / ".codex" / "skills" / hint.TRACING_SKILL).mkdir(parents=True)
    assert hint_message() is None


def test_silent_for_a_custom_skills_location(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("CLAUDECODE", "1")
    # `mlflow agent setup` can install to an arbitrary path recorded in the config.
    custom = tmp_path / "somewhere" / "else"
    (custom / hint.TRACING_SKILL).mkdir(parents=True)
    monkeypatch.setattr(hint, "_custom_skill_dirs", lambda: (custom,))
    assert hint_message() is None


def test_custom_skill_dirs_survives_an_unreadable_config(monkeypatch: pytest.MonkeyPatch):
    from mlflow.assistant.config import AssistantConfig

    monkeypatch.setattr(AssistantConfig, "load", mock.Mock(side_effect=OSError("boom")))
    assert hint._custom_skill_dirs() == ()


def test_a_different_skill_does_not_suppress_the_hint(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("CLAUDECODE", "1")
    (clean_env / ".claude" / "skills" / "some-other-skill").mkdir(parents=True)
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
