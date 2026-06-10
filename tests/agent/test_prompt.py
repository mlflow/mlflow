from __future__ import annotations

from pathlib import Path

import pytest

from mlflow.agent.agents import AGENTS
from mlflow.agent.setup.prompt import _render, build_prompt


def test_render_substitutes_placeholder():
    assert _render("hello {{ name }}", name="world") == "hello world"


def test_render_accepts_no_whitespace():
    assert _render("{{name}}", name="x") == "x"


def test_render_supports_multiple_placeholders():
    assert _render("{{ a }}/{{ b }}", a="1", b="2") == "1/2"


def test_render_leaves_single_braces_untouched():
    assert _render("mlflow.log_dict({'k': 'v'})") == "mlflow.log_dict({'k': 'v'})"


def test_render_raises_on_missing_key():
    with pytest.raises(KeyError, match="missing"):
        _render("{{ missing }}", other="x")


def test_build_prompt_with_user_uri_omits_local_server_section(tmp_path: Path):
    out = build_prompt(tmp_path, AGENTS["claude"], "http://remote:5000")
    assert "Start a local MLflow tracking server" not in out
    assert "MLFLOW_TRACKING_URI=http://remote:5000" in out
    assert 'mlflow.set_tracking_uri("http://remote:5000")' in out


def test_build_prompt_with_local_server_port_bakes_url(tmp_path: Path):
    out = build_prompt(
        tmp_path,
        AGENTS["claude"],
        "http://127.0.0.1:5050",
        local_server_port=5050,
    )
    assert "Start a local MLflow tracking server" in out
    assert "mlflow server --host 127.0.0.1 --port 5050" in out
    assert "http://127.0.0.1:5050" in out


def test_build_prompt_skills_installed_uses_agent_skills_dir(tmp_path: Path):
    out = build_prompt(tmp_path, AGENTS["claude"], "http://remote:5000", skills_installed=True)
    assert "has been installed at `.claude/skills/`" in out
    assert "are already installed; do not\n  overwrite them." in out


def test_build_prompt_skills_declined_uses_bundled_path(tmp_path: Path):
    out = build_prompt(tmp_path, AGENTS["claude"], "http://remote:5000", skills_installed=False)
    assert "bundled at" in out
    assert "/mlflow/assistant/skills" in out
    assert "are already installed; do not\n  overwrite them." not in out


@pytest.mark.parametrize("skills_installed", [True, False])
@pytest.mark.parametrize(
    ("tracking_uri", "local_server_port", "experiment_path"),
    [
        ("databricks", None, "/Users/me@example.com/my-app"),
        ("http://127.0.0.1:5050", 5050, None),
        ("http://some-remote:5000", None, None),
    ],
)
def test_build_prompt_renders_all_placeholders(
    tmp_path: Path,
    tracking_uri: str,
    local_server_port: int | None,
    experiment_path: str | None,
    skills_installed: bool,
):
    out = build_prompt(
        tmp_path,
        AGENTS["claude"],
        tracking_uri,
        local_server_port=local_server_port,
        experiment_path=experiment_path,
        skills_installed=skills_installed,
    )
    assert "{{" not in out
    assert "}}" not in out


def test_build_prompt_databricks_injects_workspace_section(tmp_path: Path):
    out = build_prompt(
        tmp_path,
        AGENTS["claude"],
        "databricks",
        experiment_path="/Users/me@example.com/my-app",
    )
    assert "Configure the Databricks workspace" in out
    assert "WorkspaceClient().current_user.me()" in out
    assert 'mlflow.set_experiment("/Users/me@example.com/my-app")' in out
    assert "Start a local MLflow tracking server" not in out


def test_build_prompt_databricks_without_experiment_path_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="experiment_path is required"):
        build_prompt(tmp_path, AGENTS["claude"], "databricks")


def test_build_prompt_with_user_uri_omits_databricks_section(tmp_path: Path):
    out = build_prompt(tmp_path, AGENTS["claude"], "http://127.0.0.1:5001")
    assert "Configure the Databricks workspace" not in out
    assert "WorkspaceClient" not in out
