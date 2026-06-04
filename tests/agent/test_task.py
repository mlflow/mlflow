from __future__ import annotations

from pathlib import Path

import pytest

from mlflow.agent.agents import AGENTS
from mlflow.agent.setup.task import _render, build_task


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


def test_build_task_with_user_uri_omits_local_server_section(tmp_path: Path):
    out = build_task(tmp_path, AGENTS["claude"], "databricks")
    assert "Start a local MLflow tracking server" not in out
    assert "MLFLOW_TRACKING_URI=databricks" in out
    assert 'mlflow.set_tracking_uri("databricks")' in out


def test_build_task_with_local_server_port_bakes_url(tmp_path: Path):
    out = build_task(
        tmp_path,
        AGENTS["claude"],
        "http://127.0.0.1:5050",
        local_server_port=5050,
    )
    assert "Start a local MLflow tracking server" in out
    assert "mlflow server --host 127.0.0.1 --port 5050" in out
    assert "http://127.0.0.1:5050" in out


def test_build_task_skills_installed_uses_agent_skills_dir(tmp_path: Path):
    out = build_task(tmp_path, AGENTS["claude"], "databricks", skills_installed=True)
    assert "has been installed at `.claude/skills/`" in out
    assert "are already installed; do not\n  overwrite them." in out


def test_build_task_skills_declined_uses_bundled_path(tmp_path: Path):
    out = build_task(tmp_path, AGENTS["claude"], "databricks", skills_installed=False)
    assert "bundled at" in out
    assert "/mlflow/assistant/skills" in out
    assert "are already installed; do not\n  overwrite them." not in out


@pytest.mark.parametrize("skills_installed", [True, False])
@pytest.mark.parametrize(
    ("tracking_uri", "local_server_port"),
    [("databricks", None), ("http://127.0.0.1:5050", 5050)],
)
def test_build_task_renders_all_placeholders(
    tmp_path: Path,
    tracking_uri: str,
    local_server_port: int | None,
    skills_installed: bool,
):
    out = build_task(
        tmp_path,
        AGENTS["claude"],
        tracking_uri,
        local_server_port=local_server_port,
        skills_installed=skills_installed,
    )
    assert "{{" not in out
    assert "}}" not in out
