from pathlib import Path

import pytest
from click.testing import CliRunner

from mlflow.assistant.skill_installer import BundledSkill
from mlflow.cli.skills import commands


@pytest.fixture
def runner():
    return CliRunner()


def test_list_command_renders_skills(runner, monkeypatch):
    skills = [
        BundledSkill(name="alpha-skill", description="Does alpha things.", path=Path("/s/alpha")),
        BundledSkill(name="beta-skill", description="", path=Path("/s/beta")),
    ]
    monkeypatch.setattr("mlflow.cli.skills.list_bundled_skills", lambda: skills)

    result = runner.invoke(commands, ["list"])

    assert result.exit_code == 0
    assert "alpha-skill" in result.output
    assert "Does alpha things." in result.output
    assert "beta-skill" in result.output
    assert str(Path("/s/alpha")) in result.output
    assert str(Path("/s/beta")) in result.output


def test_list_command_handles_no_skills(runner, monkeypatch):
    monkeypatch.setattr("mlflow.cli.skills.list_bundled_skills", lambda: [])

    result = runner.invoke(commands, ["list"])

    assert result.exit_code == 0
    assert "No MLflow skills found" in result.output
    assert "git submodule update --init" in result.output


def test_view_command_renders_skill(runner, monkeypatch):
    skills = [
        BundledSkill(name="alpha-skill", description="Does alpha things.", path=Path("/s/alpha")),
        BundledSkill(name="beta-skill", description="Does beta things.", path=Path("/s/beta")),
    ]
    monkeypatch.setattr("mlflow.cli.skills.list_bundled_skills", lambda: skills)

    result = runner.invoke(commands, ["view", "alpha-skill"])

    assert result.exit_code == 0
    assert "alpha-skill" in result.output
    assert "Does alpha things." in result.output
    assert str(Path("/s/alpha")) in result.output
    assert "beta-skill" not in result.output


def test_view_command_skill_not_found(runner, monkeypatch):
    monkeypatch.setattr("mlflow.cli.skills.list_bundled_skills", lambda: [])

    result = runner.invoke(commands, ["view", "missing-skill"])

    assert result.exit_code == 0
    assert "Skill missing-skill not found." in result.output


def test_view_command_skill_without_description(runner, monkeypatch):
    skills = [BundledSkill(name="alpha-skill", description="", path=Path("/s/alpha"))]
    monkeypatch.setattr("mlflow.cli.skills.list_bundled_skills", lambda: skills)

    result = runner.invoke(commands, ["view", "alpha-skill"])

    assert result.exit_code == 0
    assert "alpha-skill" in result.output
    assert str(Path("/s/alpha")) in result.output
    assert "\n  \n" not in result.output
