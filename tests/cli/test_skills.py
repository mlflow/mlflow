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
    assert "Skills directory: " in result.output


def test_list_command_handles_no_skills(runner, monkeypatch):
    monkeypatch.setattr("mlflow.cli.skills.list_bundled_skills", lambda: [])

    result = runner.invoke(commands, ["list"])

    assert result.exit_code == 0
    assert "No MLflow skills found" in result.output
    assert "git submodule update --init" in result.output
