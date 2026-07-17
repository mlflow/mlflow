from unittest import mock

from mlflow.assistant.skill_installer import (
    install_skills,
    list_bundled_skills,
    list_installed_skills,
)


def test_install_skills_copies_to_destination(tmp_path):
    destination = tmp_path / "skills"
    installed = install_skills(destination)

    assert destination.exists()
    assert "agent-evaluation" in installed
    assert (destination / "agent-evaluation" / "SKILL.md").exists()


def test_install_skills_overwrites_existing(tmp_path):
    destination = tmp_path / "skills"
    destination.mkdir(parents=True)

    install_skills(destination)
    assert (destination / "agent-evaluation" / "SKILL.md").exists()


def test_list_installed_skills(tmp_path):
    # Create mock installed skills
    skill1 = tmp_path / "alpha-skill"
    skill1.mkdir()
    (skill1 / "SKILL.md").touch()

    skill2 = tmp_path / "beta-skill"
    skill2.mkdir()
    (skill2 / "SKILL.md").touch()

    skills = list_installed_skills(tmp_path)

    assert skills == ["alpha-skill", "beta-skill"]  # Sorted


def test_list_installed_skills_empty(tmp_path):
    skills = list_installed_skills(tmp_path)
    assert skills == []


def test_list_installed_skills_nonexistent_path(tmp_path):
    nonexistent = tmp_path / "does-not-exist"
    skills = list_installed_skills(nonexistent)
    assert skills == []


def test_list_bundled_skills_returns_empty_when_package_missing():
    with mock.patch(
        "mlflow.assistant.skill_installer.resources.files",
        side_effect=ModuleNotFoundError,
    ) as mock_files:
        assert list_bundled_skills() == []
    mock_files.assert_called_once()


def test_list_bundled_skills():
    skills = list_bundled_skills()

    by_name = {skill.name: skill for skill in skills}
    assert "agent-evaluation" in by_name
    assert by_name["agent-evaluation"].description
    assert (by_name["agent-evaluation"].path / "SKILL.md").is_file()
    assert [skill.name for skill in skills] == sorted(skill.name for skill in skills)
