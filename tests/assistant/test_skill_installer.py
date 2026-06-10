import pytest

from mlflow.assistant.skill_installer import (
    _parse_skill_manifest,
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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("---\nname: foo\ndescription: bar\n---\nbody", {"name": "foo", "description": "bar"}),
        ("# No frontmatter\nbody", {}),
        ("---\n---\nbody", {}),
        ("---\nnot a mapping\n---\nbody", {}),
    ],
)
def test_parse_skill_manifest(text, expected):
    assert _parse_skill_manifest(text) == expected


def test_list_bundled_skills():
    skills = list_bundled_skills()

    by_name = {skill.name: skill for skill in skills}
    assert "agent-evaluation" in by_name
    assert by_name["agent-evaluation"].description
    assert (by_name["agent-evaluation"].path / "SKILL.md").is_file()
    assert [skill.name for skill in skills] == sorted(skill.name for skill in skills)
