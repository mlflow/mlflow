import textwrap

import pytest

from mlflow.exceptions import MlflowException


@pytest.fixture
def skill_dir(tmp_path):
    skill_path = tmp_path / "test-skill"
    skill_path.mkdir()
    (skill_path / "SKILL.md").write_text(
        textwrap.dedent("""\
        ---
        name: test-skill
        description: A test skill for evaluation.
        ---

        ## Instructions

        This is the skill body with {{ trace }} reference.
        """)
    )
    refs = skill_path / "references"
    refs.mkdir()
    (refs / "RUBRIC.md").write_text("# Rubric\n\nDetailed scoring guide.")
    return skill_path


@pytest.fixture
def skill_with_metadata(tmp_path):
    skill_path = tmp_path / "meta-skill"
    skill_path.mkdir()
    (skill_path / "SKILL.md").write_text(
        textwrap.dedent("""\
        ---
        name: meta-skill
        description: Skill with metadata.
        metadata:
          author: test-org
          version: "1.0"
          category: compliance
        ---

        Body content here.
        """)
    )
    return skill_path


def test_load_skill_from_directory(skill_dir):
    from mlflow.genai.skills import Skill
    from mlflow.genai.skills.parsing import load_skill

    skill = load_skill(skill_dir)
    assert isinstance(skill, Skill)
    assert skill.name == "test-skill"
    assert skill.description == "A test skill for evaluation."
    assert "skill body" in skill.body
    assert "references/RUBRIC.md" in skill.files
    assert "Detailed scoring guide" in skill.files["references/RUBRIC.md"]


def test_load_skill_from_file_path(skill_dir):
    from mlflow.genai.skills.parsing import load_skill

    skill = load_skill(skill_dir / "SKILL.md")
    assert skill.name == "test-skill"


def test_load_skill_with_metadata(skill_with_metadata):
    from mlflow.genai.skills.parsing import load_skill

    skill = load_skill(skill_with_metadata)
    assert skill.metadata == {"author": "test-org", "version": "1.0", "category": "compliance"}


def test_load_skill_no_extra_files(tmp_path):
    skill_path = tmp_path / "no-files"
    skill_path.mkdir()
    (skill_path / "SKILL.md").write_text(
        "---\nname: no-files\ndescription: No extra files.\n---\nBody."
    )
    from mlflow.genai.skills.parsing import load_skill

    skill = load_skill(skill_path)
    assert skill.files == {}


@pytest.mark.parametrize(
    "name",
    ["UPPERCASE", "-leading", "trailing-", "con--secutive", "a" * 65, ""],
)
def test_invalid_skill_name(tmp_path, name):
    skill_path = tmp_path / "bad"
    skill_path.mkdir()
    (skill_path / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Bad.\n---\nBody.")
    from mlflow.genai.skills.parsing import load_skill

    with pytest.raises(MlflowException, match="name"):
        load_skill(skill_path)


def test_missing_description(tmp_path):
    skill_path = tmp_path / "no-desc"
    skill_path.mkdir()
    (skill_path / "SKILL.md").write_text("---\nname: no-desc\n---\nBody.")
    from mlflow.genai.skills.parsing import load_skill

    with pytest.raises(MlflowException, match="description"):
        load_skill(skill_path)


def test_load_skill_ignores_non_spec_files(tmp_path):
    skill_path = tmp_path / "mixed"
    skill_path.mkdir()
    (skill_path / "SKILL.md").write_text("---\nname: mixed\ndescription: Mixed files.\n---\nBody.")
    # Unrelated file at root of skill directory
    (skill_path / "unrelated.txt").write_text("should be ignored")
    # Non-spec subdirectory
    other = skill_path / "other"
    other.mkdir()
    (other / "data.csv").write_text("a,b,c")
    # Spec directory
    refs = skill_path / "references"
    refs.mkdir()
    (refs / "GUIDE.md").write_text("guide content")

    from mlflow.genai.skills.parsing import load_skill

    skill = load_skill(skill_path)
    assert "references/GUIDE.md" in skill.files
    assert "unrelated.txt" not in skill.files
    assert "other/data.csv" not in skill.files


def test_missing_skill_md(tmp_path):
    skill_path = tmp_path / "empty"
    skill_path.mkdir()
    from mlflow.genai.skills.parsing import load_skill

    with pytest.raises(MlflowException, match="SKILL.md"):
        load_skill(skill_path)


def test_load_skill_invalid_path(tmp_path):
    from mlflow.genai.skills.parsing import load_skill

    with pytest.raises(MlflowException, match="Invalid skill path"):
        load_skill(tmp_path / "nonexistent_file.txt")


def test_load_skill_empty_body(tmp_path):
    skill_path = tmp_path / "empty-body"
    skill_path.mkdir()
    (skill_path / "SKILL.md").write_text("---\nname: empty-body\ndescription: Has no body.\n---\n")
    from mlflow.genai.skills.parsing import load_skill

    with pytest.raises(MlflowException, match="non-empty body"):
        load_skill(skill_path)


def test_load_skill_non_dict_metadata(tmp_path):
    skill_path = tmp_path / "bad-meta"
    skill_path.mkdir()
    (skill_path / "SKILL.md").write_text(
        "---\nname: bad-meta\ndescription: Bad metadata.\nmetadata: not-a-dict\n---\nBody."
    )
    from mlflow.genai.skills.parsing import load_skill

    skill = load_skill(skill_path)
    assert skill.metadata == {}


def test_parse_frontmatter_missing_opening():
    from mlflow.genai.skills.parsing import _parse_frontmatter

    with pytest.raises(MlflowException, match="must start with YAML frontmatter"):
        _parse_frontmatter("no frontmatter here")


def test_parse_frontmatter_missing_closing():
    from mlflow.genai.skills.parsing import _parse_frontmatter

    with pytest.raises(MlflowException, match="enclosed by --- delimiters"):
        _parse_frontmatter("---\nname: foo\n")


def test_skillset_from_contents_validates_types():
    from mlflow.genai.skills import SkillSet

    ss = SkillSet.from_contents([
        {
            "name": "test-skill",
            "description": "A skill.",
            "metadata": "not-a-dict",
            "body": "Body content.",
            "files": "not-a-dict",
        }
    ])
    assert len(ss.skills) == 1
    assert ss.skills[0].metadata == {}
    assert ss.skills[0].files == {}


def test_skillset_from_contents_coerces_values():
    from mlflow.genai.skills import SkillSet

    ss = SkillSet.from_contents([
        {
            "name": "coerce-skill",
            "description": "Coerce test.",
            "metadata": {"version": 1, "count": 42},
            "body": "Body.",
            "files": {"path/file.md": "content"},
        }
    ])
    skill = ss.skills[0]
    assert skill.metadata == {"version": "1", "count": "42"}
    assert skill.files == {"path/file.md": "content"}


def test_skillset_from_paths(skill_dir, skill_with_metadata):
    from mlflow.genai.skills import SkillSet

    ss = SkillSet([skill_dir, skill_with_metadata])
    assert len(ss.skills) == 2
    assert ss.get_skill("test-skill") is not None
    assert ss.get_skill("meta-skill") is not None
    assert ss.get_skill("nonexistent") is None
