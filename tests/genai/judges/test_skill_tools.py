import textwrap

import pytest

from mlflow.genai.skills import SkillSet


@pytest.fixture
def skills(tmp_path):
    skill_path = tmp_path / "test-skill"
    skill_path.mkdir()
    (skill_path / "SKILL.md").write_text(
        textwrap.dedent("""\
        ---
        name: test-skill
        description: A test skill.
        ---

        ## Body

        Skill body content here.
        """)
    )
    refs = skill_path / "references"
    refs.mkdir()
    (refs / "RUBRIC.md").write_text("# Rubric\n\nDetailed guide.")
    return SkillSet([skill_path])


def test_read_skill_returns_body(skills):
    from mlflow.genai.judges.tools.read_skill import ReadSkillTool

    tool = ReadSkillTool()
    result = tool.invoke(skills=skills, skill_name="test-skill")
    assert "Skill body content here" in result


def test_read_skill_unknown_name(skills):
    from mlflow.genai.judges.tools.read_skill import ReadSkillTool

    tool = ReadSkillTool()
    result = tool.invoke(skills=skills, skill_name="nonexistent")
    assert "Error" in result
    assert "test-skill" in result


def test_read_skill_has_valid_definition():
    from mlflow.genai.judges.tools.read_skill import ReadSkillTool

    tool = ReadSkillTool()
    defn = tool.get_definition()
    assert defn.function.name == "read_skill_markdown_content"
    assert "skill_name" in defn.function.parameters.properties


def test_read_skill_file_unknown_skill(skills):
    from mlflow.genai.judges.tools.read_skill_file import ReadSkillFileTool

    tool = ReadSkillFileTool()
    result = tool.invoke(skills=skills, skill_name="nonexistent", file_path="any.md")
    assert "Error" in result
    assert "nonexistent" in result
    assert "test-skill" in result


def test_read_skill_file_returns_content(skills):
    from mlflow.genai.judges.tools.read_skill_file import ReadSkillFileTool

    tool = ReadSkillFileTool()
    result = tool.invoke(skills=skills, skill_name="test-skill", file_path="references/RUBRIC.md")
    assert "Detailed guide" in result


def test_read_skill_file_unknown_file(skills):
    from mlflow.genai.judges.tools.read_skill_file import ReadSkillFileTool

    tool = ReadSkillFileTool()
    result = tool.invoke(skills=skills, skill_name="test-skill", file_path="references/MISSING.md")
    assert "Error" in result
    assert "RUBRIC.md" in result


@pytest.mark.parametrize("bad_path", ["../../../etc/passwd", "/etc/passwd", "../../secret.txt"])
def test_read_skill_file_path_traversal(skills, bad_path):
    from mlflow.genai.judges.tools.read_skill_file import ReadSkillFileTool

    tool = ReadSkillFileTool()
    result = tool.invoke(skills=skills, skill_name="test-skill", file_path=bad_path)
    assert "Error" in result
    assert "Invalid" in result or "relative" in result.lower()


def test_read_skill_file_has_valid_definition():
    from mlflow.genai.judges.tools.read_skill_file import ReadSkillFileTool

    tool = ReadSkillFileTool()
    defn = tool.get_definition()
    assert defn.function.name == "read_skill_companion_file"
    assert "skill_name" in defn.function.parameters.properties
    assert "file_path" in defn.function.parameters.properties
