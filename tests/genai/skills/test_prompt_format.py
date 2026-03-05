import pytest

from mlflow.genai.skills import SkillSet


@pytest.fixture
def two_skills(tmp_path):
    for name, desc in [
        ("compliance", "Compliance rules for financial services."),
        ("api-schema", "Internal API contracts."),
    ]:
        d = tmp_path / name
        d.mkdir()
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {desc}\n---\nBody of {name}."
        )
    return SkillSet([tmp_path / "compliance", tmp_path / "api-schema"])


def test_to_prompt_xml_for_claude(two_skills):
    prompt = two_skills.to_prompt(model="anthropic:/claude-sonnet-4-6")
    assert "<available_skills>" in prompt
    assert "<name>compliance</name>" in prompt
    assert "<name>api-schema</name>" in prompt
    assert "read_skill" in prompt


def test_to_prompt_markdown_for_openai(two_skills):
    prompt = two_skills.to_prompt(model="openai:/gpt-4.1")
    assert "## Available Skills" in prompt
    assert "**compliance**" in prompt
    assert "**api-schema**" in prompt
    assert "<available_skills>" not in prompt


def test_to_prompt_default_is_markdown(two_skills):
    prompt = two_skills.to_prompt(model=None)
    assert "## Available Skills" in prompt


def test_to_prompt_empty_skillset():
    ss = SkillSet([])
    assert ss.to_prompt() == ""
