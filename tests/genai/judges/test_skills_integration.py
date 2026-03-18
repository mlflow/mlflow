import textwrap

import pytest

from mlflow.genai.skills import SkillSet


@pytest.fixture
def skill_dir(tmp_path):
    skill_path = tmp_path / "test-skill"
    skill_path.mkdir()
    (skill_path / "SKILL.md").write_text(
        textwrap.dedent("""\
        ---
        name: test-skill
        description: A test skill for compliance evaluation.
        ---

        ## Compliance Rules

        Check for PII in outputs.
        """)
    )
    return skill_path


def test_make_judge_with_skills_paths(skill_dir):
    from mlflow.genai.judges import make_judge

    judge = make_judge(
        name="test",
        instructions="Evaluate {{ trace }} for compliance.",
        skills=[str(skill_dir)],
        model="openai:/gpt-4.1",
    )
    assert judge._skills is not None
    assert len(judge._skills.skills) == 1
    assert judge._skills.skills[0].name == "test-skill"


def test_make_judge_with_skillset_object(skill_dir):
    from mlflow.genai.judges import make_judge

    ss = SkillSet([skill_dir])
    judge = make_judge(
        name="test",
        instructions="Evaluate {{ trace }} for compliance.",
        skills=ss,
        model="openai:/gpt-4.1",
    )
    assert judge._skills is ss


def test_make_judge_with_skills_without_trace(skill_dir):
    from mlflow.genai.judges import make_judge

    judge = make_judge(
        name="test",
        instructions="Evaluate {{ inputs }} for compliance.",
        skills=[str(skill_dir)],
        model="openai:/gpt-4.1",
    )
    assert judge._skills is not None
    assert len(judge._skills.skills) == 1

    # Skills should appear in the system prompt even for non-trace-based judges
    system_msg = judge._build_system_message(is_trace_based=False)
    assert "test-skill" in system_msg


def test_make_judge_without_skills():
    from mlflow.genai.judges import make_judge

    judge = make_judge(
        name="test",
        instructions="Evaluate {{ trace }}.",
        model="openai:/gpt-4.1",
    )
    assert judge._skills is None


def test_system_prompt_includes_skills(skill_dir):
    from mlflow.genai.judges import make_judge

    judge = make_judge(
        name="test",
        instructions="Evaluate {{ trace }} for compliance.",
        skills=[str(skill_dir)],
        model="openai:/gpt-4.1",
    )
    system_msg = judge._build_system_message(is_trace_based=True)
    assert "test-skill" in system_msg
    assert "compliance evaluation" in system_msg.lower() or "test skill" in system_msg.lower()


def test_judge_with_skills_serialization_roundtrip(skill_dir):
    """Verify a judge with skills can be serialized and deserialized (register/reload)."""
    from mlflow.genai.judges import make_judge
    from mlflow.genai.scorers.base import Scorer

    judge = make_judge(
        name="compliance",
        instructions="Evaluate {{ trace }} for compliance.",
        skills=[str(skill_dir)],
        model="openai:/gpt-4.1",
    )

    # Serialize (what register() does)
    dumped = judge.model_dump()
    assert dumped["skill_contents"] is not None
    assert len(dumped["skill_contents"]) == 1
    assert dumped["skill_contents"][0]["name"] == "test-skill"
    assert "PII" in dumped["skill_contents"][0]["body"]

    # Deserialize (what loading a registered scorer does)
    reloaded = Scorer.model_validate(dumped)
    assert reloaded._skills is not None
    assert len(reloaded._skills.skills) == 1
    assert reloaded._skills.skills[0].name == "test-skill"
    assert "PII" in reloaded._skills.skills[0].body

    # Verify the reloaded judge produces the same system prompt
    original_prompt = judge._build_system_message(is_trace_based=True)
    reloaded_prompt = reloaded._build_system_message(is_trace_based=True)
    assert "test-skill" in reloaded_prompt
    assert original_prompt == reloaded_prompt
