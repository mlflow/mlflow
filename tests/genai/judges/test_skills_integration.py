import textwrap

import pytest

from mlflow.exceptions import MlflowException
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
    assert judge._skill_set is not None
    assert len(judge._skill_set.skills) == 1
    assert judge._skill_set.skills[0].name == "test-skill"


def test_make_judge_with_skillset_object(skill_dir):
    from mlflow.genai.judges import make_judge

    ss = SkillSet([skill_dir])
    judge = make_judge(
        name="test",
        instructions="Evaluate {{ trace }} for compliance.",
        skills=ss,
        model="openai:/gpt-4.1",
    )
    assert judge._skill_set is ss


def test_make_judge_skills_requires_trace(skill_dir):
    from mlflow.genai.judges import make_judge

    with pytest.raises(MlflowException, match="trace"):
        make_judge(
            name="test",
            instructions="Evaluate {{ inputs }} for compliance.",
            skills=[str(skill_dir)],
            model="openai:/gpt-4.1",
        )


def test_make_judge_without_skills():
    from mlflow.genai.judges import make_judge

    judge = make_judge(
        name="test",
        instructions="Evaluate {{ trace }}.",
        model="openai:/gpt-4.1",
    )
    assert judge._skill_set is None


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
