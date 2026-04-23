import textwrap

import pytest


@pytest.fixture
def skill_dir(tmp_path):
    skill_path = tmp_path / "ser-skill"
    skill_path.mkdir()
    (skill_path / "SKILL.md").write_text(
        textwrap.dedent("""\
        ---
        name: ser-skill
        description: Skill for serialization test.
        metadata:
          version: "1.0"
        ---

        Skill body for serialization.
        """)
    )
    refs = skill_path / "references"
    refs.mkdir()
    (refs / "GUIDE.md").write_text("Reference guide content.")
    return skill_path


def test_model_dump_includes_skills(skill_dir):
    from mlflow.genai.judges import make_judge

    judge = make_judge(
        name="test",
        instructions="Evaluate {{ trace }}.",
        skills=[str(skill_dir)],
        model="openai:/gpt-4.1",
    )
    dumped = judge.model_dump()
    assert dumped["skill_contents"] is not None
    assert len(dumped["skill_contents"]) == 1
    skill = dumped["skill_contents"][0]
    assert skill["name"] == "ser-skill"
    assert skill["description"] == "Skill for serialization test."
    assert "Skill body for serialization" in skill["body"]
    assert "references/GUIDE.md" in skill["files"]


def test_model_dump_without_skills():
    from mlflow.genai.judges import make_judge

    judge = make_judge(
        name="test",
        instructions="Evaluate {{ trace }}.",
        model="openai:/gpt-4.1",
    )
    dumped = judge.model_dump()
    assert dumped.get("skill_contents") is None


def test_serialization_roundtrip(skill_dir):
    from mlflow.genai.judges import make_judge
    from mlflow.genai.scorers.base import SerializedScorer

    judge = make_judge(
        name="test",
        instructions="Evaluate {{ trace }}.",
        skills=[str(skill_dir)],
        model="openai:/gpt-4.1",
    )
    dumped = judge.model_dump()

    serialized = SerializedScorer(**dumped)
    assert serialized.skill_contents is not None
    assert len(serialized.skill_contents) == 1
    assert serialized.skill_contents[0]["name"] == "ser-skill"
