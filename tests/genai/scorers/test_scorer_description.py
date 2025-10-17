import pytest

from mlflow.genai import scorer
from mlflow.genai.judges import make_judge
from mlflow.genai.judges.instructions_judge import InstructionsJudge
from mlflow.genai.scorers import RelevanceToQuery


def test_decorator_scorer_with_description():
    description = "Checks if output length exceeds 100 characters"

    @scorer(description=description)
    def length_check(outputs) -> bool:
        return len(outputs) > 100

    assert length_check.description == description
    assert length_check.name == "length_check"


def test_decorator_scorer_without_description():
    @scorer
    def simple_scorer(outputs) -> bool:
        return True

    assert simple_scorer.description is None
    assert simple_scorer.name == "simple_scorer"


def test_decorator_scorer_with_name_and_description():
    description = "Custom description for scorer"

    @scorer(name="custom_name", description=description)
    def my_scorer(outputs) -> bool:
        return True

    assert my_scorer.name == "custom_name"
    assert my_scorer.description == description


def test_builtin_scorer_with_description():
    description = "Custom description for relevance scorer"
    scorer_instance = RelevanceToQuery(description=description)

    assert scorer_instance.description == description
    assert scorer_instance.name == "relevance_to_query"


def test_builtin_scorer_without_description():
    scorer_instance = RelevanceToQuery()

    assert scorer_instance.description is None


@pytest.mark.parametrize(
    ("name", "description"),
    [
        ("test_judge", "Evaluates response quality"),
        ("another_judge", None),
        ("judge_with_desc", "This is a test description"),
    ],
)
def test_make_judge_with_description(name: str, description: str | None):
    judge = make_judge(
        name=name,
        instructions="Evaluate if {{ outputs }} is good quality",
        model="openai:/gpt-4",
        description=description,
    )

    assert judge.name == name
    assert judge.description == description


def test_instructions_judge_with_description():
    description = "Direct InstructionsJudge with description"
    judge = InstructionsJudge(
        name="test_judge",
        instructions="Evaluate {{ outputs }}",
        model="openai:/gpt-4",
        description=description,
    )

    assert judge.description == description
    assert judge.name == "test_judge"


def test_instructions_judge_without_description():
    judge = InstructionsJudge(
        name="test_judge",
        instructions="Evaluate {{ outputs }}",
        model="openai:/gpt-4",
    )

    assert judge.description is None


def test_decorator_scorer_serialization_with_description():
    description = "Test description for serialization"

    @scorer(description=description)
    def test_scorer(outputs) -> bool:
        return True

    serialized = test_scorer.model_dump()

    assert "description" in serialized
    assert serialized["description"] == description
    assert serialized["name"] == "test_scorer"


def test_decorator_scorer_serialization_without_description():
    @scorer
    def test_scorer(outputs) -> bool:
        return True

    serialized = test_scorer.model_dump()

    assert "description" in serialized
    assert serialized["description"] is None


def test_decorator_scorer_deserialization_with_description():
    from mlflow.genai.scorers.base import Scorer

    description = "Test description for deserialization"

    @scorer(description=description)
    def test_scorer(outputs) -> bool:
        return True

    # Serialize and deserialize
    serialized = test_scorer.model_dump()
    deserialized = Scorer.model_validate(serialized)

    assert deserialized.description == description
    assert deserialized.name == "test_scorer"


def test_builtin_scorer_serialization_with_description():
    description = "Builtin scorer description"
    scorer_instance = RelevanceToQuery(description=description)

    serialized = scorer_instance.model_dump()

    assert "description" in serialized
    assert serialized["description"] == description


def test_builtin_scorer_deserialization_with_description():
    from mlflow.genai.scorers.base import Scorer

    description = "Builtin scorer description"
    scorer_instance = RelevanceToQuery(description=description)

    # Serialize and deserialize
    serialized = scorer_instance.model_dump()
    deserialized = Scorer.model_validate(serialized)

    assert isinstance(deserialized, RelevanceToQuery)
    assert deserialized.description == description
    assert deserialized.name == "relevance_to_query"


def test_instructions_judge_serialization_with_description():
    description = "InstructionsJudge description"
    judge = InstructionsJudge(
        name="test_judge",
        instructions="Evaluate {{ outputs }}",
        model="openai:/gpt-4",
        description=description,
    )

    serialized = judge.model_dump()

    assert "description" in serialized
    assert serialized["description"] == description


def test_instructions_judge_deserialization_with_description():
    from mlflow.genai.scorers.base import Scorer

    description = "InstructionsJudge description"
    judge = InstructionsJudge(
        name="test_judge",
        instructions="Evaluate {{ outputs }}",
        model="openai:/gpt-4",
        description=description,
    )

    # Serialize and deserialize
    serialized = judge.model_dump()
    deserialized = Scorer.model_validate(serialized)

    assert isinstance(deserialized, InstructionsJudge)
    assert deserialized.description == description
    assert deserialized.name == "test_judge"


def test_backward_compatibility_scorer_without_description():
    # Test decorator scorer
    @scorer
    def old_scorer(outputs) -> bool:
        return True

    assert old_scorer.description is None

    # Test builtin scorer
    builtin = RelevanceToQuery()
    assert builtin.description is None

    # Test InstructionsJudge
    judge = InstructionsJudge(
        name="old_judge",
        instructions="Evaluate {{ outputs }}",
        model="openai:/gpt-4",
    )
    assert judge.description is None
