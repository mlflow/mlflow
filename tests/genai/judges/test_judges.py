from unittest.mock import patch

import pytest

from mlflow.entities.assessment import (
    AssessmentSource,
    AssessmentSourceType,
    Feedback,
    FeedbackValue,
)
from mlflow.genai import judges
from mlflow.genai.judges.databricks import _sanitize_feedback


def test_databricks_judges_are_importable():
    from mlflow.genai import judges
    from mlflow.genai.judges import (
        is_context_relevant,
        is_context_sufficient,
        is_correct,
        is_grounded,
        is_safe,
        meets_guidelines,
    )

    assert judges.is_context_relevant == is_context_relevant
    assert judges.is_context_sufficient == is_context_sufficient
    assert judges.is_correct == is_correct
    assert judges.is_grounded == is_grounded
    assert judges.is_safe == is_safe
    assert judges.meets_guidelines == meets_guidelines


def create_test_feedback(value: str) -> Feedback:
    return Feedback(
        name="test_feedback",
        source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="databricks"),
        rationale="Test rationale",
        metadata={},
        value=FeedbackValue(value=value, error=None),
        valid=True,
    )


def test_sanitize_feedback_happy_path():
    feedback = create_test_feedback("yes")
    result = _sanitize_feedback(feedback)
    assert isinstance(result.value, judges.CategoricalRating)
    assert result.value == judges.CategoricalRating.YES


def test_sanitize_feedback_no():
    feedback = create_test_feedback("no")
    result = _sanitize_feedback(feedback)
    assert isinstance(result.value, judges.CategoricalRating)
    assert result.value == judges.CategoricalRating.NO


def test_sanitize_feedback_unknown():
    feedback = create_test_feedback("unknown")
    result = _sanitize_feedback(feedback)
    assert isinstance(result.value, judges.CategoricalRating)
    assert result.value == judges.CategoricalRating.UNKNOWN


def test_meets_guidelines_happy_path():
    with patch("databricks.agents.evals.judges.guideline_adherence") as mock_judge:
        mock_judge.return_value = create_test_feedback("yes")
        result = judges.meets_guidelines(guidelines="test", context={"response": "test"})

        assert isinstance(result.value, judges.CategoricalRating)
        assert result.value == judges.CategoricalRating.YES
        mock_judge.assert_called_once()


@pytest.mark.parametrize(
    ("judge_func", "agents_judge_name", "args"),
    [
        (
            judges.is_context_relevant,
            "relevance_to_query",
            {"request": "test", "context": "test"},
        ),
        (
            judges.is_context_sufficient,
            "context_sufficiency",
            {"request": "test", "context": "test", "expected_facts": ["test"]},
        ),
        (
            judges.is_correct,
            "correctness",
            {"request": "test", "response": "test", "expected_facts": ["test"]},
        ),
        (
            judges.is_grounded,
            "groundedness",
            {"request": "test", "response": "test", "context": "test"},
        ),
        (
            judges.is_safe,
            "safety",
            {"content": "test"},
        ),
        (
            judges.meets_guidelines,
            "guideline_adherence",
            {"guidelines": "test", "context": {"response": "test"}},
        ),
    ],
)
def test_judge_functions_happy_path(judge_func, agents_judge_name, args):
    with patch(f"databricks.agents.evals.judges.{agents_judge_name}") as mock_judge:
        mock_judge.return_value = create_test_feedback("yes")
        result = judge_func(**args)
        assert isinstance(result.value, judges.CategoricalRating)
        assert result.value == judges.CategoricalRating.YES
        mock_judge.assert_called_once()
