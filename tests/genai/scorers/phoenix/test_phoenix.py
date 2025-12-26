from unittest import mock

import pytest

from mlflow.entities.assessment import Feedback


@pytest.fixture
def mock_phoenix_dependencies():
    """Fixture that mocks Phoenix dependencies for scorer tests."""
    mock_evaluator = mock.MagicMock()
    mock_model = mock.MagicMock()

    with (
        mock.patch(
            "mlflow.genai.scorers.phoenix.create_phoenix_model",
            return_value=mock_model,
        ),
        mock.patch(
            "mlflow.genai.scorers.phoenix.get_evaluator_class",
            return_value=mock.MagicMock(return_value=mock_evaluator),
        ),
    ):
        yield mock_evaluator


@pytest.mark.parametrize(
    ("scorer_class", "metric_name", "label", "score", "explanation"),
    [
        ("Hallucination", "Hallucination", "factual", 0.9, "Output is grounded."),
        ("Relevance", "Relevance", "relevant", 0.85, "Context is relevant."),
        ("Toxicity", "Toxicity", "non-toxic", 0.95, "Safe content."),
        ("QA", "QA", "correct", 1.0, "Answer is correct."),
        ("Summarization", "Summarization", "good", 0.9, "Good summary."),
    ],
)
def test_phoenix_scorer(
    mock_phoenix_dependencies,
    scorer_class,
    metric_name,
    label,
    score,
    explanation,
):
    mock_evaluator = mock_phoenix_dependencies
    mock_evaluator.evaluate.return_value = (label, score, explanation)

    # Import dynamically to work with the mock
    from mlflow.genai.scorers import phoenix

    scorer_cls = getattr(phoenix, scorer_class)
    scorer = scorer_cls(model="openai:/gpt-4")

    result = scorer(
        inputs="test input",
        outputs="test output",
        expectations={"expected_response": "test reference"},
    )

    assert isinstance(result, Feedback)
    assert result.name == metric_name
    assert result.value == label
    assert result.metadata["score"] == score
    assert result.metadata["label"] == label
    assert result.source.source_id == "openai:/gpt-4"


def test_phoenix_scorer_negative_label(mock_phoenix_dependencies):
    mock_evaluator = mock_phoenix_dependencies
    mock_evaluator.evaluate.return_value = ("hallucinated", None, "Contains made-up info.")

    from mlflow.genai.scorers.phoenix import Hallucination

    scorer = Hallucination(model="openai:/gpt-4")
    result = scorer(
        inputs="test",
        outputs="test output",
        expectations={"expected_response": "test context"},
    )

    assert isinstance(result, Feedback)
    assert result.value == "hallucinated"
    assert result.rationale == "Contains made-up info."


def test_phoenix_scorer_none_explanation(mock_phoenix_dependencies):
    mock_evaluator = mock_phoenix_dependencies
    mock_evaluator.evaluate.return_value = ("factual", 0.9, None)

    from mlflow.genai.scorers.phoenix import Hallucination

    scorer = Hallucination(model="openai:/gpt-4")
    result = scorer(
        inputs="test",
        outputs="test output",
        expectations={"expected_response": "test context"},
    )

    assert result.rationale is None


def test_phoenix_get_scorer(mock_phoenix_dependencies):
    mock_evaluator = mock_phoenix_dependencies
    mock_evaluator.evaluate.return_value = ("factual", 0.9, "Grounded.")

    from mlflow.genai.scorers.phoenix import get_scorer

    scorer = get_scorer("Hallucination", model="openai:/gpt-4")
    result = scorer(
        inputs="test",
        outputs="test output",
        expectations={"expected_response": "test context"},
    )

    assert isinstance(result, Feedback)
    assert result.name == "Hallucination"


def test_phoenix_scorer_with_evaluator_kwargs(mock_phoenix_dependencies):
    mock_evaluator = mock_phoenix_dependencies
    mock_evaluator.evaluate.return_value = ("factual", 0.9, "Grounded.")

    from mlflow.genai.scorers.phoenix import Hallucination

    # Should not raise - kwargs passed to evaluator
    scorer = Hallucination(model="openai:/gpt-4", custom_param="value")
    assert scorer is not None


def test_phoenix_scorer_error_handling(mock_phoenix_dependencies):
    mock_evaluator = mock_phoenix_dependencies
    mock_evaluator.evaluate.side_effect = RuntimeError("Evaluation failed")

    from mlflow.genai.scorers.phoenix import Hallucination

    scorer = Hallucination(model="openai:/gpt-4")
    result = scorer(
        inputs="test",
        outputs="test output",
        expectations={"expected_response": "test context"},
    )

    assert isinstance(result, Feedback)
    assert result.error is not None
    assert "Evaluation failed" in str(result.error)


def test_phoenix_scorer_exports():
    from mlflow.genai.scorers.phoenix import (
        QA,
        Hallucination,
        PhoenixScorer,
        Relevance,
        Summarization,
        Toxicity,
        get_scorer,
    )

    # Verify all expected classes are exported
    assert Hallucination is not None
    assert Relevance is not None
    assert Toxicity is not None
    assert QA is not None
    assert Summarization is not None
    assert PhoenixScorer is not None
    assert get_scorer is not None
