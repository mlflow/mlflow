from unittest.mock import Mock, patch

import phoenix.evals as phoenix_evals
import pytest

from mlflow.entities.assessment import Feedback


@pytest.fixture
def mock_model():
    mock = Mock()
    mock._verbose = False
    mock._rate_limiter = Mock()
    mock._rate_limiter._verbose = False
    return mock


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
def test_phoenix_scorer(mock_model, scorer_class, metric_name, label, score, explanation):
    with patch("mlflow.genai.scorers.phoenix.create_phoenix_model", return_value=mock_model):
        from mlflow.genai.scorers import phoenix

        scorer_cls = getattr(phoenix, scorer_class)
        scorer = scorer_cls(model="openai:/gpt-4")

    with patch.object(scorer._evaluator, "evaluate", return_value=(label, score, explanation)):
        result = scorer(
            inputs="test input",
            outputs="test output",
            expectations={"expected_response": "test reference"},
        )

    assert isinstance(result, Feedback)
    assert result.name == metric_name
    assert result.value == label
    assert result.metadata["score"] == score
    assert result.source.source_id == "openai:/gpt-4"


def test_phoenix_scorer_negative_label(mock_model):
    with patch("mlflow.genai.scorers.phoenix.create_phoenix_model", return_value=mock_model):
        from mlflow.genai.scorers.phoenix import Hallucination

        scorer = Hallucination(model="openai:/gpt-4")

    with patch.object(
        scorer._evaluator, "evaluate", return_value=("hallucinated", None, "Made-up info.")
    ):
        result = scorer(
            inputs="test",
            outputs="test output",
            expectations={"expected_response": "test context"},
        )

    assert isinstance(result, Feedback)
    assert result.value == "hallucinated"
    assert result.rationale == "Made-up info."


def test_phoenix_scorer_none_explanation(mock_model):
    with patch("mlflow.genai.scorers.phoenix.create_phoenix_model", return_value=mock_model):
        from mlflow.genai.scorers.phoenix import Hallucination

        scorer = Hallucination(model="openai:/gpt-4")

    with patch.object(scorer._evaluator, "evaluate", return_value=("factual", 0.9, None)):
        result = scorer(
            inputs="test",
            outputs="test output",
            expectations={"expected_response": "test context"},
        )

    assert result.rationale is None


def test_phoenix_get_scorer(mock_model):
    with patch("mlflow.genai.scorers.phoenix.create_phoenix_model", return_value=mock_model):
        from mlflow.genai.scorers.phoenix import get_scorer

        scorer = get_scorer("Hallucination", model="openai:/gpt-4")

    with patch.object(scorer._evaluator, "evaluate", return_value=("factual", 0.9, "Grounded.")):
        result = scorer(
            inputs="test",
            outputs="test output",
            expectations={"expected_response": "test context"},
        )

    assert isinstance(result, Feedback)
    assert result.name == "Hallucination"


def test_phoenix_scorer_evaluator_is_real_instance(mock_model):
    with patch("mlflow.genai.scorers.phoenix.create_phoenix_model", return_value=mock_model):
        from mlflow.genai.scorers.phoenix import Hallucination

        scorer = Hallucination(model="openai:/gpt-4")

    assert isinstance(scorer._evaluator, phoenix_evals.HallucinationEvaluator)


def test_phoenix_scorer_error_handling(mock_model):
    with patch("mlflow.genai.scorers.phoenix.create_phoenix_model", return_value=mock_model):
        from mlflow.genai.scorers.phoenix import Hallucination

        scorer = Hallucination(model="openai:/gpt-4")

    with patch.object(scorer._evaluator, "evaluate", side_effect=RuntimeError("Evaluation failed")):
        result = scorer(
            inputs="test",
            outputs="test output",
            expectations={"expected_response": "test context"},
        )

    assert isinstance(result, Feedback)
    assert result.error is not None
    assert "Evaluation failed" in str(result.error)
