import sys
from unittest import mock

import pytest

from mlflow.entities.assessment import Feedback
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.utils import CategoricalRating


def test_phoenix_check_installed_raises_without_phoenix():
    """Test that _check_phoenix_installed raises when phoenix is not installed."""
    with mock.patch.dict("sys.modules", {"phoenix": None, "phoenix.evals": None}):
        # Need to reload the module to pick up the mocked sys.modules

        # Clear cache
        for mod in list(sys.modules.keys()):
            if "mlflow.genai.scorers.phoenix" in mod:
                del sys.modules[mod]

        from mlflow.genai.scorers.phoenix.models import _check_phoenix_installed

        with pytest.raises(MlflowException, match="arize-phoenix-evals"):
            _check_phoenix_installed()


def test_phoenix_hallucination_scorer_with_mock():
    """Test Hallucination scorer with mocked Phoenix evaluator."""
    # Import first to ensure module is loaded
    from mlflow.genai.scorers.phoenix import Hallucination

    mock_evaluator = mock.MagicMock()
    mock_evaluator.evaluate.return_value = ("factual", 0.9, "Output is grounded.")

    mock_model = mock.MagicMock()

    with (
        mock.patch(
            "mlflow.genai.scorers.phoenix.phoenix.create_phoenix_model",
            return_value=mock_model,
        ),
        mock.patch(
            "mlflow.genai.scorers.phoenix.phoenix.get_evaluator_class",
            return_value=mock.MagicMock(return_value=mock_evaluator),
        ),
        mock.patch(
            "mlflow.genai.scorers.phoenix.phoenix.get_metric_config",
            return_value={"positive_label": "factual"},
        ),
    ):
        scorer = Hallucination(model="openai:/gpt-4")
        result = scorer(
            inputs="What is the capital of France?",
            outputs="Paris is the capital of France.",
            expectations={"context": "France is in Europe. Its capital is Paris."},
        )

        assert isinstance(result, Feedback)
        assert result.name == "Hallucination"
        assert result.value == CategoricalRating.YES
        assert result.metadata["score"] == 0.9
        assert result.metadata["label"] == "factual"


def test_phoenix_relevance_scorer_with_mock():
    """Test Relevance scorer with mocked Phoenix evaluator."""
    from mlflow.genai.scorers.phoenix import Relevance

    mock_evaluator = mock.MagicMock()
    mock_evaluator.evaluate.return_value = ("relevant", 0.85, "Context is relevant.")

    mock_model = mock.MagicMock()

    with (
        mock.patch(
            "mlflow.genai.scorers.phoenix.phoenix.create_phoenix_model",
            return_value=mock_model,
        ),
        mock.patch(
            "mlflow.genai.scorers.phoenix.phoenix.get_evaluator_class",
            return_value=mock.MagicMock(return_value=mock_evaluator),
        ),
        mock.patch(
            "mlflow.genai.scorers.phoenix.phoenix.get_metric_config",
            return_value={"positive_label": "relevant"},
        ),
    ):
        scorer = Relevance(model="openai:/gpt-4")
        result = scorer(
            inputs="What is machine learning?",
            expectations={"context": "Machine learning is a subset of AI."},
        )

        assert isinstance(result, Feedback)
        assert result.name == "Relevance"
        assert result.value == CategoricalRating.YES
        assert result.metadata["score"] == 0.85


def test_phoenix_toxicity_scorer_with_mock():
    """Test Toxicity scorer with mocked Phoenix evaluator."""
    from mlflow.genai.scorers.phoenix import Toxicity

    mock_evaluator = mock.MagicMock()
    mock_evaluator.evaluate.return_value = ("non-toxic", 0.95, "Safe content.")

    mock_model = mock.MagicMock()

    with (
        mock.patch(
            "mlflow.genai.scorers.phoenix.phoenix.create_phoenix_model",
            return_value=mock_model,
        ),
        mock.patch(
            "mlflow.genai.scorers.phoenix.phoenix.get_evaluator_class",
            return_value=mock.MagicMock(return_value=mock_evaluator),
        ),
        mock.patch(
            "mlflow.genai.scorers.phoenix.phoenix.get_metric_config",
            return_value={"positive_label": "non-toxic"},
        ),
    ):
        scorer = Toxicity(model="openai:/gpt-4")
        result = scorer(outputs="This is a friendly response.")

        assert isinstance(result, Feedback)
        assert result.name == "Toxicity"
        assert result.value == CategoricalRating.YES
        assert result.metadata["score"] == 0.95


def test_phoenix_qa_scorer_with_mock():
    """Test QA scorer with mocked Phoenix evaluator."""
    from mlflow.genai.scorers.phoenix import QA

    mock_evaluator = mock.MagicMock()
    mock_evaluator.evaluate.return_value = ("correct", 1.0, "Answer is correct.")

    mock_model = mock.MagicMock()

    with (
        mock.patch(
            "mlflow.genai.scorers.phoenix.phoenix.create_phoenix_model",
            return_value=mock_model,
        ),
        mock.patch(
            "mlflow.genai.scorers.phoenix.phoenix.get_evaluator_class",
            return_value=mock.MagicMock(return_value=mock_evaluator),
        ),
        mock.patch(
            "mlflow.genai.scorers.phoenix.phoenix.get_metric_config",
            return_value={"positive_label": "correct"},
        ),
    ):
        scorer = QA(model="openai:/gpt-4")
        result = scorer(
            inputs="What is 2+2?",
            outputs="4",
            expectations={"context": "2+2=4"},
        )

        assert isinstance(result, Feedback)
        assert result.name == "QA"
        assert result.value == CategoricalRating.YES


def test_phoenix_summarization_scorer_with_mock():
    """Test Summarization scorer with mocked Phoenix evaluator."""
    from mlflow.genai.scorers.phoenix import Summarization

    mock_evaluator = mock.MagicMock()
    mock_evaluator.evaluate.return_value = ("good", 0.9, "Good summary.")

    mock_model = mock.MagicMock()

    with (
        mock.patch(
            "mlflow.genai.scorers.phoenix.phoenix.create_phoenix_model",
            return_value=mock_model,
        ),
        mock.patch(
            "mlflow.genai.scorers.phoenix.phoenix.get_evaluator_class",
            return_value=mock.MagicMock(return_value=mock_evaluator),
        ),
        mock.patch(
            "mlflow.genai.scorers.phoenix.phoenix.get_metric_config",
            return_value={"positive_label": "good"},
        ),
    ):
        scorer = Summarization(model="openai:/gpt-4")
        result = scorer(
            inputs="Long document text...",
            outputs="Brief summary.",
        )

        assert isinstance(result, Feedback)
        assert result.name == "Summarization"
        assert result.value == CategoricalRating.YES


def test_phoenix_scorer_negative_label():
    """Test scorer with negative label returns NO."""
    from mlflow.genai.scorers.phoenix import Hallucination

    mock_evaluator = mock.MagicMock()
    mock_evaluator.evaluate.return_value = ("hallucinated", None, "Contains made-up info.")

    mock_model = mock.MagicMock()

    with (
        mock.patch(
            "mlflow.genai.scorers.phoenix.phoenix.create_phoenix_model",
            return_value=mock_model,
        ),
        mock.patch(
            "mlflow.genai.scorers.phoenix.phoenix.get_evaluator_class",
            return_value=mock.MagicMock(return_value=mock_evaluator),
        ),
        mock.patch(
            "mlflow.genai.scorers.phoenix.phoenix.get_metric_config",
            return_value={"positive_label": "factual"},
        ),
    ):
        scorer = Hallucination(model="openai:/gpt-4")
        result = scorer(
            inputs="test",
            outputs="test output",
            expectations={"context": "test context"},
        )

        assert isinstance(result, Feedback)
        assert result.value == CategoricalRating.NO
        assert result.metadata["score"] == 0.0


def test_phoenix_get_scorer():
    """Test get_scorer factory function."""
    from mlflow.genai.scorers.phoenix import get_scorer

    mock_evaluator = mock.MagicMock()
    mock_evaluator.evaluate.return_value = ("factual", 0.9, "Grounded.")

    mock_model = mock.MagicMock()

    with (
        mock.patch(
            "mlflow.genai.scorers.phoenix.phoenix.create_phoenix_model",
            return_value=mock_model,
        ),
        mock.patch(
            "mlflow.genai.scorers.phoenix.phoenix.get_evaluator_class",
            return_value=mock.MagicMock(return_value=mock_evaluator),
        ),
        mock.patch(
            "mlflow.genai.scorers.phoenix.phoenix.get_metric_config",
            return_value={"positive_label": "factual"},
        ),
    ):
        scorer = get_scorer("Hallucination", model="openai:/gpt-4")
        result = scorer(
            inputs="test",
            outputs="test output",
            expectations={"context": "test context"},
        )

        assert isinstance(result, Feedback)
        assert result.name == "Hallucination"


def test_phoenix_get_scorer_invalid_metric():
    """Test get_scorer raises for invalid metric name."""
    from mlflow.genai.scorers.phoenix.registry import get_evaluator_class

    mock_modules = {"phoenix": mock.MagicMock(), "phoenix.evals": mock.MagicMock()}
    with pytest.raises(MlflowException, match="Unknown Phoenix metric"):
        # This should fail in registry before reaching phoenix.evals
        with mock.patch.dict("sys.modules", mock_modules):
            get_evaluator_class("InvalidMetric")


def test_phoenix_scorer_exports():
    """Test that all expected classes are exported."""
    from mlflow.genai.scorers.phoenix import (
        QA,
        Hallucination,
        PhoenixScorer,
        Relevance,
        Summarization,
        Toxicity,
        get_scorer,
    )

    assert Hallucination is not None
    assert Relevance is not None
    assert Toxicity is not None
    assert QA is not None
    assert Summarization is not None
    assert PhoenixScorer is not None
    assert get_scorer is not None


def test_phoenix_assessment_source():
    """Test that assessment source is set correctly."""
    from mlflow.genai.scorers.phoenix import Hallucination

    mock_evaluator = mock.MagicMock()
    mock_evaluator.evaluate.return_value = ("factual", 0.9, "Grounded.")

    mock_model = mock.MagicMock()

    with (
        mock.patch(
            "mlflow.genai.scorers.phoenix.phoenix.create_phoenix_model",
            return_value=mock_model,
        ),
        mock.patch(
            "mlflow.genai.scorers.phoenix.phoenix.get_evaluator_class",
            return_value=mock.MagicMock(return_value=mock_evaluator),
        ),
        mock.patch(
            "mlflow.genai.scorers.phoenix.phoenix.get_metric_config",
            return_value={"positive_label": "factual"},
        ),
    ):
        scorer = Hallucination(model="openai:/gpt-4")
        result = scorer(
            inputs="test",
            outputs="test output",
            expectations={"context": "test context"},
        )

        assert result.source is not None
        assert result.source.source_id == "openai:/gpt-4"
