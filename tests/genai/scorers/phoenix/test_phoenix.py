from unittest import mock

import pytest

from mlflow.entities.assessment import Feedback
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.utils import CategoricalRating


def test_phoenix_hallucination_scorer_requires_phoenix():
    with mock.patch.dict("sys.modules", {"phoenix": None, "phoenix.evals": None}):
        from mlflow.genai.scorers.phoenix import Hallucination

        with pytest.raises(MlflowException, match="arize-phoenix-evals"):
            Hallucination(model="openai:/gpt-4")


def test_phoenix_relevance_scorer_requires_phoenix():
    with mock.patch.dict("sys.modules", {"phoenix": None, "phoenix.evals": None}):
        from mlflow.genai.scorers.phoenix import Relevance

        with pytest.raises(MlflowException, match="arize-phoenix-evals"):
            Relevance(model="openai:/gpt-4")


def test_phoenix_toxicity_scorer_requires_phoenix():
    with mock.patch.dict("sys.modules", {"phoenix": None, "phoenix.evals": None}):
        from mlflow.genai.scorers.phoenix import Toxicity

        with pytest.raises(MlflowException, match="arize-phoenix-evals"):
            Toxicity(model="openai:/gpt-4")


def test_phoenix_qa_scorer_requires_phoenix():
    with mock.patch.dict("sys.modules", {"phoenix": None, "phoenix.evals": None}):
        from mlflow.genai.scorers.phoenix import QA

        with pytest.raises(MlflowException, match="arize-phoenix-evals"):
            QA(model="openai:/gpt-4")


def test_phoenix_summarization_scorer_requires_phoenix():
    with mock.patch.dict("sys.modules", {"phoenix": None, "phoenix.evals": None}):
        from mlflow.genai.scorers.phoenix import Summarization

        with pytest.raises(MlflowException, match="arize-phoenix-evals"):
            Summarization(model="openai:/gpt-4")


@pytest.fixture
def mock_phoenix_evals():
    mock_evaluator_instance = mock.MagicMock()
    mock_evaluator_instance.evaluate.return_value = ("factual", 0.9, "Well grounded.")

    mock_evaluator_class = mock.MagicMock(return_value=mock_evaluator_instance)
    mock_model_class = mock.MagicMock()

    mock_phoenix = mock.MagicMock()
    mock_phoenix.HallucinationEvaluator = mock_evaluator_class
    mock_phoenix.RelevanceEvaluator = mock_evaluator_class
    mock_phoenix.ToxicityEvaluator = mock_evaluator_class
    mock_phoenix.QAEvaluator = mock_evaluator_class
    mock_phoenix.SummarizationEvaluator = mock_evaluator_class
    mock_phoenix.OpenAIModel = mock_model_class
    mock_phoenix.LiteLLMModel = mock_model_class

    return mock_phoenix, mock_evaluator_instance


def test_phoenix_hallucination_scorer_with_mock(mock_phoenix_evals):
    mock_phoenix, mock_evaluator = mock_phoenix_evals
    mock_evaluator.evaluate.return_value = ("factual", 0.9, "Output is grounded.")

    mock_modules = {"phoenix": mock.MagicMock(), "phoenix.evals": mock_phoenix}
    with mock.patch.dict("sys.modules", mock_modules):
        from mlflow.genai.scorers.phoenix import Hallucination

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


def test_phoenix_relevance_scorer_with_mock(mock_phoenix_evals):
    mock_phoenix, mock_evaluator = mock_phoenix_evals
    mock_evaluator.evaluate.return_value = ("relevant", 0.85, "Context is relevant.")

    mock_modules = {"phoenix": mock.MagicMock(), "phoenix.evals": mock_phoenix}
    with mock.patch.dict("sys.modules", mock_modules):
        from mlflow.genai.scorers.phoenix import Relevance

        scorer = Relevance(model="openai:/gpt-4")
        result = scorer(
            inputs="What is machine learning?",
            expectations={"context": "Machine learning is a subset of AI."},
        )

        assert isinstance(result, Feedback)
        assert result.name == "Relevance"
        assert result.value == CategoricalRating.YES
        assert result.metadata["score"] == 0.85


def test_phoenix_toxicity_scorer_with_mock(mock_phoenix_evals):
    mock_phoenix, mock_evaluator = mock_phoenix_evals
    mock_evaluator.evaluate.return_value = ("non-toxic", 0.95, "Safe content.")

    mock_modules = {"phoenix": mock.MagicMock(), "phoenix.evals": mock_phoenix}
    with mock.patch.dict("sys.modules", mock_modules):
        from mlflow.genai.scorers.phoenix import Toxicity

        scorer = Toxicity(model="openai:/gpt-4")
        result = scorer(outputs="This is a friendly response.")

        assert isinstance(result, Feedback)
        assert result.name == "Toxicity"
        assert result.value == CategoricalRating.YES
        assert result.metadata["score"] == 0.95


def test_phoenix_qa_scorer_with_mock(mock_phoenix_evals):
    mock_phoenix, mock_evaluator = mock_phoenix_evals
    mock_evaluator.evaluate.return_value = ("correct", 1.0, "Answer is correct.")

    mock_modules = {"phoenix": mock.MagicMock(), "phoenix.evals": mock_phoenix}
    with mock.patch.dict("sys.modules", mock_modules):
        from mlflow.genai.scorers.phoenix import QA

        scorer = QA(model="openai:/gpt-4")
        result = scorer(
            inputs="What is 2+2?",
            outputs="4",
            expectations={"context": "2+2=4"},
        )

        assert isinstance(result, Feedback)
        assert result.name == "QA"
        assert result.value == CategoricalRating.YES


def test_phoenix_summarization_scorer_with_mock(mock_phoenix_evals):
    mock_phoenix, mock_evaluator = mock_phoenix_evals
    mock_evaluator.evaluate.return_value = ("good", 0.9, "Good summary.")

    mock_modules = {"phoenix": mock.MagicMock(), "phoenix.evals": mock_phoenix}
    with mock.patch.dict("sys.modules", mock_modules):
        from mlflow.genai.scorers.phoenix import Summarization

        scorer = Summarization(model="openai:/gpt-4")
        result = scorer(
            inputs="Long document text...",
            outputs="Brief summary.",
        )

        assert isinstance(result, Feedback)
        assert result.name == "Summarization"
        assert result.value == CategoricalRating.YES


def test_phoenix_scorer_negative_label(mock_phoenix_evals):
    mock_phoenix, mock_evaluator = mock_phoenix_evals
    mock_evaluator.evaluate.return_value = ("hallucinated", None, "Contains made-up info.")

    mock_modules = {"phoenix": mock.MagicMock(), "phoenix.evals": mock_phoenix}
    with mock.patch.dict("sys.modules", mock_modules):
        from mlflow.genai.scorers.phoenix import Hallucination

        scorer = Hallucination(model="openai:/gpt-4")
        result = scorer(
            inputs="test",
            outputs="test output",
            expectations={"context": "test context"},
        )

        assert isinstance(result, Feedback)
        assert result.value == CategoricalRating.NO
        assert result.metadata["score"] == 0.0


def test_phoenix_get_scorer(mock_phoenix_evals):
    mock_phoenix, mock_evaluator = mock_phoenix_evals
    mock_evaluator.evaluate.return_value = ("factual", 0.9, "Grounded.")

    mock_modules = {"phoenix": mock.MagicMock(), "phoenix.evals": mock_phoenix}
    with mock.patch.dict("sys.modules", mock_modules):
        from mlflow.genai.scorers.phoenix import get_scorer

        scorer = get_scorer("Hallucination", model="openai:/gpt-4")
        result = scorer(
            inputs="test",
            outputs="test output",
            expectations={"context": "test context"},
        )

        assert isinstance(result, Feedback)
        assert result.name == "Hallucination"


def test_phoenix_get_scorer_invalid_metric():
    mock_modules = {"phoenix": mock.MagicMock(), "phoenix.evals": mock.MagicMock()}
    with mock.patch.dict("sys.modules", mock_modules):
        from mlflow.genai.scorers.phoenix import get_scorer

        with pytest.raises(MlflowException, match="Unknown Phoenix metric"):
            get_scorer("InvalidMetric", model="openai:/gpt-4")


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

    assert Hallucination is not None
    assert Relevance is not None
    assert Toxicity is not None
    assert QA is not None
    assert Summarization is not None
    assert PhoenixScorer is not None
    assert get_scorer is not None


def test_phoenix_assessment_source(mock_phoenix_evals):
    mock_phoenix, mock_evaluator = mock_phoenix_evals
    mock_evaluator.evaluate.return_value = ("factual", 0.9, "Grounded.")

    mock_modules = {"phoenix": mock.MagicMock(), "phoenix.evals": mock_phoenix}
    with mock.patch.dict("sys.modules", mock_modules):
        from mlflow.genai.scorers.phoenix import Hallucination

        scorer = Hallucination(model="openai:/gpt-4")
        result = scorer(
            inputs="test",
            outputs="test output",
            expectations={"context": "test context"},
        )

        assert result.source is not None
        assert result.source.source_id == "openai:/gpt-4"
