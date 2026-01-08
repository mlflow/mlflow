from unittest import mock

import pytest

from mlflow.entities.assessment import Feedback
from mlflow.genai.judges.utils import CategoricalRating


@pytest.fixture
def mock_trulens_dependencies():
    mock_provider = mock.MagicMock()
    mock_method = mock.MagicMock()
    mock_provider.mock_method = mock_method

    with (
        mock.patch(
            "mlflow.genai.scorers.trulens.models._check_trulens_installed",
        ),
        mock.patch(
            "mlflow.genai.scorers.trulens.create_trulens_provider",
            return_value=mock_provider,
        ),
        mock.patch(
            "mlflow.genai.scorers.trulens.get_feedback_method_name",
            return_value="mock_method",
        ),
    ):
        yield mock_method


def test_groundedness_scorer_pass(mock_trulens_dependencies):
    mock_method = mock_trulens_dependencies
    mock_method.return_value = (0.8, {"reason": "Test reason"})

    from mlflow.genai.scorers.trulens import Groundedness

    scorer = Groundedness(model="openai:/gpt-4")
    result = scorer(
        inputs="test input",
        outputs="test output",
        expectations={"context": "test context"},
    )

    assert isinstance(result, Feedback)
    assert result.name == "Groundedness"
    assert result.value == CategoricalRating.YES
    assert result.metadata["score"] == 0.8
    assert result.metadata["threshold"] == 0.5
    assert result.source.source_id == "openai:/gpt-4"


def test_groundedness_scorer_fail(mock_trulens_dependencies):
    mock_method = mock_trulens_dependencies
    mock_method.return_value = (0.3, {"reason": "Test reason"})

    from mlflow.genai.scorers.trulens import Groundedness

    scorer = Groundedness(model="openai:/gpt-4")
    result = scorer(
        outputs="test output",
        expectations={"context": "test context"},
    )

    assert result.value == CategoricalRating.NO
    assert result.metadata["score"] == 0.3


def test_context_relevance_scorer(mock_trulens_dependencies):
    mock_method = mock_trulens_dependencies
    mock_method.return_value = (0.7, {"reason": "Test reason"})

    from mlflow.genai.scorers.trulens import ContextRelevance

    scorer = ContextRelevance(model="openai:/gpt-4")
    result = scorer(
        inputs="test query",
        expectations={"context": "test context"},
    )

    assert isinstance(result, Feedback)
    assert result.name == "ContextRelevance"
    assert result.value == CategoricalRating.YES


def test_answer_relevance_scorer(mock_trulens_dependencies):
    mock_method = mock_trulens_dependencies
    mock_method.return_value = (0.9, {"reason": "Test reason"})

    from mlflow.genai.scorers.trulens import AnswerRelevance

    scorer = AnswerRelevance(model="openai:/gpt-4")
    result = scorer(
        inputs="test question",
        outputs="test answer",
    )

    assert isinstance(result, Feedback)
    assert result.name == "AnswerRelevance"
    assert result.value == CategoricalRating.YES


def test_coherence_scorer(mock_trulens_dependencies):
    mock_method = mock_trulens_dependencies
    mock_method.return_value = (0.85, {"reason": "Test reason"})

    from mlflow.genai.scorers.trulens import Coherence

    scorer = Coherence(model="openai:/gpt-4")
    result = scorer(outputs="test output")

    assert isinstance(result, Feedback)
    assert result.name == "Coherence"
    assert result.value == CategoricalRating.YES


def test_trulens_scorer_custom_threshold(mock_trulens_dependencies):
    mock_method = mock_trulens_dependencies
    mock_method.return_value = (0.6, {"reason": "Moderate score"})

    from mlflow.genai.scorers.trulens import Groundedness

    scorer = Groundedness(model="openai:/gpt-4", threshold=0.7)
    result = scorer(
        outputs="test output",
        expectations={"context": "test context"},
    )

    assert result.value == CategoricalRating.NO
    assert result.metadata["threshold"] == 0.7


def test_trulens_scorer_none_reasons(mock_trulens_dependencies):
    mock_method = mock_trulens_dependencies
    mock_method.return_value = (0.9, None)

    from mlflow.genai.scorers.trulens import Groundedness

    scorer = Groundedness(model="openai:/gpt-4")
    result = scorer(
        outputs="test output",
        expectations={"context": "test context"},
    )

    assert result.rationale is None


def test_trulens_get_scorer(mock_trulens_dependencies):
    mock_method = mock_trulens_dependencies
    mock_method.return_value = (0.9, {"reason": "Good"})

    from mlflow.genai.scorers.trulens import get_scorer

    scorer = get_scorer("Groundedness", model="openai:/gpt-4")
    result = scorer(
        outputs="test output",
        expectations={"context": "test context"},
    )

    assert isinstance(result, Feedback)
    assert result.name == "Groundedness"


def test_trulens_scorer_error_handling(mock_trulens_dependencies):
    mock_method = mock_trulens_dependencies
    mock_method.side_effect = RuntimeError("Evaluation failed")

    from mlflow.genai.scorers.trulens import Groundedness

    scorer = Groundedness(model="openai:/gpt-4")
    result = scorer(
        outputs="test output",
        expectations={"context": "test context"},
    )

    assert isinstance(result, Feedback)
    assert result.error is not None
    assert "Evaluation failed" in str(result.error)


def test_trulens_scorer_exports():
    from mlflow.genai.scorers.trulens import (
        AnswerRelevance,
        Coherence,
        ContextRelevance,
        Groundedness,
        TruLensScorer,
        get_scorer,
    )

    assert Groundedness is not None
    assert ContextRelevance is not None
    assert AnswerRelevance is not None
    assert Coherence is not None
    assert TruLensScorer is not None
    assert get_scorer is not None
