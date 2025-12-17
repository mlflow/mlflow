from unittest import mock

import pytest

from mlflow.entities.assessment import Feedback
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.utils import CategoricalRating


def test_trulens_groundedness_scorer_requires_trulens():
    with mock.patch.dict(
        "sys.modules",
        {"trulens": None, "trulens.providers": None, "trulens.providers.openai": None},
    ):
        from mlflow.genai.scorers.trulens import Groundedness

        with pytest.raises(MlflowException, match="trulens"):
            Groundedness(model="openai:/gpt-4")


def test_trulens_context_relevance_scorer_requires_trulens():
    with mock.patch.dict(
        "sys.modules",
        {"trulens": None, "trulens.providers": None, "trulens.providers.openai": None},
    ):
        from mlflow.genai.scorers.trulens import ContextRelevance

        with pytest.raises(MlflowException, match="trulens"):
            ContextRelevance(model="openai:/gpt-4")


def test_trulens_answer_relevance_scorer_requires_trulens():
    with mock.patch.dict(
        "sys.modules",
        {"trulens": None, "trulens.providers": None, "trulens.providers.openai": None},
    ):
        from mlflow.genai.scorers.trulens import AnswerRelevance

        with pytest.raises(MlflowException, match="trulens"):
            AnswerRelevance(model="openai:/gpt-4")


def test_trulens_coherence_scorer_requires_trulens():
    with mock.patch.dict(
        "sys.modules",
        {"trulens": None, "trulens.providers": None, "trulens.providers.openai": None},
    ):
        from mlflow.genai.scorers.trulens import Coherence

        with pytest.raises(MlflowException, match="trulens"):
            Coherence(model="openai:/gpt-4")


@pytest.fixture
def mock_trulens_provider():
    mock_provider = mock.MagicMock()
    mock_provider.groundedness_measure_with_cot_reasons.return_value = (
        0.9,
        {"reason": "Statement is well supported by the context."},
    )
    mock_provider.context_relevance_with_cot_reasons.return_value = (
        0.85,
        {"reason": "The context is highly relevant to the query."},
    )
    mock_provider.relevance_with_cot_reasons.return_value = (
        0.95,
        {"reason": "The answer directly addresses the question."},
    )
    mock_provider.coherence_with_cot_reasons.return_value = (
        0.8,
        {"reason": "The text flows logically."},
    )
    return mock_provider


@pytest.fixture
def mock_trulens_modules(mock_trulens_provider):
    mock_openai_class = mock.MagicMock(return_value=mock_trulens_provider)

    mock_providers_openai = mock.MagicMock()
    mock_providers_openai.OpenAI = mock_openai_class

    return {
        "trulens": mock.MagicMock(),
        "trulens.providers": mock.MagicMock(),
        "trulens.providers.openai": mock_providers_openai,
    }


def test_trulens_groundedness_scorer_with_mock(mock_trulens_modules, mock_trulens_provider):
    with mock.patch.dict("sys.modules", mock_trulens_modules):
        from mlflow.genai.scorers.trulens import Groundedness

        scorer = Groundedness(model="openai:/gpt-4")
        result = scorer(
            outputs="The Eiffel Tower is 330 meters tall.",
            expectations={"context": "The Eiffel Tower stands at 330 meters."},
        )

        assert isinstance(result, Feedback)
        assert result.name == "Groundedness"
        assert result.value == CategoricalRating.YES
        assert result.metadata["score"] == 0.9
        assert "supported" in result.rationale.lower()


def test_trulens_context_relevance_scorer_with_mock(mock_trulens_modules, mock_trulens_provider):
    with mock.patch.dict("sys.modules", mock_trulens_modules):
        from mlflow.genai.scorers.trulens import ContextRelevance

        scorer = ContextRelevance(model="openai:/gpt-4")
        result = scorer(
            inputs="What is the capital of France?",
            expectations={"context": "Paris is the capital of France."},
        )

        assert isinstance(result, Feedback)
        assert result.name == "ContextRelevance"
        assert result.value == CategoricalRating.YES
        assert result.metadata["score"] == 0.85


def test_trulens_answer_relevance_scorer_with_mock(mock_trulens_modules, mock_trulens_provider):
    with mock.patch.dict("sys.modules", mock_trulens_modules):
        from mlflow.genai.scorers.trulens import AnswerRelevance

        scorer = AnswerRelevance(model="openai:/gpt-4")
        result = scorer(
            inputs="What is machine learning?",
            outputs="Machine learning is a branch of AI.",
        )

        assert isinstance(result, Feedback)
        assert result.name == "AnswerRelevance"
        assert result.value == CategoricalRating.YES
        assert result.metadata["score"] == 0.95


def test_trulens_coherence_scorer_with_mock(mock_trulens_modules, mock_trulens_provider):
    with mock.patch.dict("sys.modules", mock_trulens_modules):
        from mlflow.genai.scorers.trulens import Coherence

        scorer = Coherence(model="openai:/gpt-4")
        result = scorer(
            outputs="Machine learning is a branch of AI. It enables systems to learn.",
        )

        assert isinstance(result, Feedback)
        assert result.name == "Coherence"
        assert result.value == CategoricalRating.YES
        assert result.metadata["score"] == 0.8


def test_trulens_scorer_below_threshold(mock_trulens_modules, mock_trulens_provider):
    mock_trulens_provider.groundedness_measure_with_cot_reasons.return_value = (
        0.3,
        {"reason": "Not well supported."},
    )

    with mock.patch.dict("sys.modules", mock_trulens_modules):
        from mlflow.genai.scorers.trulens import Groundedness

        scorer = Groundedness(model="openai:/gpt-4", threshold=0.5)
        result = scorer(
            outputs="Made up information.",
            expectations={"context": "Different context."},
        )

        assert isinstance(result, Feedback)
        assert result.value == CategoricalRating.NO
        assert result.metadata["score"] == 0.3
        assert result.metadata["threshold"] == 0.5


def test_trulens_scorer_custom_threshold(mock_trulens_modules, mock_trulens_provider):
    mock_trulens_provider.groundedness_measure_with_cot_reasons.return_value = (
        0.6,
        {"reason": "Moderately supported."},
    )

    with mock.patch.dict("sys.modules", mock_trulens_modules):
        from mlflow.genai.scorers.trulens import Groundedness

        # With threshold 0.7, score 0.6 should fail
        scorer = Groundedness(model="openai:/gpt-4", threshold=0.7)
        result = scorer(
            outputs="Test output.",
            expectations={"context": "Test context."},
        )

        assert result.value == CategoricalRating.NO

        # With threshold 0.5, score 0.6 should pass
        scorer2 = Groundedness(model="openai:/gpt-4", threshold=0.5)
        result2 = scorer2(
            outputs="Test output.",
            expectations={"context": "Test context."},
        )

        assert result2.value == CategoricalRating.YES


def test_trulens_get_scorer(mock_trulens_modules, mock_trulens_provider):
    with mock.patch.dict("sys.modules", mock_trulens_modules):
        from mlflow.genai.scorers.trulens import get_scorer

        scorer = get_scorer("Groundedness", model="openai:/gpt-4")
        result = scorer(
            outputs="Test output.",
            expectations={"context": "Test context."},
        )

        assert isinstance(result, Feedback)
        assert result.name == "Groundedness"


def test_trulens_get_scorer_invalid_metric():
    with mock.patch.dict(
        "sys.modules",
        {
            "trulens": mock.MagicMock(),
            "trulens.providers": mock.MagicMock(),
            "trulens.providers.openai": mock.MagicMock(),
        },
    ):
        from mlflow.genai.scorers.trulens import get_scorer

        with pytest.raises(MlflowException, match="Unknown TruLens metric"):
            get_scorer("InvalidMetric", model="openai:/gpt-4")


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


def test_trulens_assessment_source(mock_trulens_modules, mock_trulens_provider):
    with mock.patch.dict("sys.modules", mock_trulens_modules):
        from mlflow.genai.scorers.trulens import Groundedness

        scorer = Groundedness(model="openai:/gpt-4")
        result = scorer(
            outputs="Test output.",
            expectations={"context": "Test context."},
        )

        assert result.source is not None
        assert result.source.source_id == "openai:/gpt-4"


def test_trulens_rationale_formatting():
    from mlflow.genai.scorers.trulens.utils import format_trulens_rationale

    # Test with simple dict
    result = format_trulens_rationale({"reason": "Test reason"})
    assert "reason: Test reason" in result

    # Test with empty dict
    result = format_trulens_rationale({})
    assert "No detailed reasoning available" in result

    # Test with None
    result = format_trulens_rationale(None)
    assert "No detailed reasoning available" in result

    # Test with list value
    result = format_trulens_rationale({"reasons": ["reason1", "reason2"]})
    assert "reason1" in result
    assert "reason2" in result


def test_trulens_with_litellm_provider(mock_trulens_provider):
    mock_litellm_class = mock.MagicMock(return_value=mock_trulens_provider)
    mock_providers_litellm = mock.MagicMock()
    mock_providers_litellm.LiteLLM = mock_litellm_class

    mock_modules = {
        "trulens": mock.MagicMock(),
        "trulens.providers": mock.MagicMock(),
        "trulens.providers.openai": mock.MagicMock(),
        "trulens.providers.litellm": mock_providers_litellm,
    }

    with mock.patch.dict("sys.modules", mock_modules):
        from mlflow.genai.scorers.trulens import Groundedness

        scorer = Groundedness(model="litellm:/gpt-4")
        result = scorer(
            outputs="Test output",
            expectations={"context": "Test context"},
        )

        assert isinstance(result, Feedback)
        mock_litellm_class.assert_called_once()
