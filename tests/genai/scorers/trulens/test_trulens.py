from unittest.mock import Mock, patch

import pytest
import trulens  # noqa: F401

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.genai.judges.utils import CategoricalRating


@pytest.fixture
def mock_provider():
    mock = Mock()
    mock.groundedness_measure_with_cot_reasons = Mock()
    mock.context_relevance_with_cot_reasons = Mock()
    mock.relevance_with_cot_reasons = Mock()
    mock.coherence_with_cot_reasons = Mock()
    return mock


@pytest.mark.parametrize(
    ("scorer_name", "method_name", "score", "expected_value"),
    [
        ("Groundedness", "groundedness_measure_with_cot_reasons", 0.8, CategoricalRating.YES),
        ("ContextRelevance", "context_relevance_with_cot_reasons", 0.7, CategoricalRating.YES),
        ("AnswerRelevance", "relevance_with_cot_reasons", 0.9, CategoricalRating.YES),
        ("Coherence", "coherence_with_cot_reasons", 0.85, CategoricalRating.YES),
        ("Groundedness", "groundedness_measure_with_cot_reasons", 0.3, CategoricalRating.NO),
    ],
)
def test_trulens_scorer(mock_provider, scorer_name, method_name, score, expected_value):
    with patch("mlflow.genai.scorers.trulens.create_trulens_provider", return_value=mock_provider):
        from mlflow.genai.scorers import trulens

        scorer_cls = getattr(trulens, scorer_name)
        scorer = scorer_cls(model="openai:/gpt-4")

    method = getattr(mock_provider, method_name)
    method.return_value = (score, {"reason": "Test reason"})

    result = scorer(
        inputs="test input",
        outputs="test output",
        expectations={"context": "test context"},
    )

    assert isinstance(result, Feedback)
    assert result.name == scorer_name
    assert result.value == expected_value
    assert result.rationale == "reason: Test reason"
    assert result.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert result.source.source_id == "openai:/gpt-4"
    assert result.metadata == {
        "mlflow.scorer.framework": "trulens",
        "score": score,
        "threshold": 0.5,
    }


def test_trulens_scorer_custom_threshold(mock_provider):
    with patch("mlflow.genai.scorers.trulens.create_trulens_provider", return_value=mock_provider):
        from mlflow.genai.scorers.trulens import Groundedness

        scorer = Groundedness(model="openai:/gpt-4", threshold=0.7)

    mock_provider.groundedness_measure_with_cot_reasons.return_value = (0.6, {"reason": "Moderate"})

    result = scorer(
        outputs="test output",
        expectations={"context": "test context"},
    )

    assert result.value == CategoricalRating.NO
    assert result.metadata["threshold"] == 0.7


def test_trulens_scorer_none_reasons(mock_provider):
    with patch("mlflow.genai.scorers.trulens.create_trulens_provider", return_value=mock_provider):
        from mlflow.genai.scorers.trulens import Groundedness

        scorer = Groundedness(model="openai:/gpt-4")

    mock_provider.groundedness_measure_with_cot_reasons.return_value = (0.9, None)

    result = scorer(
        outputs="test output",
        expectations={"context": "test context"},
    )

    assert result.rationale is None


def test_trulens_get_scorer(mock_provider):
    with patch("mlflow.genai.scorers.trulens.create_trulens_provider", return_value=mock_provider):
        from mlflow.genai.scorers.trulens import get_scorer

        scorer = get_scorer("Groundedness", model="openai:/gpt-4")

    mock_provider.groundedness_measure_with_cot_reasons.return_value = (0.9, {"reason": "Good"})

    result = scorer(
        outputs="test output",
        expectations={"context": "test context"},
    )

    assert isinstance(result, Feedback)
    assert result.name == "Groundedness"


def test_trulens_scorer_provider_is_real_instance():
    from trulens.providers.litellm import LiteLLM

    from mlflow.genai.scorers.trulens import Groundedness

    try:
        scorer = Groundedness(model="openai:/gpt-4")
        assert isinstance(scorer._provider, LiteLLM)
    except AttributeError as e:
        # TruLens LiteLLM provider has an instrumentation bug with CallTypes enum
        if "CallTypes" in str(e):
            pytest.skip("TruLens LiteLLM instrumentation bug - see TruLens issue tracker")
        raise


def test_trulens_scorer_error_handling(mock_provider):
    with patch("mlflow.genai.scorers.trulens.create_trulens_provider", return_value=mock_provider):
        from mlflow.genai.scorers.trulens import Groundedness

        scorer = Groundedness(model="openai:/gpt-4")

    mock_provider.groundedness_measure_with_cot_reasons.side_effect = RuntimeError(
        "Evaluation failed"
    )

    result = scorer(
        outputs="test output",
        expectations={"context": "test context"},
    )

    assert isinstance(result, Feedback)
    assert result.error is not None
    assert "Evaluation failed" in str(result.error)
    assert result.metadata == {"mlflow.scorer.framework": "trulens"}
