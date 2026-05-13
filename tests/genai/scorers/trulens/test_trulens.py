from unittest.mock import Mock, patch

import pytest
import trulens  # noqa: F401

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.genai.scorers.base import Scorer, ScorerKind
from mlflow.genai.scorers.trulens import Groundedness


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


def test_trulens_scorer_provider_is_real_instance(monkeypatch):
    from trulens.feedback.llm_provider import LLMProvider

    from mlflow.genai.scorers.trulens import Groundedness

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    scorer = Groundedness(model="openai:/gpt-4")
    assert isinstance(scorer._provider, LLMProvider)


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


# --- Model adapter tests ---


def test_gateway_provider_create_chat_completion():
    from mlflow.genai.scorers.llm_backend import ScorerLLMClient
    from mlflow.genai.scorers.trulens.models import _create_gateway_provider

    with patch("mlflow.genai.scorers.llm_backend._get_provider_instance") as mock_gpi:
        backend = ScorerLLMClient("openai:/gpt-4")
    mock_gpi.assert_called_once()
    provider = _create_gateway_provider(backend)

    with patch(
        "mlflow.genai.scorers.llm_backend._call_llm_provider_api",
        return_value="The answer is 42.",
    ) as mock_call:
        result = provider._create_chat_completion(prompt="What is the answer?")

    assert result == "The answer is 42."
    mock_call.assert_called_once()


def test_gateway_provider_handles_messages():
    from mlflow.genai.scorers.llm_backend import ScorerLLMClient
    from mlflow.genai.scorers.trulens.models import _create_gateway_provider

    with patch("mlflow.genai.scorers.llm_backend._get_provider_instance") as mock_gpi:
        backend = ScorerLLMClient("openai:/gpt-4")
    mock_gpi.assert_called_once()
    provider = _create_gateway_provider(backend)
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
    ]

    with patch(
        "mlflow.genai.scorers.llm_backend._call_llm_provider_api",
        return_value="Hi there!",
    ) as mock_call:
        result = provider._create_chat_completion(messages=messages)

    assert result == "Hi there!"
    mock_call.assert_called_once()


@pytest.mark.parametrize(
    ("model_uri", "env_var"),
    [
        ("openai:/gpt-4", "OPENAI_API_KEY"),
        ("anthropic:/claude-3", "ANTHROPIC_API_KEY"),
    ],
)
def test_create_trulens_provider_uses_gateway_for_supported(model_uri, env_var, monkeypatch):
    from trulens.feedback.llm_provider import LLMProvider

    from mlflow.genai.scorers.trulens.models import create_trulens_provider

    monkeypatch.setenv(env_var, "test-key")
    provider = create_trulens_provider(model_uri)
    assert isinstance(provider, LLMProvider)
    assert "gateway" in provider.endpoint.name


def test_create_trulens_provider_uses_databricks_for_bare_uri():
    from trulens.feedback.llm_provider import LLMProvider

    from mlflow.genai.scorers.trulens.models import create_trulens_provider

    provider = create_trulens_provider("databricks")
    assert isinstance(provider, LLMProvider)


def test_high_level_scorer_call_chain(mock_provider):
    """Exercises the full call chain as recommended in docs/blogs:
    Groundedness(model=...) → scorer(outputs=..., expectations=...)
    """
    with patch("mlflow.genai.scorers.trulens.create_trulens_provider", return_value=mock_provider):
        from mlflow.genai.scorers.trulens import Groundedness

        scorer = Groundedness(model="openai:/gpt-4", threshold=0.7)

    mock_provider.groundedness_measure_with_cot_reasons.return_value = (
        0.9,
        {"reason": "Grounded"},
    )

    feedback = scorer(
        outputs="Paris is the capital of France.",
        expectations={"context": "France is a country. Its capital is Paris."},
    )

    assert isinstance(feedback, Feedback)
    assert feedback.name == "Groundedness"
    assert feedback.value == CategoricalRating.YES
    assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert feedback.source.source_id == "openai:/gpt-4"


def test_trulens_scorer_kind_is_third_party(mock_provider):
    with patch("mlflow.genai.scorers.trulens.create_trulens_provider", return_value=mock_provider):
        assert Groundedness(model="openai:/gpt-4").kind == ScorerKind.THIRD_PARTY


def test_trulens_scorer_serialization_round_trip(mock_provider):
    with patch("mlflow.genai.scorers.trulens.create_trulens_provider", return_value=mock_provider):
        scorer = Groundedness(model="openai:/gpt-4", threshold=0.7)
        dump = scorer.model_dump()
        assert dump["third_party_scorer_data"]["class"] == "Groundedness"
        assert dump["third_party_scorer_data"]["metric_name"] == "Groundedness"
        assert dump["third_party_scorer_data"]["model"] == "openai:/gpt-4"
        assert dump["third_party_scorer_data"]["kwargs"]["threshold"] == 0.7

        restored = Scorer.model_validate(dump)
        assert isinstance(restored, Groundedness)
        assert restored.kind == ScorerKind.THIRD_PARTY
        assert restored._model == "openai:/gpt-4"
        assert restored._threshold == 0.7


def test_trulens_scorer_register_blocked_on_databricks(mock_provider):
    with patch("mlflow.genai.scorers.trulens.create_trulens_provider", return_value=mock_provider):
        scorer = Groundedness(model="openai:/gpt-4")
        with patch("mlflow.genai.scorers.base.is_databricks_uri", return_value=True):
            with pytest.raises(MlflowException, match="Third-party scorer registration"):
                scorer.register(name="groundedness")
