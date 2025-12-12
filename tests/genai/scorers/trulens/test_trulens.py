from unittest import mock

import pytest

from mlflow.entities.assessment import Feedback
from mlflow.exceptions import MlflowException


def test_trulens_groundedness_scorer_requires_trulens():
    with mock.patch.dict(
        "sys.modules",
        {"trulens": None, "trulens.providers": None, "trulens.providers.openai": None},
    ):
        from mlflow.genai.scorers.trulens.trulens import TruLensGroundednessScorer

        scorer = TruLensGroundednessScorer()
        with pytest.raises(MlflowException, match="trulens"):
            scorer(
                outputs="test output",
                context="test context",
            )


def test_trulens_context_relevance_scorer_requires_trulens():
    with mock.patch.dict(
        "sys.modules",
        {"trulens": None, "trulens.providers": None, "trulens.providers.openai": None},
    ):
        from mlflow.genai.scorers.trulens.trulens import TruLensContextRelevanceScorer

        scorer = TruLensContextRelevanceScorer()
        with pytest.raises(MlflowException, match="trulens"):
            scorer(
                inputs={"query": "test"},
                context="test context",
            )


def test_trulens_answer_relevance_scorer_requires_trulens():
    with mock.patch.dict(
        "sys.modules",
        {"trulens": None, "trulens.providers": None, "trulens.providers.openai": None},
    ):
        from mlflow.genai.scorers.trulens.trulens import TruLensAnswerRelevanceScorer

        scorer = TruLensAnswerRelevanceScorer()
        with pytest.raises(MlflowException, match="trulens"):
            scorer(
                inputs={"query": "test"},
                outputs="test output",
            )


def test_trulens_groundedness_scorer_default_name():
    from mlflow.genai.scorers.trulens.trulens import TruLensGroundednessScorer

    scorer = TruLensGroundednessScorer()
    assert scorer.name == "trulens_groundedness"


def test_trulens_context_relevance_scorer_default_name():
    from mlflow.genai.scorers.trulens.trulens import TruLensContextRelevanceScorer

    scorer = TruLensContextRelevanceScorer()
    assert scorer.name == "trulens_context_relevance"


def test_trulens_answer_relevance_scorer_default_name():
    from mlflow.genai.scorers.trulens.trulens import TruLensAnswerRelevanceScorer

    scorer = TruLensAnswerRelevanceScorer()
    assert scorer.name == "trulens_answer_relevance"


def test_trulens_scorer_custom_name():
    from mlflow.genai.scorers.trulens.trulens import TruLensGroundednessScorer

    scorer = TruLensGroundednessScorer(name="custom_groundedness_check")
    assert scorer.name == "custom_groundedness_check"


def test_trulens_scorer_custom_model():
    from mlflow.genai.scorers.trulens.trulens import TruLensGroundednessScorer

    scorer = TruLensGroundednessScorer(model_name="gpt-4", model_provider="openai")
    assert scorer.model_name == "gpt-4"
    assert scorer.model_provider == "openai"


def test_trulens_scorer_litellm_provider():
    from mlflow.genai.scorers.trulens.trulens import TruLensGroundednessScorer

    scorer = TruLensGroundednessScorer(model_provider="litellm")
    assert scorer.model_provider == "litellm"


def test_trulens_scorer_unsupported_provider():
    from mlflow.genai.scorers.trulens.trulens import TruLensGroundednessScorer

    scorer = TruLensGroundednessScorer(model_provider="unsupported")

    mock_providers_openai = mock.MagicMock()
    with mock.patch.dict(
        "sys.modules",
        {
            "trulens": mock.MagicMock(),
            "trulens.providers": mock.MagicMock(),
            "trulens.providers.openai": mock_providers_openai,
        },
    ):
        with pytest.raises(MlflowException, match="Unsupported model provider"):
            scorer._get_trulens_provider()


@pytest.fixture
def mock_trulens_provider():
    # TruLens returns Tuple[float, dict] with score in 0-1 range
    mock_provider = mock.MagicMock()
    mock_provider.groundedness_measure_with_cot_reasons.return_value = (
        0.9,  # Score in 0-1 range
        {"reason": "Statement is well supported by the context."},
    )
    mock_provider.context_relevance_with_cot_reasons.return_value = (
        0.85,  # Score in 0-1 range
        {"reason": "The context is highly relevant to the query."},
    )
    mock_provider.relevance_with_cot_reasons.return_value = (
        0.95,  # Score in 0-1 range
        {"reason": "The answer directly addresses the question."},
    )
    return mock_provider


def test_trulens_groundedness_scorer_with_mock(mock_trulens_provider):
    mock_openai_class = mock.MagicMock(return_value=mock_trulens_provider)

    mock_providers_openai = mock.MagicMock()
    mock_providers_openai.OpenAI = mock_openai_class

    with mock.patch.dict(
        "sys.modules",
        {
            "trulens": mock.MagicMock(),
            "trulens.providers": mock.MagicMock(),
            "trulens.providers.openai": mock_providers_openai,
        },
    ):
        from mlflow.genai.scorers.trulens.trulens import TruLensGroundednessScorer

        scorer = TruLensGroundednessScorer()
        result = scorer(
            outputs="The Eiffel Tower is 330 meters tall.",
            context="The Eiffel Tower stands at 330 meters.",
        )

        assert isinstance(result, Feedback)
        assert result.name == "trulens_groundedness"
        # Score passed through directly from TruLens
        assert result.value == 0.9
        assert "supported" in result.rationale.lower()


def test_trulens_groundedness_scorer_with_list_context(mock_trulens_provider):
    mock_openai_class = mock.MagicMock(return_value=mock_trulens_provider)

    mock_providers_openai = mock.MagicMock()
    mock_providers_openai.OpenAI = mock_openai_class

    with mock.patch.dict(
        "sys.modules",
        {
            "trulens": mock.MagicMock(),
            "trulens.providers": mock.MagicMock(),
            "trulens.providers.openai": mock_providers_openai,
        },
    ):
        from mlflow.genai.scorers.trulens.trulens import TruLensGroundednessScorer

        scorer = TruLensGroundednessScorer()
        result = scorer(
            outputs="The Eiffel Tower is 330 meters tall.",
            context=["The Eiffel Tower stands at 330 meters.", "It is located in Paris."],
        )

        assert isinstance(result, Feedback)
        # Score passed through directly from TruLens
        assert result.value == 0.9

        # Verify the context list was joined with newlines
        call_args = mock_trulens_provider.groundedness_measure_with_cot_reasons.call_args
        assert (
            "The Eiffel Tower stands at 330 meters.\nIt is located in Paris."
            in call_args.kwargs["source"]
        )


def test_trulens_context_relevance_scorer_with_mock(mock_trulens_provider):
    mock_openai_class = mock.MagicMock(return_value=mock_trulens_provider)

    mock_providers_openai = mock.MagicMock()
    mock_providers_openai.OpenAI = mock_openai_class

    with mock.patch.dict(
        "sys.modules",
        {
            "trulens": mock.MagicMock(),
            "trulens.providers": mock.MagicMock(),
            "trulens.providers.openai": mock_providers_openai,
        },
    ):
        from mlflow.genai.scorers.trulens.trulens import TruLensContextRelevanceScorer

        scorer = TruLensContextRelevanceScorer()
        result = scorer(
            inputs={"query": "What is the capital of France?"},
            context="Paris is the capital and largest city of France.",
        )

        assert isinstance(result, Feedback)
        assert result.name == "trulens_context_relevance"
        # Score passed through directly from TruLens
        assert result.value == 0.85
        assert "relevant" in result.rationale.lower()


def test_trulens_answer_relevance_scorer_with_mock(mock_trulens_provider):
    mock_openai_class = mock.MagicMock(return_value=mock_trulens_provider)

    mock_providers_openai = mock.MagicMock()
    mock_providers_openai.OpenAI = mock_openai_class

    with mock.patch.dict(
        "sys.modules",
        {
            "trulens": mock.MagicMock(),
            "trulens.providers": mock.MagicMock(),
            "trulens.providers.openai": mock_providers_openai,
        },
    ):
        from mlflow.genai.scorers.trulens.trulens import TruLensAnswerRelevanceScorer

        scorer = TruLensAnswerRelevanceScorer()
        result = scorer(
            inputs={"query": "What is machine learning?"},
            outputs="Machine learning is a branch of AI that enables systems to learn from data.",
        )

        assert isinstance(result, Feedback)
        assert result.name == "trulens_answer_relevance"
        # Score passed through directly from TruLens
        assert result.value == 0.95
        assert "addresses" in result.rationale.lower()


def test_trulens_score_validation():
    from mlflow.genai.scorers.trulens.trulens import _TruLensScorerBase

    # Create a minimal instance to test validation
    class TestScorer(_TruLensScorerBase):
        name: str = "test"

        def __call__(self, **kwargs):
            pass

    scorer = TestScorer()

    # Test pass-through for valid 0-1 scores
    assert scorer._validate_score(0.0) == 0.0
    assert scorer._validate_score(0.5) == 0.5
    assert scorer._validate_score(1.0) == 1.0
    # Test clamping with warning for out-of-range values
    assert scorer._validate_score(1.5) == 1.0  # Over max, clamped with warning
    assert scorer._validate_score(-0.5) == 0.0  # Under min, clamped with warning


def test_trulens_rationale_formatting():
    from mlflow.genai.scorers.trulens.trulens import _TruLensScorerBase

    class TestScorer(_TruLensScorerBase):
        name: str = "test"

        def __call__(self, **kwargs):
            pass

    scorer = TestScorer()

    # Test with simple dict
    result = scorer._format_rationale({"reason": "Test reason"})
    assert "reason: Test reason" in result

    # Test with empty dict
    result = scorer._format_rationale({})
    assert "No detailed reasoning available" in result

    # Test with None
    result = scorer._format_rationale(None)
    assert "No detailed reasoning available" in result

    # Test with list value
    result = scorer._format_rationale({"reasons": ["reason1", "reason2"]})
    assert "reason1" in result
    assert "reason2" in result


def test_trulens_scorer_exports():
    from mlflow.genai.scorers.trulens import (
        TruLensAnswerRelevanceScorer,
        TruLensContextRelevanceScorer,
        TruLensGroundednessScorer,
    )

    assert TruLensGroundednessScorer is not None
    assert TruLensContextRelevanceScorer is not None
    assert TruLensAnswerRelevanceScorer is not None


def test_trulens_scorers_available_from_main_module():
    from mlflow.genai.scorers import (
        TruLensAnswerRelevanceScorer,
        TruLensContextRelevanceScorer,
        TruLensGroundednessScorer,
    )

    assert TruLensGroundednessScorer is not None
    assert TruLensContextRelevanceScorer is not None
    assert TruLensAnswerRelevanceScorer is not None


def test_trulens_scorer_with_litellm_provider(mock_trulens_provider):
    mock_litellm_class = mock.MagicMock(return_value=mock_trulens_provider)

    mock_providers_litellm = mock.MagicMock()
    mock_providers_litellm.LiteLLM = mock_litellm_class

    mock_providers_openai = mock.MagicMock()

    with mock.patch.dict(
        "sys.modules",
        {
            "trulens": mock.MagicMock(),
            "trulens.providers": mock.MagicMock(),
            "trulens.providers.openai": mock_providers_openai,
            "trulens.providers.litellm": mock_providers_litellm,
        },
    ):
        from mlflow.genai.scorers.trulens.trulens import TruLensGroundednessScorer

        scorer = TruLensGroundednessScorer(model_provider="litellm")
        result = scorer(
            outputs="Test output",
            context="Test context",
        )

        assert isinstance(result, Feedback)
        # Score passed through directly from TruLens
        assert result.value == 0.9
        mock_litellm_class.assert_called_once()


def test_trulens_coherence_scorer_requires_trulens():
    with mock.patch.dict(
        "sys.modules",
        {"trulens": None, "trulens.providers": None, "trulens.providers.openai": None},
    ):
        from mlflow.genai.scorers.trulens.trulens import TruLensCoherenceScorer

        scorer = TruLensCoherenceScorer()
        with pytest.raises(MlflowException, match="trulens"):
            scorer(outputs="test output")


def test_trulens_coherence_scorer_default_name():
    from mlflow.genai.scorers.trulens.trulens import TruLensCoherenceScorer

    scorer = TruLensCoherenceScorer()
    assert scorer.name == "trulens_coherence"


def test_trulens_coherence_scorer_with_mock(mock_trulens_provider):
    # Add coherence method to mock
    mock_trulens_provider.coherence_with_cot_reasons.return_value = (
        0.8,  # Score in 0-1 range
        {"reason": "The text flows logically and is well-structured."},
    )

    mock_openai_class = mock.MagicMock(return_value=mock_trulens_provider)

    mock_providers_openai = mock.MagicMock()
    mock_providers_openai.OpenAI = mock_openai_class

    with mock.patch.dict(
        "sys.modules",
        {
            "trulens": mock.MagicMock(),
            "trulens.providers": mock.MagicMock(),
            "trulens.providers.openai": mock_providers_openai,
        },
    ):
        from mlflow.genai.scorers.trulens.trulens import TruLensCoherenceScorer

        scorer = TruLensCoherenceScorer()
        result = scorer(
            outputs="Machine learning is a branch of AI. It enables systems to learn.",
        )

        assert isinstance(result, Feedback)
        assert result.name == "trulens_coherence"
        # Score passed through directly from TruLens
        assert result.value == 0.8
        assert "logical" in result.rationale.lower() or "structured" in result.rationale.lower()


def test_trulens_coherence_scorer_export():
    from mlflow.genai.scorers.trulens import TruLensCoherenceScorer

    assert TruLensCoherenceScorer is not None


def test_trulens_coherence_scorer_available_from_main_module():
    from mlflow.genai.scorers import TruLensCoherenceScorer

    assert TruLensCoherenceScorer is not None
