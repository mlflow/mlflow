from unittest.mock import Mock, patch

import phoenix.evals as phoenix_evals
import pytest

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.base import Scorer, ScorerKind
from mlflow.genai.scorers.phoenix import Hallucination


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


def test_high_level_scorer_call_chain(monkeypatch):
    """Exercises the full call chain: Hallucination(model=...) -> scorer(inputs=..., outputs=...)
    as recommended in docs.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    from mlflow.genai.scorers.phoenix import Hallucination

    scorer = Hallucination(model="openai:/gpt-4")

    with patch.object(
        scorer._evaluator, "evaluate", return_value=("factual", 0.95, "Grounded.")
    ) as mock_evaluate:
        feedback = scorer(
            inputs="What is MLflow?",
            outputs="MLflow is an open-source platform for managing ML workflows.",
            expectations={"context": "MLflow is an open-source ML platform."},
        )

    mock_evaluate.assert_called_once()
    assert isinstance(feedback, Feedback)
    assert feedback.name == "Hallucination"
    assert feedback.value == "factual"
    assert feedback.metadata["score"] == 0.95
    assert feedback.rationale == "Grounded."
    assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert feedback.source.source_id == "openai:/gpt-4"


def test_phoenix_scorer_kind_is_third_party(mock_model):
    with patch("mlflow.genai.scorers.phoenix.create_phoenix_model", return_value=mock_model):
        assert Hallucination(model="openai:/gpt-4").kind == ScorerKind.THIRD_PARTY


def test_phoenix_scorer_serialization_round_trip(mock_model):
    with patch("mlflow.genai.scorers.phoenix.create_phoenix_model", return_value=mock_model):
        scorer = Hallucination(model="openai:/gpt-4")
        dump = scorer.model_dump()
        assert dump["third_party_scorer_data"]["class"] == "Hallucination"
        assert dump["third_party_scorer_data"]["metric_name"] == "Hallucination"
        assert dump["third_party_scorer_data"]["model"] == "openai:/gpt-4"

        restored = Scorer.model_validate(dump)
        assert isinstance(restored, Hallucination)
        assert restored.kind == ScorerKind.THIRD_PARTY
        assert restored._model == "openai:/gpt-4"


def test_phoenix_scorer_register_blocked_on_databricks(mock_model):
    with patch("mlflow.genai.scorers.phoenix.create_phoenix_model", return_value=mock_model):
        scorer = Hallucination(model="openai:/gpt-4")
        with patch("mlflow.genai.scorers.base.is_databricks_uri", return_value=True):
            with pytest.raises(MlflowException, match="Third-party scorer registration"):
                scorer.register(name="hallucination")
