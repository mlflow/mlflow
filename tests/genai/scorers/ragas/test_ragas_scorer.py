from unittest.mock import MagicMock, patch

import pytest
from ragas.embeddings.base import BaseRagasEmbedding

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.genai.scorers import FRAMEWORK_METADATA_KEY
from mlflow.genai.scorers.base import ScorerKind
from mlflow.genai.scorers.ragas import (
    AgentGoalAccuracyWithoutReference,
    AgentGoalAccuracyWithReference,
    AnswerRelevancy,
    AspectCritic,
    ContextEntityRecall,
    ContextPrecision,
    ContextRecall,
    DiscreteMetric,
    ExactMatch,
    FactualCorrectness,
    Faithfulness,
    InstanceSpecificRubrics,
    NoiseSensitivity,
    RagasScorer,
    RougeScore,
    RubricsScore,
    SemanticSimilarity,
    StringPresence,
    SummarizationScore,
    ToolCallAccuracy,
    ToolCallF1,
    TopicAdherence,
    get_scorer,
)
from mlflow.telemetry.client import TelemetryClient
from mlflow.telemetry.events import GenAIEvaluateEvent, ScorerCallEvent

from tests.telemetry.helper_functions import validate_telemetry_record


def make_mock_ascore(return_value=1.0, error=None):
    async def mock_ascore(response=None, reference=None):
        if error:
            raise error
        return return_value

    return mock_ascore


@pytest.fixture(autouse=True)
def mock_get_telemetry_client(mock_telemetry_client: TelemetryClient):
    with patch(
        "mlflow.telemetry.track.get_telemetry_client",
        return_value=mock_telemetry_client,
    ):
        yield


def test_ragas_scorer_with_exact_match_metric():
    judge = get_scorer("ExactMatch")
    result = judge(
        inputs="What is MLflow?",
        outputs="MLflow is a platform",
        expectations={"expected_output": "MLflow is a platform"},
    )

    assert isinstance(result, Feedback)
    assert result.name == "ExactMatch"
    assert result.value == 1.0
    assert result.source.source_type == AssessmentSourceType.CODE
    assert result.source.source_id == "ExactMatch"
    assert result.metadata[FRAMEWORK_METADATA_KEY] == "ragas"


def test_ragas_scorer_handles_failure_with_exact_match():
    judge = get_scorer("ExactMatch")
    result = judge(
        inputs="What is MLflow?",
        outputs="MLflow is different",
        expectations={"expected_output": "MLflow is a platform"},
    )

    assert result.value == 0.0


def test_deterministic_metric_does_not_require_model():
    judge = get_scorer("ExactMatch")
    result = judge(
        outputs="test",
        expectations={"expected_output": "test"},
    )

    assert result.value == 1.0


def test_ragas_scorer_with_threshold_returns_categorical():
    judge = get_scorer("ExactMatch")
    judge._metric.threshold = 0.5

    with patch.object(judge._metric, "ascore", make_mock_ascore(0.8)):
        result = judge(
            inputs="What is MLflow?",
            outputs="MLflow is a platform",
            expectations={"expected_output": "MLflow is a platform"},
        )
        assert result.value == CategoricalRating.YES
        assert result.metadata["score"] == 0.8
        assert result.metadata["threshold"] == 0.5


def test_ragas_scorer_with_threshold_returns_no_when_below():
    judge = get_scorer("ExactMatch")
    judge._metric.threshold = 0.5

    with patch.object(judge._metric, "ascore", make_mock_ascore(0.0)):
        result = judge(
            inputs="What is MLflow?",
            outputs="Databricks is a company",
            expectations={"expected_output": "MLflow is a platform"},
        )

        assert result.value == CategoricalRating.NO
        assert result.metadata["score"] == 0.0
        assert result.metadata["threshold"] == 0.5


def test_ragas_scorer_without_threshold_returns_float():
    judge = get_scorer("ExactMatch")
    result = judge(
        outputs="test",
        expectations={"expected_output": "test"},
    )
    assert isinstance(result.value, float)
    assert result.value == 1.0
    assert "threshold" not in result.metadata


def test_ragas_scorer_returns_error_feedback_on_exception():
    judge = get_scorer("ExactMatch")

    with patch.object(judge._metric, "ascore", make_mock_ascore(error=RuntimeError("Test error"))):
        result = judge(inputs="What is MLflow?", outputs="Test output")

    assert isinstance(result, Feedback)
    assert result.name == "ExactMatch"
    assert result.value is None
    assert result.error is not None
    assert result.error.error_code == "RuntimeError"
    assert result.error.error_message == "Test error"
    assert result.source.source_type == AssessmentSourceType.CODE


def test_unknown_metric_raises_error():
    with pytest.raises(MlflowException, match="Unknown metric: 'NonExistentMetric'"):
        get_scorer("NonExistentMetric")


def test_missing_reference_parameter_returns_mlflow_error():
    judge = get_scorer("ContextPrecision")
    result = judge(
        inputs="What is MLflow?",
        expectations={"expected_output": "MLflow is a platform"},
    )
    assert isinstance(result, Feedback)
    assert result.error is not None
    assert "ContextPrecision" in result.error.error_message  # metric name
    assert "trace with retrieval spans" in result.error.error_message  # mlflow error message


@pytest.mark.parametrize(
    ("scorer_class", "expected_metric_name", "metric_kwargs"),
    [
        # RAG Metrics
        (ContextPrecision, "ContextPrecision", {}),
        (ContextRecall, "ContextRecall", {}),
        (ContextEntityRecall, "ContextEntityRecall", {}),
        (NoiseSensitivity, "NoiseSensitivity", {}),
        (Faithfulness, "Faithfulness", {}),
        # Comparison Metrics
        (FactualCorrectness, "FactualCorrectness", {}),
        (RougeScore, "RougeScore", {}),
        (StringPresence, "StringPresence", {}),
        (ExactMatch, "ExactMatch", {}),
        # General Purpose Metrics
        (AspectCritic, "AspectCritic", {"name": "test", "definition": "test"}),
        (DiscreteMetric, "DiscreteMetric", {"name": "test", "prompt": "test"}),
        (RubricsScore, "RubricsScore", {}),
        (InstanceSpecificRubrics, "InstanceSpecificRubrics", {}),
        # Summarization Metrics
        (SummarizationScore, "SummarizationScore", {}),
        # Agentic Metrics
        (TopicAdherence, "TopicAdherence", {}),
        (ToolCallAccuracy, "ToolCallAccuracy", {}),
        (ToolCallF1, "ToolCallF1", {}),
        (AgentGoalAccuracyWithReference, "AgentGoalAccuracyWithReference", {}),
        (AgentGoalAccuracyWithoutReference, "AgentGoalAccuracyWithoutReference", {}),
        # Embeddings-based Metrics
        (
            AnswerRelevancy,
            "AnswerRelevancy",
            {"embeddings": MagicMock(spec=BaseRagasEmbedding)},
        ),
        (
            SemanticSimilarity,
            "SemanticSimilarity",
            {"embeddings": MagicMock(spec=BaseRagasEmbedding)},
        ),
    ],
)
def test_namespaced_class_properly_instantiates(scorer_class, expected_metric_name, metric_kwargs):
    assert issubclass(scorer_class, RagasScorer)
    assert scorer_class.metric_name == expected_metric_name
    scorer = scorer_class(**metric_kwargs)
    assert isinstance(scorer, RagasScorer)
    assert scorer.name == expected_metric_name


def test_ragas_scorer_kind_property():
    scorer = get_scorer("ExactMatch")
    assert scorer.kind == ScorerKind.THIRD_PARTY


@pytest.mark.parametrize("method_name", ["register", "start", "update", "stop"])
def test_ragas_scorer_registration_methods_not_supported(method_name):
    scorer = get_scorer("ExactMatch")
    method = getattr(scorer, method_name)

    with pytest.raises(MlflowException, match=f"'{method_name}\\(\\)' is not supported"):
        method()


def test_ragas_scorer_align_not_supported():
    scorer = get_scorer("ExactMatch")

    with pytest.raises(MlflowException, match="'align\\(\\)' is not supported"):
        scorer.align()


def test_ragas_scorer_kind_property_with_llm_metric():
    scorer = Faithfulness()
    assert scorer.kind == ScorerKind.THIRD_PARTY


@pytest.mark.parametrize(
    ("scorer_factory", "expected_class"),
    [
        (lambda: ExactMatch(), "Ragas:ExactMatch"),
        (lambda: get_scorer("ExactMatch"), "Ragas:ExactMatch"),
    ],
    ids=["direct_instantiation", "get_scorer"],
)
def test_ragas_scorer_telemetry_direct_call(
    enable_telemetry_in_tests,
    mock_requests,
    mock_telemetry_client,
    scorer_factory,
    expected_class,
):
    ragas_scorer = scorer_factory()

    with patch.object(ragas_scorer._metric, "ascore", make_mock_ascore(1.0)):
        result = ragas_scorer(
            inputs="What is MLflow?",
            outputs="MLflow is a platform",
            expectations={"expected_output": "MLflow is a platform"},
        )

    assert result.value == 1.0

    mock_telemetry_client.flush()

    validate_telemetry_record(
        mock_telemetry_client,
        mock_requests,
        ScorerCallEvent.name,
        {
            "scorer_class": expected_class,
            "scorer_kind": "third_party",
            "is_session_level_scorer": False,
            "callsite": "direct_scorer_call",
            "has_feedback_error": False,
        },
    )


@pytest.mark.parametrize(
    ("scorer_factory", "expected_class"),
    [
        (lambda: ExactMatch(), "Ragas:ExactMatch"),
        (lambda: get_scorer("ExactMatch"), "Ragas:ExactMatch"),
    ],
    ids=["direct_instantiation", "get_scorer"],
)
def test_ragas_scorer_telemetry_in_genai_evaluate(
    enable_telemetry_in_tests,
    mock_requests,
    mock_telemetry_client,
    scorer_factory,
    expected_class,
):
    ragas_scorer = scorer_factory()

    data = [
        {
            "inputs": {"question": "What is MLflow?"},
            "outputs": "MLflow is a platform",
            "expectations": {"expected_output": "MLflow is a platform"},
        }
    ]

    with patch.object(ragas_scorer._metric, "ascore", make_mock_ascore(1.0)):
        mlflow.genai.evaluate(data=data, scorers=[ragas_scorer])

    validate_telemetry_record(
        mock_telemetry_client,
        mock_requests,
        GenAIEvaluateEvent.name,
        {
            "predict_fn_provided": False,
            "scorer_info": [
                {"class": expected_class, "kind": "third_party", "scope": "response"},
            ],
            "eval_data_type": "list[dict]",
            "eval_data_size": 1,
            "eval_data_provided_fields": ["expectations", "inputs", "outputs"],
        },
    )


@pytest.mark.parametrize(
    ("scorer_class", "expectations", "sample_assertion"),
    [
        (
            ToolCallAccuracy,
            {
                "expected_tool_calls": [
                    {"name": "weather_check", "arguments": {"location": "Paris"}},
                ]
            },
            lambda sample: sample.reference_tool_calls is not None,
        ),
        (
            TopicAdherence,
            {"reference_topics": ["machine learning", "data science"]},
            lambda sample: sample.reference_topics == ["machine learning", "data science"],
        ),
        (
            AgentGoalAccuracyWithReference,
            {"expected_output": "Table booked at a Chinese restaurant for 8pm"},
            lambda sample: sample.reference == "Table booked at a Chinese restaurant for 8pm",
        ),
    ],
)
def test_agentic_scorer_with_expectations(scorer_class, expectations, sample_assertion):
    scorer = scorer_class()

    async def mock_ascore(sample):
        assert sample_assertion(sample)
        return 0.9

    with patch.object(scorer._metric, "ascore", mock_ascore):
        result = scorer(expectations=expectations)

    assert isinstance(result, Feedback)
    assert result.name == scorer_class.metric_name
