import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.ragas.registry import (
    get_metric_class,
    is_agentic_or_multiturn_metric,
    requires_embeddings,
    requires_llm_at_score_time,
    requires_llm_in_constructor,
)


def test_get_metric_class_returns_valid_class():
    metric_class = get_metric_class("Faithfulness")
    assert metric_class.__name__ == "Faithfulness"


def test_get_metric_class_raises_error_for_invalid_name():
    with pytest.raises(MlflowException, match="Unknown metric: 'InvalidMetric'"):
        get_metric_class("InvalidMetric")


@pytest.mark.parametrize(
    ("metric_name", "expected"),
    [
        ("TopicAdherence", True),
        ("ToolCallAccuracy", True),
        ("ToolCallF1", True),
        ("AgentGoalAccuracyWithReference", True),
        ("AgentGoalAccuracyWithoutReference", True),
        ("Faithfulness", False),
        ("ExactMatch", False),
        ("ContextPrecision", False),
    ],
)
def test_is_agentic_or_multiturn_metric(metric_name, expected):
    assert is_agentic_or_multiturn_metric(metric_name) is expected


@pytest.mark.parametrize(
    ("metric_name", "expected"),
    [
        ("AnswerRelevancy", True),
        ("SemanticSimilarity", True),
        ("Faithfulness", False),
        ("ExactMatch", False),
    ],
)
def test_requires_embeddings(metric_name, expected):
    assert requires_embeddings(metric_name) is expected


@pytest.mark.parametrize(
    ("metric_name", "expected"),
    [
        ("Faithfulness", True),
        ("ContextPrecision", True),
        ("AgentGoalAccuracyWithReference", True),
        ("AnswerRelevancy", True),
        ("SemanticSimilarity", False),
        ("DiscreteMetric", False),
    ],
)
def test_requires_llm_in_constructor(metric_name, expected):
    assert requires_llm_in_constructor(metric_name) is expected


@pytest.mark.parametrize(
    ("metric_name", "expected"),
    [
        ("DiscreteMetric", True),
        ("Faithfulness", False),
        ("ExactMatch", False),
    ],
)
def test_requires_llm_at_score_time(metric_name, expected):
    assert requires_llm_at_score_time(metric_name) is expected
