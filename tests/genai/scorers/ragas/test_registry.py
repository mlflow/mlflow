from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.ragas.registry import (
    get_metric_class,
    is_agentic_or_multiturn_metric,
    requires_args_from_placeholders,
    requires_embeddings,
    requires_llm_at_score_time,
    requires_llm_in_constructor,
)


def test_get_metric_class_returns_valid_class():
    metric_class = get_metric_class("Faithfulness")
    assert metric_class.__name__ == "Faithfulness"


def test_get_metric_class_raises_error_for_invalid_name():
    with pytest.raises(MlflowException, match="Unknown RAGAS metric: 'InvalidMetric'"):
        get_metric_class("InvalidMetric")


def test_get_metric_class_dynamic_import():
    mock_metric_class = mock.MagicMock()
    mock_metric_class.__name__ = "NewMetric"

    mock_module = mock.MagicMock()
    mock_module.NewMetric = mock_metric_class

    with mock.patch.dict("sys.modules", {"ragas.metrics.collections": mock_module}):
        result = get_metric_class("NewMetric")
        assert result is mock_metric_class


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


def test_is_agentic_or_multiturn_metric_unknown():
    assert not is_agentic_or_multiturn_metric("UnknownMetric")


def test_requires_embeddings_unknown():
    assert not requires_embeddings("UnknownMetric")


def test_requires_llm_in_constructor_unknown():
    assert requires_llm_in_constructor("UnknownMetric")


def test_requires_llm_at_score_time_unknown():
    assert not requires_llm_at_score_time("UnknownMetric")


def test_requires_args_from_placeholders_unknown():
    assert not requires_args_from_placeholders("UnknownMetric")
