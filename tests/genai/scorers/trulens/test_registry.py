import pytest

from mlflow.genai.scorers.trulens.registry import get_argument_mapping, get_feedback_method_name


@pytest.mark.parametrize(
    ("metric_name", "expected_method"),
    [
        ("Groundedness", "groundedness_measure_with_cot_reasons"),
        ("ContextRelevance", "context_relevance_with_cot_reasons"),
        ("AnswerRelevance", "relevance_with_cot_reasons"),
        ("Coherence", "coherence_with_cot_reasons"),
    ],
)
def test_get_feedback_method_name(metric_name, expected_method):
    assert get_feedback_method_name(metric_name) == expected_method


def test_get_feedback_method_name_unknown_metric():
    # Unknown metrics should return inferred method name
    assert get_feedback_method_name("NewMetric") == "new_metric_with_cot_reasons"
    assert get_feedback_method_name("SomeOtherMetric") == "some_other_metric_with_cot_reasons"


@pytest.mark.parametrize(
    ("metric_name", "expected_mapping"),
    [
        ("Groundedness", {"context": "source", "output": "statement"}),
        ("ContextRelevance", {"input": "question", "context": "context"}),
        ("AnswerRelevance", {"input": "prompt", "output": "response"}),
        ("Coherence", {"output": "text"}),
    ],
)
def test_get_argument_mapping(metric_name, expected_mapping):
    assert get_argument_mapping(metric_name) == expected_mapping


def test_get_argument_mapping_unknown_metric():
    # Unknown metrics should return empty mapping
    assert get_argument_mapping("UnknownMetric") == {}
