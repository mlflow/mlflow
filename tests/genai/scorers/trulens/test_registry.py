import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.trulens.registry import (
    build_trulens_args,
    get_feedback_method_name,
    get_metric_arg_mapping,
)


@pytest.mark.parametrize(
    ("metric_name", "expected_method", "expected_mapping"),
    [
        (
            "Groundedness",
            "groundedness_measure_with_cot_reasons",
            {"context": "source", "output": "statement"},
        ),
        (
            "ContextRelevance",
            "context_relevance_with_cot_reasons",
            {"input": "question", "context": "context"},
        ),
        (
            "AnswerRelevance",
            "relevance_with_cot_reasons",
            {"input": "prompt", "output": "response"},
        ),
        (
            "Coherence",
            "coherence_with_cot_reasons",
            {"output": "text"},
        ),
    ],
)
def test_metric_config(metric_name, expected_method, expected_mapping):
    assert get_feedback_method_name(metric_name) == expected_method
    assert get_metric_arg_mapping(metric_name) == expected_mapping


def test_get_feedback_method_name_invalid_metric():
    with pytest.raises(MlflowException, match="Unknown TruLens metric"):
        get_feedback_method_name("InvalidMetric")


def test_get_metric_arg_mapping_invalid_metric():
    with pytest.raises(MlflowException, match="Unknown TruLens metric"):
        get_metric_arg_mapping("InvalidMetric")


@pytest.mark.parametrize(
    ("metric_name", "input_str", "output_str", "context_str", "expected"),
    [
        (
            "Groundedness",
            "input",
            "output",
            "context",
            {"source": "context", "statement": "output"},
        ),
        (
            "ContextRelevance",
            "question",
            "answer",
            "context",
            {"question": "question", "context": "context"},
        ),
        (
            "AnswerRelevance",
            "prompt",
            "response",
            "context",
            {"prompt": "prompt", "response": "response"},
        ),
        (
            "Coherence",
            "input",
            "text",
            "context",
            {"text": "text"},
        ),
    ],
)
def test_build_trulens_args(metric_name, input_str, output_str, context_str, expected):
    result = build_trulens_args(
        metric_name=metric_name,
        input_str=input_str,
        output_str=output_str,
        context_str=context_str,
    )
    assert result == expected
