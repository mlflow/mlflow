import pytest

import mlflow
from mlflow.genai.scorers.trulens.utils import (
    format_rationale,
    map_scorer_inputs_to_trulens_args,
)


def _create_test_trace(
    inputs: dict[str, str] | None = None,
    outputs: dict[str, str] | None = None,
):
    with mlflow.start_span() as span:
        if inputs is not None:
            span.set_inputs(inputs)
        if outputs is not None:
            span.set_outputs(outputs)

    return mlflow.get_trace(span.trace_id)


@pytest.mark.parametrize(
    ("metric_name", "inputs", "outputs", "expectations", "expected"),
    [
        (
            "Groundedness",
            None,
            "The answer is 42.",
            {"context": "The answer to everything is 42."},
            {"source": "The answer to everything is 42.", "statement": "The answer is 42."},
        ),
        (
            "ContextRelevance",
            "What is the answer?",
            None,
            {"context": "The answer is 42."},
            {"question": "What is the answer?", "context": "The answer is 42."},
        ),
        (
            "AnswerRelevance",
            "What is MLflow?",
            "MLflow is a platform for ML lifecycle.",
            None,
            {"prompt": "What is MLflow?", "response": "MLflow is a platform for ML lifecycle."},
        ),
        (
            "Coherence",
            None,
            "This is a well-structured response.",
            None,
            {"text": "This is a well-structured response."},
        ),
    ],
)
def test_map_scorer_inputs_metric_mappings(metric_name, inputs, outputs, expectations, expected):
    result = map_scorer_inputs_to_trulens_args(
        metric_name=metric_name,
        inputs=inputs,
        outputs=outputs,
        expectations=expectations,
    )
    assert result == expected


def test_map_scorer_inputs_context_from_list():
    result = map_scorer_inputs_to_trulens_args(
        metric_name="Groundedness",
        outputs="Combined answer.",
        expectations={"context": ["First context.", "Second context."]},
    )
    assert result["source"] == "First context.\nSecond context."


def test_map_scorer_inputs_context_priority_order():
    result = map_scorer_inputs_to_trulens_args(
        metric_name="Groundedness",
        outputs="test",
        expectations={
            "context": "primary context",
            "reference": "should be ignored",
        },
    )
    assert result["source"] == "primary context"


def test_map_scorer_inputs_reference_fallback():
    result = map_scorer_inputs_to_trulens_args(
        metric_name="Groundedness",
        outputs="test",
        expectations={"reference": "reference context"},
    )
    assert result["source"] == "reference context"


def test_map_scorer_inputs_with_trace():
    trace = _create_test_trace(
        inputs={"question": "What is MLflow?"},
        outputs={"answer": "MLflow is a platform for ML lifecycle."},
    )

    result = map_scorer_inputs_to_trulens_args(
        metric_name="AnswerRelevance",
        expectations={"context": "test context"},
        trace=trace,
    )

    assert result["prompt"] == "{'question': 'What is MLflow?'}"
    assert result["response"] == '{"answer": "MLflow is a platform for ML lifecycle."}'


@pytest.mark.parametrize(
    ("reasons", "expected"),
    [
        (None, None),
        ({}, None),
        ({"reason": "Good answer"}, "reason: Good answer"),
        ({"reasons": ["A", "B", "C"]}, "reasons: A; B; C"),
        (
            {"reason1": "First reason", "reason2": "Second reason"},
            "reason1: First reason | reason2: Second reason",
        ),
        ({"details": {"key": "value"}}, "details: {'key': 'value'}"),
    ],
)
def test_format_rationale(reasons, expected):
    assert format_rationale(reasons) == expected
