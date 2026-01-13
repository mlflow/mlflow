from unittest.mock import Mock, patch

import pytest

from mlflow.genai.scorers.trulens.utils import (
    format_trulens_rationale,
    map_scorer_inputs_to_trulens_args,
)


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
        (
            "UnknownMetric",
            "input text",
            "output text",
            {"context": "context text"},
            {"input": "input text", "output": "output text", "context": "context text"},
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
    mock_trace = Mock()

    with (
        patch(
            "mlflow.genai.scorers.trulens.utils.resolve_inputs_from_trace",
            return_value="resolved input",
        ),
        patch(
            "mlflow.genai.scorers.trulens.utils.resolve_outputs_from_trace",
            return_value="resolved output",
        ),
        patch(
            "mlflow.genai.scorers.trulens.utils.resolve_expectations_from_trace",
            return_value={"context": "resolved context"},
        ),
    ):
        result = map_scorer_inputs_to_trulens_args(
            metric_name="Groundedness",
            trace=mock_trace,
        )

        assert result["source"] == "resolved context"
        assert result["statement"] == "resolved output"


def test_map_scorer_inputs_trace_context_fallback():
    mock_trace = Mock()

    with (
        patch(
            "mlflow.genai.scorers.trulens.utils.resolve_inputs_from_trace",
            return_value="input",
        ),
        patch(
            "mlflow.genai.scorers.trulens.utils.resolve_outputs_from_trace",
            return_value="output",
        ),
        patch(
            "mlflow.genai.scorers.trulens.utils.resolve_expectations_from_trace",
            return_value=None,
        ),
        patch(
            "mlflow.genai.scorers.trulens.utils.extract_retrieval_context_from_trace",
            return_value={"span1": [{"content": "trace context"}]},
        ),
    ):
        result = map_scorer_inputs_to_trulens_args(
            metric_name="Groundedness",
            trace=mock_trace,
        )

        assert result["source"] == "trace context"


@pytest.mark.parametrize(
    ("reasons", "expected"),
    [
        (None, None),
        ({}, None),
        ({"reason": "Good answer"}, "reason: Good answer"),
        ({"reasons": ["A", "B", "C"]}, "reasons: A; B; C"),
    ],
)
def test_format_trulens_rationale(reasons, expected):
    assert format_trulens_rationale(reasons) == expected


def test_format_trulens_rationale_multiple_reasons():
    result = format_trulens_rationale(
        {
            "reason1": "First reason",
            "reason2": "Second reason",
        }
    )
    assert "reason1: First reason" in result
    assert "reason2: Second reason" in result
    assert " | " in result


def test_format_trulens_rationale_dict_reason():
    result = format_trulens_rationale({"details": {"key": "value"}})
    assert "details:" in result
    assert "key" in result
