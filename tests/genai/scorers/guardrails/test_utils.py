import sys
from unittest.mock import patch

import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.guardrails.utils import (
    check_guardrails_installed,
    map_scorer_inputs_to_text,
)


def _create_test_trace(inputs=None, outputs=None):
    """Create a test trace using mlflow.start_span()."""
    with mlflow.start_span() as span:
        if inputs is not None:
            span.set_inputs(inputs)
        if outputs is not None:
            span.set_outputs(outputs)
    return mlflow.get_trace(span.trace_id)


def test_check_guardrails_installed_failure():
    original_guardrails = sys.modules.get("guardrails")

    try:
        if "guardrails" in sys.modules:
            del sys.modules["guardrails"]

        with patch.dict(sys.modules, {"guardrails": None}):
            with pytest.raises(MlflowException, match="guardrails-ai"):
                check_guardrails_installed()
    finally:
        if original_guardrails is not None:
            sys.modules["guardrails"] = original_guardrails


@pytest.mark.parametrize(
    ("inputs", "outputs", "expected"),
    [
        ("input text", "output text", "output text"),
        ("input only", None, "input only"),
        ({"query": "test"}, None, "test"),
    ],
)
def test_map_scorer_inputs_to_text(inputs, outputs, expected):
    result = map_scorer_inputs_to_text(inputs=inputs, outputs=outputs)

    assert expected in result


def test_map_scorer_inputs_to_text_with_trace():
    trace = _create_test_trace(
        inputs={"question": "What is MLflow?"},
        outputs={"answer": "MLflow is an ML platform."},
    )

    result = map_scorer_inputs_to_text(trace=trace)

    assert "MLflow is an ML platform" in result


def test_map_scorer_inputs_to_text_requires_input_or_output():
    with pytest.raises(MlflowException, match="require either 'outputs' or 'inputs'"):
        map_scorer_inputs_to_text(inputs=None, outputs=None)
