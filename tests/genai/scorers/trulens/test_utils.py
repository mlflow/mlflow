import json
import time

import pytest
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

from mlflow.entities.span import Span
from mlflow.entities.trace import Trace, TraceData, TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.genai.scorers.trulens.utils import (
    format_rationale,
    map_scorer_inputs_to_trulens_args,
)
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION, TRACE_SCHEMA_VERSION_KEY
from mlflow.tracing.utils import build_otel_context


def _create_test_trace(
    inputs: dict[str, str] | None = None,
    outputs: dict[str, str] | None = None,
) -> Trace:
    current_time_ns = int(time.time() * 1e9)
    trace_id = "test_trace_001"

    attributes = {"mlflow.traceRequestId": json.dumps(trace_id)}
    if inputs is not None:
        attributes["mlflow.spanInputs"] = json.dumps(inputs)
    if outputs is not None:
        attributes["mlflow.spanOutputs"] = json.dumps(outputs)
    attributes["mlflow.spanType"] = json.dumps("CHAIN")

    otel_span = OTelReadableSpan(
        name="root_span",
        context=build_otel_context(12345, 111),
        parent=None,
        start_time=current_time_ns,
        end_time=current_time_ns + 1000000,
        attributes=attributes,
    )

    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=int(time.time() * 1000),
        state=TraceState.OK,
        execution_duration=1000,
        trace_metadata={TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION)},
        tags={},
        assessments=[],
        request_preview=json.dumps(inputs) if inputs else None,
        response_preview=json.dumps(outputs) if outputs else None,
    )

    trace_data = TraceData(spans=[Span(otel_span)])
    return Trace(info=trace_info, data=trace_data)


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
