import json
import sys
import time
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

from mlflow.entities.span import Span
from mlflow.entities.trace import Trace, TraceData, TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.guardrails.utils import (
    check_guardrails_installed,
    map_scorer_inputs_to_text,
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


def test_check_guardrails_installed_success():
    with patch.dict("sys.modules", {"guardrails": object()}):
        check_guardrails_installed()


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
