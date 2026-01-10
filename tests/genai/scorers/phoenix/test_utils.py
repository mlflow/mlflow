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
from mlflow.genai.scorers.phoenix.utils import (
    map_scorer_inputs_to_phoenix_record,
)
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION, TRACE_SCHEMA_VERSION_KEY
from mlflow.tracing.utils import build_otel_context


def _create_test_trace(
    inputs: dict[str, str] | None = None,
    outputs: dict[str, str] | None = None,
) -> Trace:
    """Create a realistic trace for testing."""
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


def test_check_phoenix_installed_raises_without_phoenix():
    with patch.dict("sys.modules", {"phoenix": None, "phoenix.evals": None}):
        for mod in list(sys.modules.keys()):
            if "mlflow.genai.scorers.phoenix" in mod:
                del sys.modules[mod]

        from mlflow.genai.scorers.phoenix.utils import check_phoenix_installed as check_fn

        with pytest.raises(MlflowException, match="arize-phoenix-evals"):
            check_fn()


def test_map_scorer_inputs_basic():
    record = map_scorer_inputs_to_phoenix_record(
        inputs="What is MLflow?",
        outputs="MLflow is a platform",
    )

    assert record["input"] == "What is MLflow?"
    assert record["output"] == "MLflow is a platform"
    assert "reference" not in record


def test_map_scorer_inputs_with_expected_response():
    record = map_scorer_inputs_to_phoenix_record(
        inputs="What is MLflow?",
        outputs="MLflow is a platform",
        expectations={"expected_response": "MLflow is an ML platform."},
    )

    assert record["input"] == "What is MLflow?"
    assert record["output"] == "MLflow is a platform"
    assert record["reference"] == "MLflow is an ML platform."


def test_map_scorer_inputs_with_context():
    record = map_scorer_inputs_to_phoenix_record(
        inputs="What is MLflow?",
        outputs="MLflow is a platform",
        expectations={"context": "MLflow context here."},
    )

    assert record["reference"] == "MLflow context here."


def test_map_scorer_inputs_expected_response_priority():
    record = map_scorer_inputs_to_phoenix_record(
        inputs="test",
        outputs="test output",
        expectations={
            "expected_response": "priority value",
            "context": "should be ignored",
            "reference": "also ignored",
        },
    )

    assert record["reference"] == "priority value"


def test_map_scorer_inputs_with_trace():
    trace = _create_test_trace(
        inputs={"question": "What is MLflow?"},
        outputs={"answer": "MLflow is a platform for ML lifecycle."},
    )

    record = map_scorer_inputs_to_phoenix_record(
        expectations={"expected_response": "MLflow is an ML platform."},
        trace=trace,
    )

    assert record["input"] == "{'question': 'What is MLflow?'}"
    assert record["output"] == '{"answer": "MLflow is a platform for ML lifecycle."}'
    assert record["reference"] == "MLflow is an ML platform."


def test_map_scorer_inputs_none_values():
    record = map_scorer_inputs_to_phoenix_record()

    assert "input" not in record
    assert "output" not in record
    assert "reference" not in record
