import pytest
from typing import Any, cast

from opentelemetry import trace as otel_trace

import mlflow
from mlflow.entities.span import SpanStatusCode, encode_span_id
from mlflow.entities.trace_location import MlflowExperimentLocation
from mlflow.entities.trace_state import TraceState
from mlflow.environment_variables import MLFLOW_USE_DEFAULT_TRACER_PROVIDER
from mlflow.utils.os import is_windows

from tests.tracing.helper import get_traces


@pytest.mark.skipif(is_windows(), reason="Skipping as this is flaky on Windows")
def test_mlflow_and_opentelemetry_unified_tracing_with_otel_root_span(monkeypatch):
    monkeypatch.setenv(MLFLOW_USE_DEFAULT_TRACER_PROVIDER.name, "false")

    # Use set_destination to trigger tracer provider initialization
    experiment_id = mlflow.set_experiment("test_experiment").experiment_id
    mlflow.tracing.set_destination(MlflowExperimentLocation(experiment_id))

    otel_tracer = otel_trace.get_tracer(__name__)
    with otel_tracer.start_as_current_span("parent_span") as root_span:
        root_span.set_attribute("key1", "value1")
        root_span.add_event("event1", attributes={"key2": "value2"})

        # Active span id should be set
        active_span = mlflow.get_current_active_span()
        assert active_span is not None
        assert active_span.span_id == encode_span_id(root_span.get_span_context().span_id)

        with mlflow.start_span("mlflow_span") as mlflow_span:
            mlflow_span.set_inputs({"text": "hello"})
            mlflow_span.set_attributes({"key3": "value3"})

            with otel_tracer.start_as_current_span("child_span") as child_span:
                child_span.set_attribute("key4", "value4")
                child_span.set_status(otel_trace.Status(otel_trace.StatusCode.OK))

            mlflow_span.set_outputs({"text": "world"})

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.trace_id.startswith("tr-")  # trace ID should be in MLflow format
    assert trace.info.trace_id == mlflow.get_last_active_trace_id()
    assert trace.info.experiment_id == experiment_id
    assert trace.info.status == TraceState.OK

    otel_root_span = cast(Any, root_span)
    assert trace.info.request_time == otel_root_span.start_time // 1_000_000
    end_time_ns = cast(int, mlflow_span.end_time_ns)
    assert trace.info.execution_duration == (end_time_ns - mlflow_span.start_time_ns) // 1_000_000

    assert trace.info.request_preview == ""
    assert trace.info.response_preview == ""

    spans = trace.data.spans
    assert len(spans) == 3
    assert spans[0].name == "parent_span"
    assert spans[0].attributes["key1"] == "value1"
    assert len(spans[0].events) == 1
    assert spans[0].events[0].name == "event1"
    assert spans[0].events[0].attributes["key2"] == "value2"
    assert spans[0].parent_id is None
    assert spans[0].status.status_code == SpanStatusCode.UNSET
    assert spans[1].name == "mlflow_span"
    assert spans[1].attributes["key3"] == "value3"
    assert spans[1].events == []
    assert spans[1].parent_id == spans[0].span_id
    assert spans[1].status.status_code == SpanStatusCode.OK
    assert spans[2].name == "child_span"
    assert spans[2].attributes["key4"] == "value4"
    assert spans[2].events == []
    assert spans[2].parent_id == spans[1].span_id
    assert spans[2].status.status_code == SpanStatusCode.OK


@pytest.mark.skipif(is_windows(), reason="Skipping as this is flaky on Windows")
def test_otel_root_span_preview_from_genai_operation_details_event(monkeypatch):
    monkeypatch.setenv(MLFLOW_USE_DEFAULT_TRACER_PROVIDER.name, "false")

    experiment_id = mlflow.set_experiment("test_experiment").experiment_id
    mlflow.tracing.set_destination(MlflowExperimentLocation(experiment_id))

    otel_tracer = otel_trace.get_tracer(__name__)

    input_messages = (
        '[{"role": "user", "parts": [{"type": "text", "content": "hello"}]}]'
    )
    output_messages = (
        '[{"role": "assistant", "parts": [{"type": "text", "content": "world"}]}]'
    )

    with otel_tracer.start_as_current_span("chat") as root_span:
        root_span.set_attribute("gen_ai.operation.name", "chat")
        root_span.add_event(
            "gen_ai.client.inference.operation.details",
            attributes={"gen_ai.input.messages": input_messages},
        )
        root_span.add_event(
            "gen_ai.client.inference.operation.details",
            attributes={"gen_ai.output.messages": output_messages},
        )
        root_span.set_status(otel_trace.Status(otel_trace.StatusCode.OK))

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]

    assert trace.info.request_preview == "hello"
    assert trace.info.response_preview == "world"


@pytest.mark.skipif(is_windows(), reason="Skipping as this is flaky on Windows")
def test_mlflow_and_opentelemetry_unified_tracing_with_mlflow_root_span(monkeypatch):
    monkeypatch.setenv(MLFLOW_USE_DEFAULT_TRACER_PROVIDER.name, "false")

    experiment_id = mlflow.set_experiment("test_experiment").experiment_id

    otel_tracer = otel_trace.get_tracer(__name__)
    with mlflow.start_span("mlflow_span") as mlflow_span:
        mlflow_span.set_inputs({"text": "hello"})

        with otel_tracer.start_as_current_span("otel_span") as otel_span:
            otel_span.set_attributes({"key3": "value3"})
            otel_span.set_status(otel_trace.Status(otel_trace.StatusCode.OK))

            with mlflow.start_span("child_span") as child_span:
                child_span.set_attribute("key4", "value4")

        mlflow_span.set_outputs({"text": "world"})

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.trace_id.startswith("tr-")  # trace ID should be in MLflow format
    assert trace.info.trace_id == mlflow.get_last_active_trace_id()
    assert trace.info.experiment_id == experiment_id
    assert trace.info.status == TraceState.OK
    assert trace.info.request_time == mlflow_span.start_time_ns // 1_000_000
    end_time_ns = cast(int, mlflow_span.end_time_ns)
    assert trace.info.execution_duration == (end_time_ns - mlflow_span.start_time_ns) // 1_000_000
    assert trace.info.request_preview == '{"text": "hello"}'
    assert trace.info.response_preview == '{"text": "world"}'

    spans = trace.data.spans
    assert len(spans) == 3
    assert spans[0].name == "mlflow_span"
    assert spans[0].inputs == {"text": "hello"}
    assert spans[0].outputs == {"text": "world"}
    assert spans[0].status.status_code == SpanStatusCode.OK
    assert spans[1].name == "otel_span"
    assert spans[1].attributes["key3"] == "value3"
    assert spans[1].events == []
    assert spans[1].parent_id == spans[0].span_id
    assert spans[1].status.status_code == SpanStatusCode.OK
    assert spans[2].name == "child_span"
    assert spans[2].attributes["key4"] == "value4"
    assert spans[2].events == []
    assert spans[2].parent_id == spans[1].span_id
    assert spans[2].status.status_code == SpanStatusCode.OK


def test_mlflow_and_opentelemetry_isolated_tracing(monkeypatch):
    monkeypatch.setenv(MLFLOW_USE_DEFAULT_TRACER_PROVIDER.name, "true")

    experiment_id = mlflow.set_experiment("test_experiment").experiment_id

    otel_tracer = otel_trace.get_tracer(__name__)

    with otel_tracer.start_as_current_span("parent_span") as root_span:
        root_span.set_attribute("key1", "value1")

    with mlflow.start_span("mlflow_span") as mlflow_span:
        mlflow_span.set_inputs({"text": "hello"})
        mlflow_span.set_outputs({"text": "world"})

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace is not None
    assert trace.info.experiment_id == experiment_id
    assert trace.info.trace_id.startswith("tr-")  # trace ID should be in MLflow format
    assert trace.info.status == TraceState.OK
    assert trace.info.request_time == mlflow_span.start_time_ns // 1_000_000
    end_time_ns = cast(int, mlflow_span.end_time_ns)
    assert trace.info.execution_duration == (end_time_ns - mlflow_span.start_time_ns) // 1_000_000
    assert trace.info.request_preview == '{"text": "hello"}'
    assert trace.info.response_preview == '{"text": "world"}'

    spans = trace.data.spans
    assert len(spans) == 1
    assert spans[0].name == "mlflow_span"
    assert spans[0].inputs == {"text": "hello"}
    assert spans[0].outputs == {"text": "world"}
    assert spans[0].status.status_code == SpanStatusCode.OK
