import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider

import mlflow
from mlflow.entities.span import SpanStatusCode, encode_span_id
from mlflow.entities.trace_location import MlflowExperimentLocation
from mlflow.entities.trace_state import TraceState
from mlflow.environment_variables import MLFLOW_USE_DEFAULT_TRACER_PROVIDER
from mlflow.tracing.processor.mlflow_v3 import MlflowV3SpanProcessor
from mlflow.tracing.provider import provider, set_destination
from mlflow.utils.os import is_windows

from tests.tracing.helper import get_traces


@pytest.fixture(autouse=True)
def reset_tracing():
    yield
    # Explicitly reset all tracing state to ensure test isolation when tests
    # switch between MLFLOW_USE_DEFAULT_TRACER_PROVIDER modes. This is needed
    # because mlflow.tracing.reset() only resets the state for the current mode,
    # but this fixture runs when env var is at default.
    otel_trace._TRACER_PROVIDER = None
    otel_trace._TRACER_PROVIDER_SET_ONCE._done = False
    # Also reset MLflow's internal once flags for both modes
    provider._global_provider_init_once._done = False
    provider._isolated_tracer_provider_once._done = False


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
        assert mlflow.get_current_active_span().span_id == encode_span_id(root_span.context.span_id)

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
    assert trace.info.request_time == root_span.start_time // 1_000_000
    assert trace.info.execution_duration == (root_span.end_time - root_span.start_time) // 1_000_000
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
    assert (
        trace.info.execution_duration
        == (mlflow_span.end_time_ns - mlflow_span.start_time_ns) // 1_000_000
    )
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
    assert (
        trace.info.execution_duration
        == (mlflow_span.end_time_ns - mlflow_span.start_time_ns) // 1_000_000
    )
    assert trace.info.request_preview == '{"text": "hello"}'
    assert trace.info.response_preview == '{"text": "world"}'

    spans = trace.data.spans
    assert len(spans) == 1
    assert spans[0].name == "mlflow_span"
    assert spans[0].inputs == {"text": "hello"}
    assert spans[0].outputs == {"text": "world"}
    assert spans[0].status.status_code == SpanStatusCode.OK


def test_mlflow_adds_processors_to_existing_tracer_provider(monkeypatch):
    monkeypatch.setenv(MLFLOW_USE_DEFAULT_TRACER_PROVIDER.name, "false")
    experiment_id = mlflow.set_experiment("test_experiment").experiment_id

    external_provider = TracerProvider()
    otel_trace.set_tracer_provider(external_provider)

    # Trigger MLflow initialization - this adds MLflow's processors to the external provider
    set_destination(MlflowExperimentLocation(experiment_id))

    # Verify the external provider was NOT replaced
    assert otel_trace.get_tracer_provider() is external_provider

    # Verify MLflow's processors were added to the external provider
    processors = external_provider._active_span_processor._span_processors
    assert any(isinstance(p, MlflowV3SpanProcessor) for p in processors)

    otel_tracer = otel_trace.get_tracer("external_lib")
    with otel_tracer.start_as_current_span("http_request_parent") as external_span:
        external_span.set_attribute("http.method", "GET")

        with mlflow.start_span("model_prediction") as mlflow_span:
            mlflow_span.set_inputs({"query": "test"})
            mlflow_span.set_outputs({"result": "success"})

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.trace_id.startswith("tr-")
    assert trace.info.status == TraceState.OK

    spans = trace.data.spans
    assert len(spans) == 2
    assert spans[0].name == "http_request_parent"
    assert spans[0].parent_id is None
    assert spans[1].name == "model_prediction"
    assert spans[1].parent_id == spans[0].span_id
    assert spans[1].inputs == {"query": "test"}
    assert spans[1].outputs == {"result": "success"}
    assert spans[1].status.status_code == SpanStatusCode.OK


def test_mlflow_does_not_add_duplicate_processors_global_mode(monkeypatch):
    monkeypatch.setenv(MLFLOW_USE_DEFAULT_TRACER_PROVIDER.name, "false")
    experiment_id = mlflow.set_experiment("test_experiment").experiment_id

    external_provider = TracerProvider()
    otel_trace.set_tracer_provider(external_provider)

    # First call to initialize tracer provider - adds MLflow's processors
    set_destination(MlflowExperimentLocation(experiment_id))

    processors = external_provider._active_span_processor._span_processors
    assert len(processors) == 1
    assert isinstance(processors[0], MlflowV3SpanProcessor)

    # Second call to initialize tracer provider - should NOT add duplicate processors
    set_destination(MlflowExperimentLocation(experiment_id))

    latest_processors = external_provider._active_span_processor._span_processors
    assert latest_processors == processors


def test_mlflow_does_not_add_duplicate_processors_isolated_mode(monkeypatch):
    monkeypatch.setenv(MLFLOW_USE_DEFAULT_TRACER_PROVIDER.name, "true")
    experiment_id = mlflow.set_experiment("test_experiment").experiment_id

    with mlflow.start_span("mlflow_span"):
        pass

    current_provider = provider.get()
    processors = current_provider._active_span_processor._span_processors
    assert len(processors) == 1
    assert isinstance(processors[0], MlflowV3SpanProcessor)

    # Second call to initialize tracer provider - should NOT add duplicate processors
    set_destination(MlflowExperimentLocation(experiment_id))

    latest_processors = current_provider._active_span_processor._span_processors
    assert latest_processors == processors


@pytest.mark.parametrize(
    "use_default_tracer_provider",
    [True, False],
)
def test_initialize_tracer_provider_without_otel_provider_set(
    monkeypatch, use_default_tracer_provider
):
    monkeypatch.setenv(MLFLOW_USE_DEFAULT_TRACER_PROVIDER.name, str(use_default_tracer_provider))
    experiment_id = mlflow.set_experiment("test_experiment").experiment_id
    set_destination(MlflowExperimentLocation(experiment_id))
    # no external provider set, we should always use mlflow own tracer provider
    processors = provider.get()._active_span_processor._span_processors
    assert len(processors) == 1
    assert isinstance(processors[0], MlflowV3SpanProcessor)
