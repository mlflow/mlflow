import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import mlflow
from mlflow.entities.span import SpanStatusCode, encode_span_id
from mlflow.entities.trace_location import MlflowExperimentLocation
from mlflow.entities.trace_state import TraceState
from mlflow.environment_variables import (
    MLFLOW_TRACE_PROPAGATE_TO_OTEL_CONTEXT,
    MLFLOW_USE_DEFAULT_TRACER_PROVIDER,
)
from mlflow.tracing.processor.mlflow_v3 import MlflowV3SpanProcessor
from mlflow.tracing.provider import get_bridged_tracer_provider, provider, set_destination
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
    assert (
        abs(
            trace.info.execution_duration - (root_span.end_time - root_span.start_time) // 1_000_000
        )
        <= 1
    )
    assert trace.info.request_preview is None
    assert trace.info.response_preview is None

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
        abs(
            trace.info.execution_duration
            - (mlflow_span.end_time_ns - mlflow_span.start_time_ns) // 1_000_000
        )
        <= 1
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

    # Set up otel tracer
    tracer_provider = TracerProvider(resource=None)
    exporter = InMemorySpanExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(tracer_provider)
    otel_tracer = otel_trace.get_tracer(__name__)

    with otel_tracer.start_as_current_span("otel_root") as root_span:
        root_span.set_attribute("key1", "value1")

        with mlflow.start_span("mlflow_root") as mlflow_span:
            mlflow_span.set_inputs({"text": "hello"})
            mlflow_span.set_outputs({"text": "world"})

            with otel_tracer.start_as_current_span("otel_child") as child_span:
                child_span.set_attribute("key2", "value2")

                with mlflow.start_span("mlflow_child") as mlflow_child_span:
                    mlflow_child_span.set_attribute("key3", "value3")

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace is not None
    assert trace.info.experiment_id == experiment_id
    assert trace.info.trace_id.startswith("tr-")  # trace ID should be in MLflow format
    assert trace.info.status == TraceState.OK
    assert trace.info.request_time == mlflow_span.start_time_ns // 1_000_000
    assert (
        abs(
            trace.info.execution_duration
            - (mlflow_span.end_time_ns - mlflow_span.start_time_ns) // 1_000_000
        )
        <= 1
    )
    assert trace.info.request_preview == '{"text": "hello"}'
    assert trace.info.response_preview == '{"text": "world"}'

    spans = trace.data.spans
    assert len(spans) == 2
    assert spans[0].name == "mlflow_root"
    assert spans[0].inputs == {"text": "hello"}
    assert spans[0].outputs == {"text": "world"}
    assert spans[0].status.status_code == SpanStatusCode.OK
    assert spans[1].name == "mlflow_child"
    assert spans[1].attributes["key3"] == "value3"
    assert spans[1].status.status_code == SpanStatusCode.OK
    assert spans[1].parent_id == spans[0].span_id

    # Otel span should be exported independently of MLflow span
    otel_spans = exporter.get_finished_spans()
    assert len(otel_spans) == 2
    assert otel_spans[0].name == "otel_child"
    assert otel_spans[0].attributes["key2"] == "value2"
    assert otel_spans[0].parent.span_id == otel_spans[1].context.span_id
    assert otel_spans[1].name == "otel_root"
    assert otel_spans[1].attributes["key1"] == "value1"


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


def test_mlflow_span_does_not_leak_to_otel_context_by_default(monkeypatch):
    """Regression test for https://github.com/mlflow/mlflow/issues/24105

    In isolated tracer provider mode (the default), the MLflow span must NOT leak into the
    process-global OTel context. This preserves the isolation guarantee so unrelated OTel
    instrumentation (e.g. FastAPI, requests) does not accidentally nest under MLflow spans.
    """
    monkeypatch.setenv(MLFLOW_USE_DEFAULT_TRACER_PROVIDER.name, "true")
    monkeypatch.delenv(MLFLOW_TRACE_PROPAGATE_TO_OTEL_CONTEXT.name, raising=False)
    mlflow.set_experiment("test_experiment")

    with mlflow.start_span(name="parent"):
        current = otel_trace.get_current_span()
        # Isolated mode keeps the MLflow span out of the global OTel context.
        assert not current.is_recording()
        assert type(current).__name__ == "NonRecordingSpan"


def test_mlflow_trace_decorator_sets_otel_parent_context_when_opted_in(monkeypatch):
    """Regression test for https://github.com/mlflow/mlflow/issues/24105

    With MLFLOW_TRACE_PROPAGATE_TO_OTEL_CONTEXT enabled, @mlflow.trace should propagate the
    span to the global OTel context so that pure-OTel libraries (e.g. strands-agents) can see
    it as a parent and create properly nested child spans.
    """
    monkeypatch.setenv(MLFLOW_USE_DEFAULT_TRACER_PROVIDER.name, "true")
    monkeypatch.setenv(MLFLOW_TRACE_PROPAGATE_TO_OTEL_CONTEXT.name, "true")
    mlflow.set_experiment("test_experiment")

    captured = {}

    @mlflow.trace(name="parent")
    def my_function():
        current = otel_trace.get_current_span()
        captured["is_recording"] = current.is_recording()
        captured["span_type"] = type(current).__name__
        return 42

    my_function()

    assert captured["is_recording"], (
        "OTel current span should be recording inside @mlflow.trace"
    )
    assert captured["span_type"] != "NonRecordingSpan"


def test_mlflow_start_span_sets_otel_parent_context_when_opted_in(monkeypatch):
    """Regression test for https://github.com/mlflow/mlflow/issues/24105

    With MLFLOW_TRACE_PROPAGATE_TO_OTEL_CONTEXT enabled, mlflow.start_span() should propagate
    the span to the global OTel context so that pure-OTel libraries can nest under it.
    """
    monkeypatch.setenv(MLFLOW_USE_DEFAULT_TRACER_PROVIDER.name, "true")
    monkeypatch.setenv(MLFLOW_TRACE_PROPAGATE_TO_OTEL_CONTEXT.name, "true")
    mlflow.set_experiment("test_experiment")

    with mlflow.start_span(name="parent"):
        current = otel_trace.get_current_span()
        is_recording = current.is_recording()
        span_type_name = type(current).__name__

    assert is_recording, (
        "OTel current span should be recording inside mlflow.start_span()"
    )
    assert span_type_name != "NonRecordingSpan"


def test_mlflow_span_cleans_up_otel_context_after_exit(monkeypatch):
    """Verify the OTel global context is cleaned up when the MLflow span ends (opt-in)."""
    monkeypatch.setenv(MLFLOW_USE_DEFAULT_TRACER_PROVIDER.name, "true")
    monkeypatch.setenv(MLFLOW_TRACE_PROPAGATE_TO_OTEL_CONTEXT.name, "true")
    mlflow.set_experiment("test_experiment")

    with mlflow.start_span(name="parent"):
        inside = otel_trace.get_current_span()
        assert inside.is_recording()

    outside = otel_trace.get_current_span()
    assert not outside.is_recording(), (
        "OTel context should be cleaned up after the MLflow span exits"
    )


def test_nested_mlflow_spans_maintain_otel_context_when_opted_in(monkeypatch):
    """Verify nested MLflow spans properly manage the OTel context stack (opt-in)."""
    monkeypatch.setenv(MLFLOW_USE_DEFAULT_TRACER_PROVIDER.name, "true")
    monkeypatch.setenv(MLFLOW_TRACE_PROPAGATE_TO_OTEL_CONTEXT.name, "true")
    mlflow.set_experiment("test_experiment")

    with mlflow.start_span(name="outer") as outer_span:
        outer_otel = otel_trace.get_current_span()
        assert outer_otel.is_recording()

        with mlflow.start_span(name="inner") as inner_span:
            inner_otel = otel_trace.get_current_span()
            assert inner_otel.is_recording()
            assert inner_span.parent_id == outer_span.span_id

        after_inner = otel_trace.get_current_span()
        assert after_inner.is_recording(), (
            "Outer span's OTel context should be restored after inner span exits"
        )


def test_get_bridged_tracer_provider_returns_mlflow_provider_isolated(monkeypatch):
    """get_bridged_tracer_provider should hand out MLflow's isolated provider by default.

    This lets OTel instrumentors that accept `tracer_provider=` route spans through MLflow's
    pipeline without depending on the process-global provider (issue #24105).
    """
    monkeypatch.setenv(MLFLOW_USE_DEFAULT_TRACER_PROVIDER.name, "true")
    experiment_id = mlflow.set_experiment("test_experiment").experiment_id
    set_destination(MlflowExperimentLocation(experiment_id))

    bridged = get_bridged_tracer_provider()
    assert bridged is provider.get()
    processors = bridged._active_span_processor._span_processors
    assert any(isinstance(p, MlflowV3SpanProcessor) for p in processors)


def test_get_bridged_tracer_provider_routes_spans_to_mlflow(monkeypatch):
    """Spans created via the bridged provider's tracer should be captured by MLflow."""
    monkeypatch.setenv(MLFLOW_USE_DEFAULT_TRACER_PROVIDER.name, "true")
    experiment_id = mlflow.set_experiment("test_experiment").experiment_id
    set_destination(MlflowExperimentLocation(experiment_id))

    bridged = get_bridged_tracer_provider()
    otel_tracer = bridged.get_tracer("external_instrumentor")
    with otel_tracer.start_as_current_span("external_span") as span:
        span.set_attribute("key", "value")

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.trace_id.startswith("tr-")
    assert trace.info.experiment_id == experiment_id
    assert any(s.name == "external_span" for s in trace.data.spans)


def test_get_bridged_tracer_provider_returns_global_provider_unified(monkeypatch):
    """In unified mode, the bridged provider is the shared global provider."""
    monkeypatch.setenv(MLFLOW_USE_DEFAULT_TRACER_PROVIDER.name, "false")
    experiment_id = mlflow.set_experiment("test_experiment").experiment_id
    set_destination(MlflowExperimentLocation(experiment_id))

    bridged = get_bridged_tracer_provider()
    assert bridged is otel_trace.get_tracer_provider()
