import asyncio

import pytest

langfuse = pytest.importorskip("langfuse", reason="langfuse is not installed")
from langfuse import observe
from langfuse._client.resource_manager import LangfuseResourceManager
from opentelemetry import trace as otel_trace_api
from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider

import mlflow.otel
from mlflow.entities.span import SpanType

from tests.tracing.helper import get_traces


@pytest.fixture(autouse=True)
def langfuse_otel_env(monkeypatch):
    """Set dummy Langfuse credentials and reset OTEL / Langfuse state between tests.

    Langfuse needs valid credentials to create real OTEL spans; without them the
    client falls back to a ``NoOpTracer``.  A dummy host is fine — the Langfuse
    ``BatchSpanProcessor`` silently drops spans it cannot export.
    """
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test-dummy")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test-dummy")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:9999")

    # Start each test with a fresh global TracerProvider so processors from
    # previous tests don't interfere.
    otel_trace_api.set_tracer_provider(SdkTracerProvider())
    mlflow.otel._active_processor = None

    yield

    # Teardown: disable MLflow processor, then reset Langfuse singleton so it
    # re-initialises on the next test's fresh TracerProvider.
    mlflow.otel.autolog(disable=True)
    mlflow.otel._active_processor = None
    LangfuseResourceManager.reset()


def test_sync_observe_autolog():
    mlflow.otel.autolog()

    @observe()
    def add(x, y):
        return x + y

    result = add(2, 3)
    assert result == 5

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "add"
    assert span.inputs == {"args": [2, 3], "kwargs": {}}
    assert span.outputs == 5


def test_sync_observe_with_custom_name():
    mlflow.otel.autolog()

    @observe(name="custom-add")
    def add(x, y):
        return x + y

    result = add(2, 3)
    assert result == 5

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]
    assert span.name == "custom-add"


def test_async_observe_autolog():
    mlflow.otel.autolog()

    @observe()
    async def async_add(x, y):
        return x + y

    result = asyncio.run(async_add(10, 20))
    assert result == 30

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "async_add"


def test_nested_observe_autolog():
    mlflow.otel.autolog()

    @observe()
    def inner(x):
        return x * 2

    @observe()
    def outer(x):
        return inner(x) + 1

    result = outer(10)
    assert result == 21

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 2
    span_names = sorted(s.name for s in traces[0].data.spans)
    assert span_names == ["inner", "outer"]

    # Verify parent-child relationship
    spans_by_name = {s.name: s for s in traces[0].data.spans}
    outer_span = spans_by_name["outer"]
    inner_span = spans_by_name["inner"]
    assert outer_span.parent_id is None
    assert inner_span.parent_id == outer_span.span_id


def test_disable_autolog():
    mlflow.otel.autolog()

    @observe()
    def add(x, y):
        return x + y

    add(1, 2)
    traces = get_traces()
    assert len(traces) == 1

    mlflow.otel.autolog(disable=True)

    result = add(3, 4)
    assert result == 7

    # No new trace should be created
    traces = get_traces()
    assert len(traces) == 1


def test_exception_propagation():
    mlflow.otel.autolog()

    @observe()
    def fail():
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        fail()

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "ERROR"


def test_observe_without_parentheses():
    mlflow.otel.autolog()

    @observe
    def add(x, y):
        return x + y

    result = add(2, 3)
    assert result == 5

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]
    assert span.name == "add"


def test_autolog_is_additive():
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

    exported_spans: list[object] = []

    class RecordingExporter(SpanExporter):
        def export(self, spans):
            exported_spans.extend(spans)
            return SpanExportResult.SUCCESS

    mlflow.otel.autolog()

    # Register a recording exporter on the same provider to prove that
    # the TracerProvider dispatches spans to every processor, not just MLflow's.
    provider = otel_trace_api.get_tracer_provider()
    provider.add_span_processor(SimpleSpanProcessor(RecordingExporter()))

    @observe()
    def add(x, y):
        return x + y

    result = add(2, 3)
    assert result == 5

    # MLflow received the trace
    traces = get_traces()
    assert len(traces) == 1

    # The recording exporter also received the span, proving dispatch to
    # all processors (including Langfuse's).
    assert any(s.name == "add" for s in exported_spans)


@pytest.mark.parametrize(
    ("langfuse_type", "expected_mlflow_type"),
    [
        ("generation", SpanType.LLM),
        ("tool", SpanType.TOOL),
        ("retriever", SpanType.RETRIEVER),
        ("embedding", SpanType.EMBEDDING),
        ("agent", SpanType.AGENT),
        ("chain", SpanType.CHAIN),
        ("evaluator", SpanType.EVALUATOR),
        ("guardrail", SpanType.GUARDRAIL),
        ("span", SpanType.UNKNOWN),
    ],
)
def test_span_type_mapping(langfuse_type, expected_mlflow_type):
    mlflow.otel.autolog()

    @observe(as_type=langfuse_type)
    def func(x):
        return x

    func("test")

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]
    assert span.span_type == expected_mlflow_type
