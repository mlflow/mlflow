from unittest import mock

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import mlflow
from mlflow.entities.trace_info import TraceInfo, TraceLocation, TraceState
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION_KEY, SpanAttributeKey
from mlflow.tracing.processor.otel import OtelSpanProcessor
from mlflow.tracing.provider import provider
from mlflow.tracing.trace_manager import ManagerTrace

from tests.tracing.helper import create_mock_otel_span


def _make_processor():
    exporter = mock.MagicMock()
    exporter.export.return_value = SpanExportResult.SUCCESS
    processor = OtelSpanProcessor(span_exporter=exporter, export_metrics=False)
    return processor, exporter


def _make_manager_trace(tags: dict[str, str]) -> ManagerTrace:
    from mlflow.entities import Trace, TraceData

    info = TraceInfo(
        trace_id="tr-abc123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=0,
        execution_duration=None,
        state=TraceState.OK,
        trace_metadata={TRACE_SCHEMA_VERSION_KEY: "3"},
        tags=tags,
    )
    return ManagerTrace(trace=Trace(info=info, data=TraceData()), prompts=[], is_remote_trace=False)


def _run_on_end(processor, span, manager_trace):
    with (
        mock.patch.object(
            processor._trace_manager, "pop_trace", return_value=manager_trace
        ) as mock_pop_trace,
        mock.patch.object(BatchSpanProcessor, "on_end") as mock_super_on_end,
    ):
        processor.on_end(span)
    mock_pop_trace.assert_called_once_with(span.context.trace_id)
    mock_super_on_end.assert_called_once()
    return mock_super_on_end.call_args.args[0]


def test_on_end_emits_per_tag_attributes():
    processor, _ = _make_processor()
    otel_trace_id = 0xDEADBEEF

    root_span = create_mock_otel_span(trace_id=otel_trace_id, span_id=0x1234)
    assert root_span.parent is None

    manager_trace = _make_manager_trace(tags={"env": "production", "version": "1.0"})

    exported = _run_on_end(processor, root_span, manager_trace)

    assert exported.attributes.get(SpanAttributeKey.TRACE_TAG_PREFIX + "env") == "production"
    assert exported.attributes.get(SpanAttributeKey.TRACE_TAG_PREFIX + "version") == "1.0"


def test_on_end_exports_copy_without_mutating_shared_attributes():
    processor, _ = _make_processor()
    tracer = TracerProvider().get_tracer(__name__)
    root_span = tracer.start_span("root", attributes={"existing": "value"})
    root_span.end()
    readable_span = root_span._readable_span()

    manager_trace = _make_manager_trace(tags={"env": "production"})

    exported = _run_on_end(processor, readable_span, manager_trace)

    assert exported is not readable_span
    assert exported.name == "root"
    assert exported.context == root_span.context
    assert exported.end_time == root_span.end_time
    assert dict(exported.attributes) == {
        "existing": "value",
        SpanAttributeKey.TRACE_TAG_PREFIX + "env": "production",
    }
    assert dict(readable_span.attributes) == {"existing": "value"}
    assert dict(root_span.attributes) == {"existing": "value"}


def test_on_end_exports_span_when_tag_emission_fails():
    processor, _ = _make_processor()
    root_span = create_mock_otel_span(trace_id=0xDEADBEEF, span_id=0x1234)
    manager_trace = _make_manager_trace(tags={"env": "production"})

    with mock.patch.object(
        processor, "_with_trace_tag_attributes", side_effect=RuntimeError("boom")
    ):
        exported = _run_on_end(processor, root_span, manager_trace)

    assert exported is root_span


def test_otlp_export_keeps_root_span_of_tagged_trace():
    exporter = InMemorySpanExporter()
    with (
        mock.patch("mlflow.tracing.provider.should_use_otlp_exporter", return_value=True),
        mock.patch("mlflow.tracing.provider.get_otlp_exporter", return_value=exporter),
    ):
        mlflow.tracing.reset()

        @mlflow.trace(name="tagged")
        def tagged():
            mlflow.update_current_trace(tags={"format": "vscode"})

        tagged()
        provider.get().force_flush(5000)

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert list(spans) == ["tagged"]
    assert spans["tagged"].attributes[SpanAttributeKey.TRACE_TAG_PREFIX + "format"] == "vscode"


def test_on_end_skips_tag_emission_when_no_tags():
    processor, _ = _make_processor()
    otel_trace_id = 0xDEADBEEF

    root_span = create_mock_otel_span(trace_id=otel_trace_id, span_id=0x1234)
    manager_trace = _make_manager_trace(tags={})

    exported = _run_on_end(processor, root_span, manager_trace)

    assert exported is root_span
    assert not any(k.startswith(SpanAttributeKey.TRACE_TAG_PREFIX) for k in exported.attributes)


def test_on_end_skips_tag_emission_for_child_span():
    processor, _ = _make_processor()
    otel_trace_id = 0xDEADBEEF

    child_span = create_mock_otel_span(trace_id=otel_trace_id, span_id=0x5678, parent_id=0x1234)
    assert child_span.parent is not None

    with mock.patch.object(processor._trace_manager, "pop_trace") as mock_pop:
        processor.on_end(child_span)
        mock_pop.assert_not_called()

    assert not any(k.startswith(SpanAttributeKey.TRACE_TAG_PREFIX) for k in child_span._attributes)


def test_on_end_handles_missing_trace_gracefully():
    processor, _ = _make_processor()
    otel_trace_id = 0xDEADBEEF

    root_span = create_mock_otel_span(trace_id=otel_trace_id, span_id=0x1234)

    exported = _run_on_end(processor, root_span, None)

    assert exported is root_span
    assert not any(k.startswith(SpanAttributeKey.TRACE_TAG_PREFIX) for k in exported.attributes)


def test_on_end_filters_mlflow_prefixed_tags():
    processor, _ = _make_processor()
    otel_trace_id = 0xDEADBEEF

    root_span = create_mock_otel_span(trace_id=otel_trace_id, span_id=0x1234)
    tags = {
        "user_tag": "keep_me",
        "mlflow.traceName": "my-trace",
        "mlflow.trace.spansLocation": "TRACKING_STORE",
        "mlflow.user": "alice",
        "mlflow.artifactLocation": "s3://bucket",
    }
    manager_trace = _make_manager_trace(tags=tags)

    exported = _run_on_end(processor, root_span, manager_trace)

    assert exported.attributes.get(SpanAttributeKey.TRACE_TAG_PREFIX + "user_tag") == "keep_me"
    mlflow_tag_attrs = [
        k
        for k in exported.attributes
        if k.startswith(SpanAttributeKey.TRACE_TAG_PREFIX)
        and k != SpanAttributeKey.TRACE_TAG_PREFIX + "user_tag"
    ]
    assert mlflow_tag_attrs == []
