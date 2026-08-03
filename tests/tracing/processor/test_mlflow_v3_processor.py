import json
from unittest import mock

import pytest
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExportResult

import mlflow.tracking.context.default_context
from mlflow.entities.span import LiveSpan
from mlflow.entities.trace_status import TraceStatus
from mlflow.environment_variables import MLFLOW_TRACKING_USERNAME
from mlflow.tracing.constant import (
    SpanAttributeKey,
    TraceMetadataKey,
)
from mlflow.tracing.export.mlflow_v3 import MlflowV3SpanExporter
from mlflow.tracing.processor.mlflow_v3 import MlflowV3SpanProcessor
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import encode_trace_id

from tests.tracing.helper import (
    create_mock_otel_span,
    create_test_trace_info,
    skip_when_testing_trace_sdk,
)


def test_on_start(monkeypatch):
    monkeypatch.setattr(mlflow.tracking.context.default_context, "_get_source_name", lambda: "test")
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "bob")

    # Root span should create a new trace on start
    trace_id = 12345
    span = create_mock_otel_span(trace_id=trace_id, span_id=1, parent_id=None, start_time=5_000_000)

    processor = MlflowV3SpanProcessor(span_exporter=mock.MagicMock(), export_metrics=False)
    processor.on_start(span)

    # V3 processor uses encoded Otel trace_id as request_id
    request_id = "tr-" + encode_trace_id(trace_id)
    assert len(request_id) == 35  # 3 for "tr-" prefix + 32 for encoded trace_id
    assert span.attributes.get(SpanAttributeKey.REQUEST_ID) == json.dumps(request_id)
    assert request_id in InMemoryTraceManager.get_instance()._traces

    # Child span should not create a new trace
    child_span = create_mock_otel_span(
        trace_id=trace_id, span_id=2, parent_id=1, start_time=8_000_000
    )
    processor.on_start(child_span)
    assert child_span.attributes.get(SpanAttributeKey.REQUEST_ID) == json.dumps(request_id)


@skip_when_testing_trace_sdk
def test_on_start_during_model_evaluation():
    from mlflow.pyfunc.context import Context, set_prediction_context

    trace_id = 12345
    request_id = "tr-" + encode_trace_id(trace_id)

    # Root span should create a new trace on start
    span = create_mock_otel_span(trace_id=trace_id, span_id=1)
    processor = MlflowV3SpanProcessor(span_exporter=mock.MagicMock(), export_metrics=False)

    with set_prediction_context(Context(request_id=request_id, is_evaluate=True)):
        processor.on_start(span)

    assert span.attributes.get(SpanAttributeKey.REQUEST_ID) == json.dumps(request_id)


@skip_when_testing_trace_sdk
def test_on_start_during_run(monkeypatch):
    monkeypatch.setattr(mlflow.tracking.context.default_context, "_get_source_name", lambda: "test")
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "bob")

    span = create_mock_otel_span(trace_id=12345, span_id=1, parent_id=None, start_time=5_000_000)

    env_experiment_name = "env_experiment_id"
    run_experiment_name = "run_experiment_id"

    mlflow.create_experiment(env_experiment_name)
    run_experiment_id = mlflow.create_experiment(run_experiment_name)

    mlflow.set_experiment(experiment_name=env_experiment_name)
    processor = MlflowV3SpanProcessor(span_exporter=mock.MagicMock(), export_metrics=False)

    with mlflow.start_run(experiment_id=run_experiment_id) as run:
        processor.on_start(span)

        trace_id = "tr-" + encode_trace_id(span.context.trace_id)
        trace = InMemoryTraceManager.get_instance()._traces[trace_id]
        assert trace.info.experiment_id == run_experiment_id
        assert trace.info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run.info.run_id


def test_incremental_span_name_no_deduplication():
    InMemoryTraceManager.reset()
    trace_manager = InMemoryTraceManager.get_instance()

    trace_id = 12345
    request_id = "tr-" + encode_trace_id(trace_id)
    processor = MlflowV3SpanProcessor(span_exporter=mock.MagicMock(), export_metrics=False)

    # Helper to create and register a span
    def create_and_register(name, span_id, parent_id=1):
        span = create_mock_otel_span(
            name=name,
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
            start_time=span_id * 1_000_000,
            end_time=(span_id + 1) * 1_000_000,
        )
        processor.on_start(span)
        live_span = LiveSpan(span, request_id)
        trace_manager.register_span(live_span)
        processor.on_end(span)
        return span

    # Create root and 4 child spans: 3 "process" and 2 "query"
    create_and_register("process", 1, parent_id=None)
    create_and_register("process", 2)
    create_and_register("query", 3)
    create_and_register("process", 4)
    create_and_register("query", 5)

    with trace_manager.get_trace(request_id) as trace:
        names = [s.name for s in trace.span_dict.values() if s.name == "process"]
        assert len(names) == 3

    with trace_manager.get_trace(request_id) as trace:
        names = [s.name for s in trace.span_dict.values() if s.name == "query"]
        assert len(names) == 2

    with trace_manager.get_trace(request_id) as trace:
        spans_sorted_by_creation = sorted(trace.span_dict.values(), key=lambda s: s.start_time_ns)
        final_names = [s.name for s in spans_sorted_by_creation]
        assert final_names == ["process", "process", "query", "process", "query"]


def test_on_end():
    trace_info = create_test_trace_info("request_id", 0)
    trace_manager = InMemoryTraceManager.get_instance()
    trace_manager.register_trace("trace_id", trace_info)

    otel_span = create_mock_otel_span(
        name="foo",
        trace_id="trace_id",
        span_id=1,
        parent_id=None,
        start_time=5_000_000,
        end_time=9_000_000,
    )
    span = LiveSpan(otel_span, "request_id")
    span.set_status("OK")
    span.set_inputs({"input1": "very long input" * 100})
    span.set_outputs({"output": "very long output" * 100})

    mock_exporter = mock.MagicMock()
    mock_client = mock.MagicMock()
    mock_client._start_tracked_trace.side_effect = Exception("error")
    processor = MlflowV3SpanProcessor(span_exporter=mock_exporter, export_metrics=False)

    processor.on_end(otel_span)

    mock_exporter.export.assert_called_once_with((otel_span,))

    # Child spans should be exported
    mock_exporter.reset_mock()
    child_span = create_mock_otel_span(trace_id="trace_id", span_id=2, parent_id=1)
    # Set the REQUEST_ID attribute so the processor can find the trace
    child_span.set_attribute(SpanAttributeKey.REQUEST_ID, json.dumps("request_id"))
    processor.on_end(child_span)
    mock_exporter.export.assert_called_once_with((child_span,))

    # Trace info should be updated according to the span attributes
    manager_trace = trace_manager.pop_trace("trace_id")
    trace_info = manager_trace.trace.info
    assert trace_info.status == TraceStatus.OK
    assert trace_info.execution_time_ms == 4
    assert trace_info.tags == {}


# ── Batch span processor tests ──────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_trace_manager():
    InMemoryTraceManager.reset()
    yield
    InMemoryTraceManager.reset()


def _create_processor(*, use_batch: bool = False, exporter=None):
    return MlflowV3SpanProcessor(
        span_exporter=exporter or mock.MagicMock(),
        export_metrics=False,
        use_batch_processor=use_batch,
    )


def test_on_end_delegates_to_batch_processor():
    mock_exporter = mock.MagicMock()
    processor = _create_processor(use_batch=True, exporter=mock_exporter)
    try:
        trace_info = create_test_trace_info("request_id", 0)
        trace_manager = InMemoryTraceManager.get_instance()
        trace_manager.register_trace("trace_id", trace_info)

        otel_span = create_mock_otel_span(
            trace_id="trace_id",
            span_id=1,
            parent_id=None,
            start_time=5_000_000,
            end_time=9_000_000,
        )
        LiveSpan(otel_span, "request_id")

        with mock.patch.object(processor._batch_delegate, "on_end") as mock_on_end:
            processor.on_end(otel_span)
            mock_on_end.assert_called_once_with(otel_span)

        mock_exporter.export.assert_not_called()
    finally:
        processor.shutdown()


def test_on_end_uses_simple_processor_when_batch_disabled():
    mock_exporter = mock.MagicMock()
    processor = _create_processor(use_batch=False, exporter=mock_exporter)

    trace_info = create_test_trace_info("request_id", 0)
    trace_manager = InMemoryTraceManager.get_instance()
    trace_manager.register_trace("trace_id", trace_info)

    otel_span = create_mock_otel_span(
        trace_id="trace_id",
        span_id=1,
        parent_id=None,
        start_time=5_000_000,
        end_time=9_000_000,
    )
    LiveSpan(otel_span, "request_id")

    processor.on_end(otel_span)

    mock_exporter.export.assert_called_once_with((otel_span,))


def test_shutdown_delegates_to_batch_processor():
    processor = _create_processor(use_batch=True)

    with mock.patch.object(
        processor._batch_delegate,
        "shutdown",
        wraps=processor._batch_delegate.shutdown,
    ) as mock_shutdown:
        processor.shutdown()
        mock_shutdown.assert_called_once()


def test_shutdown_uses_simple_processor_when_batch_disabled():
    mock_exporter = mock.MagicMock()
    processor = _create_processor(use_batch=False, exporter=mock_exporter)

    processor.shutdown()

    mock_exporter.shutdown.assert_called_once()


def test_force_flush_delegates_to_batch_processor():
    processor = _create_processor(use_batch=True)
    try:
        with mock.patch.object(
            processor._batch_delegate, "force_flush", return_value=True
        ) as mock_flush:
            result = processor.force_flush(timeout_millis=5000)
            mock_flush.assert_called_once_with(5000)
            assert result is True
    finally:
        processor.shutdown()


def test_force_flush_uses_simple_processor_when_batch_disabled():
    mock_exporter = mock.MagicMock()
    processor = _create_processor(use_batch=False, exporter=mock_exporter)

    result = processor.force_flush()

    assert result is True


def test_batch_delegate_is_none_when_batch_disabled():
    processor = _create_processor(use_batch=False)
    assert processor._batch_delegate is None


def test_batch_delegate_is_created_when_batch_enabled():
    processor = _create_processor(use_batch=True)
    try:
        assert isinstance(processor._batch_delegate, BatchSpanProcessor)
    finally:
        processor.shutdown()


def test_batch_processor_reads_existing_env_vars(monkeypatch):
    mock_exporter = mock.MagicMock()
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_MAX_INTERVAL_MILLIS", "1000")
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_MAX_SPAN_BATCH_SIZE", "256")

    with mock.patch("mlflow.tracing.processor.base_mlflow.BatchSpanProcessor") as mock_batch_cls:
        MlflowV3SpanProcessor(
            span_exporter=mock_exporter,
            export_metrics=False,
            use_batch_processor=True,
        )
        mock_batch_cls.assert_called_once_with(
            mock_exporter,
            schedule_delay_millis=1000,
            max_queue_size=2048,
            max_export_batch_size=256,
        )


def test_spans_exported_in_batch_mode():

    mock_exporter = mock.MagicMock()
    mock_exporter.export.return_value = SpanExportResult.SUCCESS
    processor = _create_processor(use_batch=True, exporter=mock_exporter)
    try:
        trace_id = 12345

        spans = {}
        span_defs = [
            ("root", 1, None, 1_000_000, 20_000_000),
            ("child_a", 2, 1, 2_000_000, 15_000_000),
            ("child_b", 3, 1, 3_000_000, 18_000_000),
            ("grandchild_a1", 4, 2, 4_000_000, 8_000_000),
            ("grandchild_a2", 5, 2, 5_000_000, 10_000_000),
            ("grandchild_b1", 6, 3, 6_000_000, 12_000_000),
        ]
        for name, span_id, parent_id, start, end in span_defs:
            span = create_mock_otel_span(
                name=name,
                trace_id=trace_id,
                span_id=span_id,
                parent_id=parent_id,
                start_time=start,
                end_time=end,
            )
            spans[name] = span
            processor.on_start(span)

        for name in [
            "grandchild_a1",
            "grandchild_a2",
            "grandchild_b1",
            "child_a",
            "child_b",
            "root",
        ]:
            processor.on_end(spans[name])

        processor.force_flush()

        exported_spans = []
        for call in mock_exporter.export.call_args_list:
            exported_spans.extend(call[0][0])

        assert len(exported_spans) == 6
        exported_names = {s.name for s in exported_spans}
        assert exported_names == {
            "root",
            "child_a",
            "child_b",
            "grandchild_a1",
            "grandchild_a2",
            "grandchild_b1",
        }
    finally:
        processor.shutdown()


@skip_when_testing_trace_sdk
def test_on_end_bypasses_batch_during_evaluation():

    mock_exporter = mock.MagicMock()
    processor = _create_processor(use_batch=True, exporter=mock_exporter)
    try:
        trace_info = create_test_trace_info("request_id", 0)
        trace_manager = InMemoryTraceManager.get_instance()
        trace_manager.register_trace("trace_id", trace_info)

        otel_span = create_mock_otel_span(
            trace_id="trace_id",
            span_id=1,
            parent_id=None,
            start_time=5_000_000,
            end_time=9_000_000,
        )
        LiveSpan(otel_span, "request_id")

        from mlflow.pyfunc.context import Context, set_prediction_context

        with set_prediction_context(Context(request_id="eval-req-1", is_evaluate=True)):
            processor.on_end(otel_span)

        # Should call export directly (simple path), not batch delegate
        mock_exporter.export.assert_called_once_with((otel_span,))
    finally:
        processor.shutdown()


def test_set_last_active_trace_id_called_once_for_root_span(monkeypatch):

    exporter = MlflowV3SpanExporter()
    processor = _create_processor(exporter=exporter)

    trace_info = create_test_trace_info("request_id", 0)
    trace_manager = InMemoryTraceManager.get_instance()
    trace_manager.register_trace("trace_id", trace_info)

    otel_span = create_mock_otel_span(
        trace_id="trace_id",
        span_id=1,
        parent_id=None,
        start_time=5_000_000,
        end_time=9_000_000,
    )
    LiveSpan(otel_span, "request_id")

    # Patch both the canonical function and the already-imported reference
    with mock.patch("mlflow.tracing.fluent._set_last_active_trace_id") as mock_set_id:
        monkeypatch.setattr(
            "mlflow.tracing.processor.base_mlflow._set_last_active_trace_id", mock_set_id
        )
        processor.on_end(otel_span)
        # Must be called exactly once — from the processor only, not again in the exporter
        mock_set_id.assert_called_once_with("request_id")
