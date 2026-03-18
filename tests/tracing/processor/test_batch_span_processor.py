from unittest import mock

import pytest

from mlflow.entities.span import LiveSpan
from mlflow.tracing.processor.mlflow_v3 import MlflowV3SpanProcessor
from mlflow.tracing.trace_manager import InMemoryTraceManager

from tests.tracing.helper import create_mock_otel_span, create_test_trace_info


@pytest.fixture(autouse=True)
def _reset_trace_manager():
    InMemoryTraceManager.reset()
    yield
    InMemoryTraceManager.reset()


def _create_processor(use_batch, exporter=None):
    return MlflowV3SpanProcessor(
        span_exporter=exporter or mock.MagicMock(),
        export_metrics=False,
        use_batch_processor=use_batch,
    )


def test_on_end_delegates_to_batch_processor():
    mock_exporter = mock.MagicMock()
    processor = _create_processor(use_batch=True, exporter=mock_exporter)

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

    with mock.patch.object(processor._batch_delegate, "shutdown") as mock_shutdown:
        processor.shutdown()
        mock_shutdown.assert_called_once()


def test_shutdown_uses_simple_processor_when_batch_disabled():
    mock_exporter = mock.MagicMock()
    processor = _create_processor(use_batch=False, exporter=mock_exporter)

    processor.shutdown()

    mock_exporter.shutdown.assert_called_once()


def test_force_flush_delegates_to_batch_processor():
    processor = _create_processor(use_batch=True)

    with mock.patch.object(
        processor._batch_delegate, "force_flush", return_value=True
    ) as mock_flush:
        result = processor.force_flush(timeout_millis=5000)
        mock_flush.assert_called_once_with(5000)
        assert result is True


def test_force_flush_uses_simple_processor_when_batch_disabled():
    mock_exporter = mock.MagicMock()
    processor = _create_processor(use_batch=False, exporter=mock_exporter)

    result = processor.force_flush(timeout_millis=5000)

    assert result is True


def test_batch_delegate_is_none_when_batch_disabled():
    processor = _create_processor(use_batch=False)
    assert processor._batch_delegate is None


def test_batch_delegate_is_created_when_batch_enabled():
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    processor = _create_processor(use_batch=True)
    assert isinstance(processor._batch_delegate, BatchSpanProcessor)
    processor.shutdown()


def test_custom_batch_params():
    mock_exporter = mock.MagicMock()
    with mock.patch("mlflow.tracing.processor.base_mlflow.BatchSpanProcessor") as mock_batch_cls:
        MlflowV3SpanProcessor(
            span_exporter=mock_exporter,
            export_metrics=False,
            use_batch_processor=True,
            batch_schedule_delay_millis=1000,
            batch_max_export_size=256,
        )
        mock_batch_cls.assert_called_once_with(
            mock_exporter,
            schedule_delay_millis=1000,
            max_export_batch_size=256,
        )


def test_spans_exported_in_batch_mode():
    """Verify that a multi-level span tree is fully exported via the batch path.

    Span tree:
        root (span_id=1)
        ├── child_a (span_id=2)
        │   ├── grandchild_a1 (span_id=4)
        │   └── grandchild_a2 (span_id=5)
        └── child_b (span_id=3)
            └── grandchild_b1 (span_id=6)
    """
    mock_exporter = mock.MagicMock()
    mock_exporter.export.return_value = None
    processor = _create_processor(use_batch=True, exporter=mock_exporter)

    trace_id = 12345

    # Build the span tree top-down (on_start order)
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

    # End spans leaf-first, as a real tracer would
    for name in ["grandchild_a1", "grandchild_a2", "grandchild_b1", "child_a", "child_b", "root"]:
        processor.on_end(spans[name])

    processor.force_flush(timeout_millis=5000)

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
    processor.shutdown()
