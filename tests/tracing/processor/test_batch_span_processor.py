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
    mock_exporter = mock.MagicMock()
    mock_exporter.export.return_value = None
    processor = _create_processor(use_batch=True, exporter=mock_exporter)

    trace_id = 12345

    root_span = create_mock_otel_span(
        trace_id=trace_id, span_id=1, parent_id=None, start_time=5_000_000
    )
    processor.on_start(root_span)

    child_span = create_mock_otel_span(
        trace_id=trace_id,
        span_id=2,
        parent_id=1,
        start_time=6_000_000,
        end_time=7_000_000,
    )
    processor.on_start(child_span)

    root_span._end_time = 9_000_000
    processor.on_end(child_span)
    processor.on_end(root_span)

    processor.force_flush(timeout_millis=5000)

    assert mock_exporter.export.call_count >= 1
    exported_spans = []
    for call in mock_exporter.export.call_args_list:
        exported_spans.extend(call[0][0])
    assert len(exported_spans) == 2
    processor.shutdown()
