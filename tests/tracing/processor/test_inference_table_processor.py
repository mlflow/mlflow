import json
from unittest import mock

import pytest

from mlflow.entities.span import LiveSpan
from mlflow.entities.trace_status import TraceStatus
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.processor.inference_table import (
    _HEADER_REQUEST_ID_KEY,
    InferenceTableSpanProcessor,
)
from mlflow.tracing.trace_manager import InMemoryTraceManager

from tests.tracing.helper import create_mock_otel_span, create_test_trace_info

_TRACE_ID = 12345
_REQUEST_ID = f"tr-{_TRACE_ID}"


@pytest.fixture
def flask_request():
    with mock.patch(
        "mlflow.tracing.processor.inference_table._get_flask_request"
    ) as mock_get_flask_request:
        request = mock_get_flask_request.return_value
        request.headers = {_HEADER_REQUEST_ID_KEY: _REQUEST_ID}
        yield request


def test_on_start(flask_request):
    # Root span should create a new trace on start
    span = create_mock_otel_span(
        trace_id=_TRACE_ID, span_id=1, parent_id=None, start_time=5_000_000
    )
    trace_manager = InMemoryTraceManager.get_instance()
    processor = InferenceTableSpanProcessor(span_exporter=mock.MagicMock())

    processor.on_start(span)

    assert span.attributes.get(SpanAttributeKey.REQUEST_ID) == json.dumps(_REQUEST_ID)
    assert _REQUEST_ID in InMemoryTraceManager.get_instance()._traces

    with trace_manager.get_trace(_REQUEST_ID) as trace:
        assert trace.info.request_id == _REQUEST_ID
        assert trace.info.experiment_id is None
        assert trace.info.timestamp_ms == 5
        assert trace.info.execution_time_ms is None
        assert trace.info.status == TraceStatus.IN_PROGRESS

    # Child span should not create a new trace
    child_span = create_mock_otel_span(
        trace_id=_TRACE_ID, span_id=2, parent_id=1, start_time=8_000_000
    )
    processor.on_start(child_span)

    assert child_span.attributes.get(SpanAttributeKey.REQUEST_ID) == json.dumps(_REQUEST_ID)

    # start time should not be overwritten
    with trace_manager.get_trace(_REQUEST_ID) as trace:
        assert trace.info.timestamp_ms == 5


def test_on_end():
    trace_info = create_test_trace_info(_REQUEST_ID, 0)
    trace_manager = InMemoryTraceManager.get_instance()
    trace_manager.register_trace(_TRACE_ID, trace_info)

    otel_span = create_mock_otel_span(
        name="foo",
        trace_id=_TRACE_ID,
        span_id=1,
        parent_id=None,
        start_time=5_000_000,
        end_time=9_000_000,
    )
    span = LiveSpan(otel_span, request_id=_REQUEST_ID)
    span.set_status("OK")
    span.set_inputs({"input1": "very long input" * 100})
    span.set_outputs({"output": "very long output" * 100})

    mock_exporter = mock.MagicMock()
    processor = InferenceTableSpanProcessor(span_exporter=mock_exporter)

    processor.on_end(otel_span)

    mock_exporter.export.assert_called_once_with((otel_span,))
    # Trace info should be updated according to the span attributes
    assert trace_info.status == TraceStatus.OK
    assert trace_info.execution_time_ms == 4

    # Non-root span should not be exported
    mock_exporter.reset_mock()
    child_span = create_mock_otel_span(trace_id=_TRACE_ID, span_id=2, parent_id=1)
    processor.on_end(child_span)
    mock_exporter.export.assert_not_called()
