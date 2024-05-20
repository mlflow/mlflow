from unittest import mock

import mlflow
from mlflow.entities import LiveSpan, Trace
from mlflow.tracing.export.inference_table import (
    _TRACE_BUFFER,
    InferenceTableSpanExporter,
    _initialize_trace_buffer,
    pop_trace,
)
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import encode_span_id, encode_trace_id

from tests.tracing.helper import create_mock_otel_span, create_test_trace_info

_TRACE_ID = 12345
_REQUEST_ID = f"tr-{_TRACE_ID}"
_REQUEST_ID_2 = f"tr-{_TRACE_ID + 1}"


def test_export():
    otel_span = create_mock_otel_span(
        name="root",
        trace_id=_TRACE_ID,
        span_id=1,
        parent_id=None,
        start_time=0,
        end_time=1_000_000,  # 1 milisecond
    )
    span = LiveSpan(otel_span, request_id=_REQUEST_ID)
    span.set_inputs({"input1": "very long input" * 100})
    span.set_outputs("very long output" * 100)
    _register_span_and_trace(span)

    child_otel_span = create_mock_otel_span(
        name="child", trace_id=_TRACE_ID, span_id=2, parent_id=1
    )
    child_span = LiveSpan(child_otel_span, request_id=_REQUEST_ID)
    _register_span_and_trace(child_span)

    # Invalid span should be also ignored
    invalid_otel_span = create_mock_otel_span(trace_id=23456, span_id=1)

    exporter = InferenceTableSpanExporter()

    exporter.export([otel_span, invalid_otel_span])

    # Spans should be cleared from the trace manager
    assert len(exporter._trace_manager._traces) == 0

    # Trace should be added to the in-memory buffer and can be extracted
    assert len(_TRACE_BUFFER) == 1
    trace_dict = pop_trace(_REQUEST_ID)
    trace_info = trace_dict["info"]
    assert trace_info["timestamp_ms"] == 0
    assert trace_info["execution_time_ms"] == 1

    spans = trace_dict["data"]["spans"]
    assert len(spans) == 2
    assert spans[0]["name"] == "root"
    assert spans[0]["context"] == {
        "trace_id": encode_trace_id(_TRACE_ID),
        "span_id": encode_span_id(1),
    }
    assert isinstance(spans[0]["attributes"], dict)


def test_export_warn_invalid_attributes():
    otel_span = create_mock_otel_span(trace_id=_TRACE_ID, span_id=1)
    span = LiveSpan(otel_span, request_id=_REQUEST_ID)
    span.set_attribute("valid", "value")
    # # Users may set attribute directly to the OpenTelemetry span
    # otel_span.set_attribute("int", 1)
    span.set_attribute("str", "a")
    _register_span_and_trace(span)

    exporter = InferenceTableSpanExporter()
    exporter.export([otel_span])

    trace_dict = pop_trace(_REQUEST_ID)
    trace = Trace.from_dict(trace_dict)
    stored_span = trace.data.spans[0]
    assert stored_span.attributes == {
        "mlflow.traceRequestId": _REQUEST_ID,
        "mlflow.spanType": "UNKNOWN",
        "valid": "value",
        "str": "a",
    }

    # Users shouldn't set attribute directly to the OTel span
    otel_span.set_attribute("int", 1)
    exporter.export([otel_span])
    with mock.patch("mlflow.entities.span._logger.warning") as mock_warning:
        span.attributes
        mock_warning.assert_called_once()
        msg = mock_warning.call_args[0][0]
        assert msg.startswith("Failed to get value for key int")


def test_export_trace_buffer_not_exceeds_max_size(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACE_BUFFER_MAX_SIZE", "1")
    monkeypatch.setattr(
        mlflow.tracing.export.inference_table, "_TRACE_BUFFER", _initialize_trace_buffer()
    )

    exporter = InferenceTableSpanExporter()

    otel_span_1 = create_mock_otel_span(name="1", trace_id=_TRACE_ID, span_id=1)
    _register_span_and_trace(LiveSpan(otel_span_1, request_id=_REQUEST_ID))

    exporter.export([otel_span_1])

    assert pop_trace(_REQUEST_ID) is not None

    otel_span_2 = create_mock_otel_span(name="2", trace_id=_TRACE_ID + 1, span_id=1)
    _register_span_and_trace(LiveSpan(otel_span_2, request_id=_REQUEST_ID_2))

    exporter.export([otel_span_2])

    assert pop_trace(_REQUEST_ID) is None
    assert pop_trace(_REQUEST_ID_2) is not None


def _register_span_and_trace(span: LiveSpan):
    trace_manager = InMemoryTraceManager.get_instance()
    if span.parent_id is None:
        trace_info = create_test_trace_info(span.request_id, 0)
        trace_manager.register_trace(span._span.context.trace_id, trace_info)
    trace_manager.register_span(span)
