from datetime import datetime

from mlflow.entities import LiveSpan
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION_KEY
from mlflow.tracing.export.inference_table import (
    _TRACE_BUFFER,
    InferenceTableSpanExporter,
    get_completed_trace,
)
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import encode_span_id, encode_trace_id

from tests.tracing.helper import create_mock_otel_span, create_test_trace_info

_TRACE_ID = 12345
_REQUEST_ID = f"tr-{_TRACE_ID}"


def test_export():
    otel_span = create_mock_otel_span(
        name="root",
        trace_id=_TRACE_ID,
        span_id=1,
        parent_id=None,
        start_time=0,
        end_time=1_000_000_000,  # 1 second
    )
    span = LiveSpan(otel_span, request_id=_REQUEST_ID)
    span.set_inputs({"input1": "very long input" * 100})
    span.set_outputs("very long output" * 100)

    trace_info = create_test_trace_info(_REQUEST_ID, 0)
    trace_manager = InMemoryTraceManager.get_instance()
    trace_manager.register_trace(_TRACE_ID, trace_info)
    trace_manager.register_span(span)

    child_otel_span = create_mock_otel_span(
        name="child", trace_id=_TRACE_ID, span_id=2, parent_id=1
    )
    child_span = LiveSpan(child_otel_span, request_id=_REQUEST_ID)
    trace_manager.register_span(child_span)

    # Invalid span should be also ignored
    invalid_otel_span = create_mock_otel_span(trace_id=23456, span_id=1)

    exporter = InferenceTableSpanExporter()

    exporter.export([otel_span, invalid_otel_span])

    # Spans should be cleared from the trace manager
    assert len(exporter._trace_manager._traces) == 0

    # Trace should be added to the in-memory buffer and can be extracted
    assert len(_TRACE_BUFFER) == 1
    trace_dict = get_completed_trace(_REQUEST_ID)
    assert trace_dict[TRACE_SCHEMA_VERSION_KEY] == 2
    assert trace_dict["start_timestamp"] == datetime(1970, 1, 1, 0, 0)
    assert trace_dict["end_timestamp"] == datetime(1970, 1, 1, 0, 0, 1)
    spans = trace_dict["spans"]
    assert len(spans) == 2
    assert spans[0]["name"] == "root"
    assert spans[0]["context"] == {
        "trace_id": encode_trace_id(_TRACE_ID),
        "span_id": encode_span_id(1),
    }
    assert isinstance(spans[0]["attributes"], str)
