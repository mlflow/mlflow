from __future__ import annotations

import json

import pytest
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

from mlflow.entities.span import Span
from mlflow.exceptions import MlflowException
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.otel.otel_archival import (
    TRACE_ARCHIVAL_ARTIFACT_PATH,
    TRACE_ARCHIVAL_FILENAME,
    spans_to_traces_data_pb,
    traces_data_pb_to_spans,
)
from mlflow.tracing.utils import build_otel_context


def _make_span(trace_id: int, span_id: int, request_id: str = "tr-abc123") -> Span:
    otel_span = OTelReadableSpan(
        name="test_span",
        context=build_otel_context(trace_id, span_id),
        start_time=1_000_000,
        end_time=2_000_000,
        attributes={
            SpanAttributeKey.REQUEST_ID: request_id,
            SpanAttributeKey.INPUTS: json.dumps({"q": "hello"}),
            SpanAttributeKey.OUTPUTS: json.dumps({"a": "world"}),
            SpanAttributeKey.SPAN_TYPE: json.dumps("UNKNOWN"),
        },
    )
    return Span(otel_span)


def test_empty_list_returns_valid_bytes():
    data = spans_to_traces_data_pb([])
    assert isinstance(data, bytes)
    spans = traces_data_pb_to_spans(data)
    assert spans == []


def test_round_trip_single_span():
    original = [_make_span(trace_id=1, span_id=10)]
    data = spans_to_traces_data_pb(original)
    restored = traces_data_pb_to_spans(data)
    assert len(restored) == 1
    assert restored[0].name == "test_span"


def test_round_trip_multiple_spans_same_trace():
    originals = [
        _make_span(trace_id=1, span_id=10),
        _make_span(trace_id=1, span_id=20),
    ]
    data = spans_to_traces_data_pb(originals)
    restored = traces_data_pb_to_spans(data)
    assert len(restored) == 2


def test_rejects_multiple_trace_ids():
    spans = [
        _make_span(trace_id=1, span_id=10, request_id="tr-aaa"),
        _make_span(trace_id=2, span_id=20, request_id="tr-bbb"),
    ]
    with pytest.raises(MlflowException, match="distinct request IDs"):
        spans_to_traces_data_pb(spans)


def test_deserialize_empty_bytes_returns_empty_list():
    from opentelemetry.proto.trace.v1.trace_pb2 import TracesData

    data = TracesData().SerializeToString()
    assert traces_data_pb_to_spans(data) == []


def test_artifact_path_constant():
    assert TRACE_ARCHIVAL_ARTIFACT_PATH == "artifacts"


def test_filename_constant():
    assert TRACE_ARCHIVAL_FILENAME == "traces.pb"
