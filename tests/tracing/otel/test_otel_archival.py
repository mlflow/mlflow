from __future__ import annotations

import json

import pytest
from opentelemetry.sdk.resources import Resource as OTelResource
from opentelemetry.sdk.trace import Event as OTelEvent
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.proto.trace.v1.trace_pb2 import TracesData
from opentelemetry.trace import Status as OTelStatus
from opentelemetry.trace import StatusCode as OTelStatusCode

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
from mlflow.tracing.utils.otlp import _decode_otel_proto_anyvalue


def _make_span(
    trace_id: int,
    span_id: int,
    request_id: str = "tr-abc123",
    *,
    name: str = "test_span",
    parent_span_id: int | None = None,
    resource: OTelResource | None = None,
) -> Span:
    otel_span = OTelReadableSpan(
        name=name,
        context=build_otel_context(trace_id, span_id),
        parent=build_otel_context(trace_id, parent_span_id) if parent_span_id is not None else None,
        start_time=1_000_000,
        end_time=2_000_000,
        attributes={
            SpanAttributeKey.REQUEST_ID: json.dumps(request_id),
            SpanAttributeKey.INPUTS: json.dumps({"q": "hello"}),
            SpanAttributeKey.OUTPUTS: json.dumps({"a": "world"}),
            SpanAttributeKey.SPAN_TYPE: json.dumps("UNKNOWN"),
        },
        events=[
            OTelEvent(
                name="test_event",
                timestamp=1_500_000,
                attributes={"event.attr": "value"},
            )
        ],
        status=OTelStatus(OTelStatusCode.ERROR, "test failure"),
        resource=resource,
    )
    return Span(otel_span)


def test_rejects_empty_span_list():
    with pytest.raises(MlflowException, match="at least one span"):
        spans_to_traces_data_pb([])


def test_round_trip_single_span():
    original = [_make_span(trace_id=1, span_id=10)]
    restored = traces_data_pb_to_spans(spans_to_traces_data_pb(original))
    assert [span.to_dict() for span in restored] == [span.to_dict() for span in original]


def test_round_trip_multiple_spans_same_trace():
    originals = [
        _make_span(trace_id=1, span_id=10, name="root_span"),
        _make_span(trace_id=1, span_id=20, name="child_span", parent_span_id=10),
    ]
    restored = traces_data_pb_to_spans(spans_to_traces_data_pb(originals))
    assert [span.to_dict() for span in restored] == [span.to_dict() for span in originals]


def test_serialized_traces_data_preserves_resource_attributes():
    resource = OTelResource.create({
        "service.name": "test-service",
        "service.version": "1.0.0",
        "deployment.environment.name": "test",
    })

    traces_data = TracesData()
    traces_data.ParseFromString(
        spans_to_traces_data_pb([_make_span(trace_id=1, span_id=10, resource=resource)])
    )

    attrs = {
        attr.key: _decode_otel_proto_anyvalue(attr.value)
        for attr in traces_data.resource_spans[0].resource.attributes
    }

    assert attrs["service.name"] == "test-service"
    assert attrs["service.version"] == "1.0.0"
    assert attrs["deployment.environment.name"] == "test"


def test_rejects_multiple_otlp_trace_ids_even_with_same_request_id():
    spans = [
        _make_span(trace_id=1, span_id=10, request_id="tr-shared"),
        _make_span(trace_id=2, span_id=20, request_id="tr-shared"),
    ]
    with pytest.raises(MlflowException, match="distinct trace IDs"):
        spans_to_traces_data_pb(spans)


def test_rejects_empty_bytes():
    with pytest.raises(MlflowException, match="non-empty OTLP TracesData protobuf"):
        traces_data_pb_to_spans(TracesData().SerializeToString())


def test_artifact_path_constant():
    assert TRACE_ARCHIVAL_ARTIFACT_PATH == "artifacts"


def test_filename_constant():
    assert TRACE_ARCHIVAL_FILENAME == "traces.pb"
