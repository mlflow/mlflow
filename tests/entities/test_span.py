import json
from datetime import datetime

import opentelemetry.trace as trace_api
import pytest
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

import mlflow
from mlflow.entities import LiveSpan, Span, SpanEvent, SpanStatus, SpanStatusCode, SpanType
from mlflow.entities.span import NoOpSpan, create_mlflow_span
from mlflow.exceptions import MlflowException
from mlflow.tracing.provider import _get_tracer, trace_disabled
from mlflow.tracing.utils import encode_span_id, encode_trace_id


def test_create_live_span():
    request_id = "tr-12345"

    tracer = _get_tracer("test")
    with tracer.start_as_current_span("parent") as parent_span:
        span = create_mlflow_span(parent_span, request_id=request_id, span_type=SpanType.LLM)
        assert isinstance(span, LiveSpan)
        assert span.request_id == request_id
        assert span._trace_id == encode_trace_id(parent_span.context.trace_id)
        assert span.span_id == encode_span_id(parent_span.context.span_id)
        assert span.name == "parent"
        assert span.start_time_ns == parent_span.start_time
        assert span.end_time_ns is None
        assert span.parent_id is None

        span.set_inputs({"input": 1})
        span.set_outputs(2)
        assert span.inputs == {"input": 1}
        assert span.outputs == 2

        span.set_attribute("key", 3)
        assert span.get_attribute("key") == 3

        # non-serializable value should be stored as string
        non_serializable = datetime.now()
        span.set_attribute("non_serializable", non_serializable)
        assert span.get_attribute("non_serializable") == str(non_serializable)
        assert parent_span._attributes == {
            "mlflow.traceRequestId": json.dumps(request_id),
            "mlflow.spanInputs": '{"input": 1}',
            "mlflow.spanOutputs": "2",
            "mlflow.spanType": '"LLM"',
            "key": "3",
            "non_serializable": json.dumps(str(non_serializable)),
        }

        span.set_status("OK")
        assert span.status == SpanStatus(SpanStatusCode.OK)

        span.add_event(SpanEvent("test_event", timestamp=99999, attributes={"foo": "bar"}))
        assert len(span.events) == 1
        assert span.events[0].name == "test_event"
        assert span.events[0].timestamp == 99999
        assert span.events[0].attributes == {"foo": "bar"}

        # Test child span
        with tracer.start_as_current_span("child") as child_span:
            span = create_mlflow_span(child_span, request_id=request_id)
            assert isinstance(span, LiveSpan)
            assert span.name == "child"
            assert span.parent_id == encode_span_id(parent_span.context.span_id)


def test_create_non_live_span():
    request_id = "tr-12345"
    parent_span_context = trace_api.SpanContext(
        trace_id=12345, span_id=111, is_remote=False, trace_flags=trace_api.TraceFlags(1)
    )
    readable_span = OTelReadableSpan(
        name="test",
        context=trace_api.SpanContext(
            trace_id=12345, span_id=222, is_remote=False, trace_flags=trace_api.TraceFlags(1)
        ),
        parent=parent_span_context,
        attributes={
            "mlflow.traceRequestId": json.dumps(request_id),
            "mlflow.spanInputs": '{"input": 1, "nested": {"foo": "bar"}}',
            "mlflow.spanOutputs": "2",
            "key": "3",
        },
        start_time=99999,
        end_time=100000,
    )
    span = create_mlflow_span(readable_span, request_id)

    assert isinstance(span, Span)
    assert not isinstance(span, LiveSpan)
    assert not isinstance(span, NoOpSpan)
    assert span.request_id == request_id
    assert span._trace_id == encode_trace_id(12345)
    assert span.span_id == encode_span_id(222)
    assert span.name == "test"
    assert span.start_time_ns == 99999
    assert span.end_time_ns == 100000
    assert span.parent_id == encode_span_id(111)
    assert span.inputs == {"input": 1, "nested": {"foo": "bar"}}
    assert span.outputs == 2
    assert span.status == SpanStatus(SpanStatusCode.UNSET, description="")
    assert span.get_attribute("key") == 3

    # Non-live span should not implement setter methods
    with pytest.raises(AttributeError, match="set_inputs"):
        span.set_inputs({"input": 1})


def test_create_noop_span():
    request_id = "tr-12345"

    @trace_disabled
    def f():
        tracer = _get_tracer("test")
        with tracer.start_as_current_span("span") as otel_span:
            span = create_mlflow_span(otel_span, request_id=request_id)
        assert isinstance(span, NoOpSpan)

    # create from None
    span = create_mlflow_span(None, request_id=request_id)
    assert isinstance(span, NoOpSpan)


def test_create_raise_for_invalid_otel_span():
    with pytest.raises(MlflowException, match=r"The `otel_span` argument must be"):
        create_mlflow_span(otel_span=123, request_id="tr-12345")


@pytest.mark.parametrize(
    "status",
    [SpanStatus("OK"), SpanStatus(SpanStatusCode.ERROR, "Error!"), "OK", "ERROR"],
)
def test_set_status(status):
    with mlflow.start_span("test_span") as span:
        span.set_status(status)

    assert isinstance(span.status, SpanStatus)


def test_set_status_raise_for_invalid_value():
    with mlflow.start_span("test_span") as span:
        with pytest.raises(MlflowException, match=r"INVALID is not a valid SpanStatusCode value."):
            span.set_status("INVALID")


def test_dict_conversion():
    request_id = "tr-12345"

    tracer = _get_tracer("test")
    with tracer.start_as_current_span("parent") as parent_span:
        span = LiveSpan(parent_span, request_id=request_id, span_type=SpanType.LLM)

    span_dict = span.to_dict()
    recovered_span = Span.from_dict(span_dict)

    assert span.request_id == recovered_span.request_id
    assert span._trace_id == recovered_span._trace_id
    assert span.span_id == recovered_span.span_id
    assert span.name == recovered_span.name
    assert span.start_time_ns == recovered_span.start_time_ns
    assert span.end_time_ns == recovered_span.end_time_ns
    assert span.parent_id == recovered_span.parent_id
    assert span.status == recovered_span.status
    assert span.inputs == recovered_span.inputs
    assert span.outputs == recovered_span.outputs
    assert span.attributes == recovered_span.attributes
    assert span.events == recovered_span.events

    # Loaded span should not implement setter methods
    with pytest.raises(AttributeError, match="set_status"):
        recovered_span.set_status("OK")


def test_to_immutable_span():
    request_id = "tr-12345"

    tracer = _get_tracer("test")
    with tracer.start_as_current_span("parent") as parent_span:
        live_span = LiveSpan(parent_span, request_id=request_id, span_type=SpanType.LLM)
        live_span.set_inputs({"input": 1})
        live_span.set_outputs(2)
        live_span.set_attribute("key", 3)
        live_span.set_status("OK")
        live_span.add_event(SpanEvent("test_event", timestamp=0, attributes={"foo": "bar"}))

    span = live_span.to_immutable_span()

    assert isinstance(span, Span)
    assert span.request_id == request_id
    assert span._trace_id == encode_trace_id(parent_span.context.trace_id)
    assert span.span_id == encode_span_id(parent_span.context.span_id)
    assert span.name == "parent"
    assert span.start_time_ns == parent_span.start_time
    assert span.end_time_ns is not None
    assert span.parent_id is None
    assert span.inputs == {"input": 1}
    assert span.outputs == 2
    assert span.get_attribute("key") == 3
    assert span.status == SpanStatus(SpanStatusCode.OK, description="")
    assert span.events == [SpanEvent("test_event", timestamp=0, attributes={"foo": "bar"})]

    with pytest.raises(AttributeError, match="set_attribute"):
        span.set_attribute("OK")


def test_from_dict_raises_when_request_id_is_empty():
    with pytest.raises(MlflowException, match=r"Failed to create a Span object from "):
        Span.from_dict(
            {
                "name": "predict",
                "context": {
                    "trace_id": "0x12345",
                    "span_id": "0x12345",
                },
                "parent_id": None,
                "start_time": 0,
                "end_time": 1,
                "status_code": "OK",
                "status_message": "",
                "attributes": {
                    "mlflow.traceRequestId": None,
                },
                "events": [],
            }
        )
