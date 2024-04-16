import json
from datetime import datetime

import opentelemetry.trace as trace_api
import pytest
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

import mlflow
from mlflow.entities import Span, SpanEvent, SpanStatus, SpanStatusCode, SpanType
from mlflow.exceptions import MlflowException
from mlflow.tracing.provider import get_tracer
from mlflow.tracing.utils import format_span_id

from tests.tracing.conftest import clear_singleton  # noqa: F401


def test_wrap_active_span(clear_singleton):
    request_id = "tr-12345"

    tracer = get_tracer("test")
    with tracer.start_as_current_span("parent") as parent_span:
        span = Span(parent_span, request_id=request_id, span_type=SpanType.LLM)
        assert span.request_id == request_id
        assert span._trace_id == format_span_id(parent_span.context.trace_id)
        assert span.span_id == format_span_id(parent_span.context.span_id)
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
            span = Span(child_span, request_id=request_id)
            assert span.name == "child"
            assert span.parent_id == format_span_id(parent_span.context.span_id)


def test_wrap_non_active_span():
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
    span = Span(readable_span, request_id=request_id)

    assert span.request_id == request_id
    assert span._trace_id == format_span_id(12345)
    assert span.span_id == format_span_id(222)
    assert span.name == "test"
    assert span.start_time_ns == 99999
    assert span.end_time_ns == 100000
    assert span.parent_id == format_span_id(111)
    assert span.inputs == {"input": 1, "nested": {"foo": "bar"}}
    assert span.outputs == 2
    assert span.status == SpanStatus(SpanStatusCode.UNSPECIFIED, description="")
    assert span.get_attribute("key") == 3

    # Validate APIs that are not allowed to be called on non-active spans
    with pytest.raises(MlflowException, match=r"Calling set_inputs\(\) is not allowed"):
        span.set_inputs({"input": 1})

    with pytest.raises(MlflowException, match=r"Calling set_outputs\(\) is not allowed"):
        span.set_outputs(2)

    with pytest.raises(MlflowException, match=r"Calling set_attribute\(\) is not allowed"):
        span.set_attribute("key", 3)

    with pytest.raises(MlflowException, match=r"Calling set_status\(\) is not allowed"):
        span.set_status("OK")

    with pytest.raises(MlflowException, match=r"Calling add_event\(\) is not allowed"):
        span.add_event(SpanEvent("test_event"))

    with pytest.raises(MlflowException, match=r"Calling end\(\) is not allowed"):
        span.end()


def test_wrap_raise_for_invalid_otel_span():
    with pytest.raises(MlflowException, match=r"Invalid span instance is passed."):
        Span(None, request_id="tr-12345")


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


def test_dict_conversion(clear_singleton):
    request_id = "tr-12345"

    tracer = get_tracer("test")
    with tracer.start_as_current_span("parent") as parent_span:
        span = Span(parent_span, request_id=request_id, span_type=SpanType.LLM)

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

    # Loaded span should be immutable
    with pytest.raises(MlflowException, match=r"Calling set_status\(\) is not allowed"):
        recovered_span.set_status("OK")
