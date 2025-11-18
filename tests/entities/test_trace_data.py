import json
from unittest import mock

import pytest
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.trace import SpanContext

import mlflow
from mlflow.entities import (
    Span,
    SpanStatus,
    SpanStatusCode,
    SpanType,
    Trace,
    TraceData,
    TraceInfo,
)
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.utils import dump_span_attribute_value


def test_json_deserialization():
    class TestModel:
        @mlflow.trace()
        def predict(self, x, y):
            z = x + y

            with mlflow.start_span(name="with_ok_event") as span:
                span.add_event(SpanEvent(name="ok_event", attributes={"foo": "bar"}))

            self.always_fail()
            return z

        @mlflow.trace(span_type=SpanType.LLM, name="always_fail_name", attributes={"delta": 1})
        def always_fail(self):
            raise Exception("Error!")

    model = TestModel()

    # Verify the exception is not absorbed by the context manager
    with pytest.raises(Exception, match="Error!"):
        model.predict(2, 5)

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    trace_data = trace.data

    # Compare events separately as it includes exception stacktrace which is hard to hardcode
    trace_data_dict = trace_data.to_dict()
    span_to_events = {span["name"]: span.get("events") for span in trace_data_dict["spans"]}

    assert trace_data_dict == {
        "spans": [
            {
                "name": "predict",
                "trace_id": mock.ANY,
                "span_id": mock.ANY,
                "parent_span_id": None,
                "start_time_unix_nano": trace.data.spans[0].start_time_ns,
                "end_time_unix_nano": trace.data.spans[0].end_time_ns,
                "status": {
                    "code": "STATUS_CODE_ERROR",
                    "message": "Exception: Error!",
                },
                "attributes": {
                    "mlflow.traceRequestId": json.dumps(trace.info.trace_id),
                    "mlflow.spanType": '"UNKNOWN"',
                    "mlflow.spanFunctionName": '"predict"',
                    "mlflow.spanInputs": '{"x": 2, "y": 5}',
                },
                "events": [
                    {
                        "name": "exception",
                        "time_unix_nano": trace.data.spans[0].events[0].timestamp,
                        "attributes": {
                            "exception.message": "Error!",
                            "exception.type": "Exception",
                            "exception.stacktrace": mock.ANY,
                            "exception.escaped": "False",
                        },
                    }
                ],
            },
            {
                "name": "with_ok_event",
                "trace_id": mock.ANY,
                "span_id": mock.ANY,
                "parent_span_id": mock.ANY,
                "start_time_unix_nano": trace.data.spans[1].start_time_ns,
                "end_time_unix_nano": trace.data.spans[1].end_time_ns,
                "status": {
                    "code": "STATUS_CODE_OK",
                    "message": "",
                },
                "attributes": {
                    "mlflow.traceRequestId": json.dumps(trace.info.trace_id),
                    "mlflow.spanType": '"UNKNOWN"',
                },
                "events": [
                    {
                        "name": "ok_event",
                        "time_unix_nano": trace.data.spans[1].events[0].timestamp,
                        "attributes": {"foo": "bar"},
                    }
                ],
            },
            {
                "name": "always_fail_name",
                "trace_id": mock.ANY,
                "span_id": mock.ANY,
                "parent_span_id": mock.ANY,
                "start_time_unix_nano": trace.data.spans[2].start_time_ns,
                "end_time_unix_nano": trace.data.spans[2].end_time_ns,
                "status": {
                    "code": "STATUS_CODE_ERROR",
                    "message": "Exception: Error!",
                },
                "attributes": {
                    "delta": "1",
                    "mlflow.traceRequestId": json.dumps(trace.info.trace_id),
                    "mlflow.spanType": '"LLM"',
                    "mlflow.spanFunctionName": '"always_fail"',
                    "mlflow.spanInputs": "{}",
                },
                "events": [
                    {
                        "name": "exception",
                        "time_unix_nano": trace.data.spans[2].events[0].timestamp,
                        "attributes": {
                            "exception.message": "Error!",
                            "exception.type": "Exception",
                            "exception.stacktrace": mock.ANY,
                            "exception.escaped": "False",
                        },
                    }
                ],
            },
        ],
    }

    ok_events = span_to_events["with_ok_event"]
    assert len(ok_events) == 1
    assert ok_events[0]["name"] == "ok_event"
    assert ok_events[0]["attributes"] == {"foo": "bar"}

    error_events = span_to_events["always_fail_name"]
    assert len(error_events) == 1
    assert error_events[0]["name"] == "exception"
    assert error_events[0]["attributes"]["exception.message"] == "Error!"
    assert error_events[0]["attributes"]["exception.type"] == "Exception"
    assert error_events[0]["attributes"]["exception.stacktrace"] is not None

    parent_events = span_to_events["predict"]
    assert len(parent_events) == 1
    assert parent_events[0]["name"] == "exception"
    assert parent_events[0]["attributes"]["exception.message"] == "Error!"
    assert parent_events[0]["attributes"]["exception.type"] == "Exception"
    # Parent span includes exception event bubbled up from the child span, hence the
    # stack trace includes the function call
    assert "self.always_fail()" in parent_events[0]["attributes"]["exception.stacktrace"]

    # Convert back from dict to TraceData and compare
    trace_data_from_dict = TraceData.from_dict(trace_data_dict)
    assert trace_data.to_dict() == trace_data_from_dict.to_dict()


def test_intermediate_outputs_from_attribute():
    intermediate_outputs = {
        "retrieved_documents": ["document 1", "document 2"],
        "generative_prompt": "prompt",
    }

    def run():
        with mlflow.start_span(name="run") as span:
            span.set_attribute("mlflow.trace.intermediate_outputs", intermediate_outputs)

    run()
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())

    assert trace.data.intermediate_outputs == intermediate_outputs


def test_intermediate_outputs_from_spans():
    @mlflow.trace()
    def retrieved_documents():
        return ["document 1", "document 2"]

    @mlflow.trace()
    def llm(i):
        return f"Hi, this is LLM {i}"

    @mlflow.trace()
    def predict():
        retrieved_documents()
        llm(1)
        llm(2)

    predict()
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())

    assert trace.data.intermediate_outputs == {
        "retrieved_documents": ["document 1", "document 2"],
        "llm_1": "Hi, this is LLM 1",
        "llm_2": "Hi, this is LLM 2",
    }


def test_intermediate_outputs_no_value():
    def run():
        with mlflow.start_span(name="run") as span:
            span.set_outputs(1)

    run()
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())

    assert trace.data.intermediate_outputs is None


def test_to_dict():
    with mlflow.start_span():
        pass
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    trace_dict = trace.data.to_dict()
    assert len(trace_dict["spans"]) == 1
    # Ensure the legacy properties are not present
    assert "request" not in trace_dict
    assert "response" not in trace_dict


def test_request_and_response_are_still_available():
    with mlflow.start_span() as s:
        s.set_inputs("foo")
        s.set_outputs("bar")

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    trace_data = trace.data
    assert trace_data.request == '"foo"'
    assert trace_data.response == '"bar"'

    with mlflow.start_span():
        pass

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    trace_data = trace.data
    assert trace_data.request == "{}"
    assert trace_data.response == '""'


@pytest.fixture
def span_without_inputs():
    trace_id = 12345
    span_id = 67890

    context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=0x01,
    )

    otel_span = OTelReadableSpan(
        name="external_function",
        context=context,
        parent=None,
        start_time=1000000000,
        end_time=1001000000,
        attributes={
            SpanAttributeKey.REQUEST_ID: dump_span_attribute_value(f"tr-test-{trace_id}"),
            SpanAttributeKey.SPAN_TYPE: dump_span_attribute_value("UNKNOWN"),
            SpanAttributeKey.OUTPUTS: dump_span_attribute_value("some output"),
            "custom.attribute": "some_value",
        },
        status=SpanStatus(SpanStatusCode.OK).to_otel_status(),
        events=[],
        resource=None,
    )

    return Span(otel_span)


def test_trace_data_request_returns_empty_dict_when_span_has_no_inputs(span_without_inputs):
    trace_data = TraceData(spans=[span_without_inputs])

    assert trace_data.request == "{}"


def test_trace_data_request_returns_empty_dict_when_no_root_span():
    trace_id = 12345
    parent_id = 11111
    span_id = 67890

    context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=0x01,
    )

    parent_context = SpanContext(
        trace_id=trace_id,
        span_id=parent_id,
        is_remote=False,
        trace_flags=0x01,
    )

    otel_span = OTelReadableSpan(
        name="child_span",
        context=context,
        parent=parent_context,
        start_time=1000000000,
        end_time=1001000000,
        attributes={
            SpanAttributeKey.REQUEST_ID: dump_span_attribute_value(f"tr-test-{trace_id}"),
        },
        status=SpanStatus(SpanStatusCode.OK).to_otel_status(),
        events=[],
        resource=None,
    )

    span = Span(otel_span)
    trace_data = TraceData(spans=[span])

    assert trace_data.request == "{}"


def test_trace_to_dataframe_with_missing_inputs(span_without_inputs):
    trace_info = TraceInfo(
        trace_id=span_without_inputs.trace_id,
        trace_location=TraceLocation.from_experiment_id("1"),
        request_time=1000,
        execution_duration=1000,
        state=TraceState.OK,
    )
    trace = Trace(info=trace_info, data=TraceData(spans=[span_without_inputs]))

    row = trace.to_pandas_dataframe_row()

    assert row["request"] == {}
    assert row["response"] == "some output"
