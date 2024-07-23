import json

import pytest

import mlflow
from mlflow.entities import SpanType, TraceData
from mlflow.entities.span_event import SpanEvent


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

    trace = mlflow.get_last_active_trace()
    trace_data = trace.data

    # Compare events separately as it includes exception stacktrace which is hard to hardcode
    trace_data_dict_copy = trace_data.to_dict().copy()
    span_to_events = {span["name"]: span.pop("events") for span in trace_data_dict_copy["spans"]}

    assert trace_data_dict_copy == {
        "request": '{"x": 2, "y": 5}',
        "response": None,
        "spans": [
            {
                "name": "predict",
                "context": {
                    "trace_id": trace.data.spans[0]._trace_id,
                    "span_id": trace.data.spans[0].span_id,
                },
                "parent_id": None,
                "start_time": trace.data.spans[0].start_time_ns,
                "end_time": trace.data.spans[0].end_time_ns,
                "status_code": "ERROR",
                "status_message": "Exception: Error!",
                "attributes": {
                    "mlflow.traceRequestId": json.dumps(trace.info.request_id),
                    "mlflow.spanType": '"UNKNOWN"',
                    "mlflow.spanFunctionName": '"predict"',
                    "mlflow.spanInputs": '{"x": 2, "y": 5}',
                },
                # "events": ...,
            },
            {
                "name": "with_ok_event",
                "context": {
                    "trace_id": trace.data.spans[1]._trace_id,
                    "span_id": trace.data.spans[1].span_id,
                },
                "parent_id": trace.data.spans[0].span_id,
                "start_time": trace.data.spans[1].start_time_ns,
                "end_time": trace.data.spans[1].end_time_ns,
                "status_code": "OK",
                "status_message": "",
                "attributes": {
                    "mlflow.traceRequestId": json.dumps(trace.info.request_id),
                    "mlflow.spanType": '"UNKNOWN"',
                },
                # "events": ...,
            },
            {
                "name": "always_fail_name",
                "context": {
                    "trace_id": trace.data.spans[2]._trace_id,
                    "span_id": trace.data.spans[2].span_id,
                },
                "parent_id": trace.data.spans[0].span_id,
                "start_time": trace.data.spans[2].start_time_ns,
                "end_time": trace.data.spans[2].end_time_ns,
                "status_code": "ERROR",
                "status_message": "Exception: Error!",
                "attributes": {
                    "delta": "1",
                    "mlflow.traceRequestId": json.dumps(trace.info.request_id),
                    "mlflow.spanType": '"LLM"',
                    "mlflow.spanFunctionName": '"always_fail"',
                    "mlflow.spanInputs": "{}",
                },
                # "events": ...,
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
    trace_data_from_dict = TraceData.from_dict(trace_data.to_dict())
    assert trace_data.to_dict() == trace_data_from_dict.to_dict()
