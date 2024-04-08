import pytest

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.span_event import SpanEvent

from tests.tracing.conftest import mock_client as mock_trace_client  # noqa: F401


def test_json_deserialization(mock_trace_client):
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

    trace_data = mlflow.get_traces()[0].data
    trace_data_dict = trace_data.to_dict()

    # Compare events separately as it includes exception stacktrace which is hard to hardcode
    span_to_events = {span["name"]: span.pop("events") for span in trace_data_dict["spans"]}

    assert trace_data_dict == {
        "request": {"x": 2, "y": 5},
        "response": None,
        "spans": [
            {
                "name": "predict",
                "context": {
                    "request_id": trace_data.spans[0].context.request_id,
                    "span_id": trace_data.spans[0].context.span_id,
                },
                "span_type": "UNKNOWN",
                "parent_span_id": None,
                "start_time": trace_data.spans[0].start_time,
                "end_time": trace_data.spans[0].end_time,
                "status": {
                    "status_code": "ERROR",
                    "description": "Exception: Error!",
                },
                "inputs": {"x": 2, "y": 5},
                "outputs": None,
                "attributes": {"function_name": "predict"},
                # "events": ...,
            },
            {
                "name": "with_ok_event",
                "context": {
                    "request_id": trace_data.spans[1].context.request_id,
                    "span_id": trace_data.spans[1].context.span_id,
                },
                "span_type": "UNKNOWN",
                "parent_span_id": trace_data.spans[0].context.span_id,
                "start_time": trace_data.spans[1].start_time,
                "end_time": trace_data.spans[1].end_time,
                "status": {
                    "status_code": "OK",
                    "description": "",
                },
                "inputs": None,
                "outputs": None,
                "attributes": {},
                # "events": ...,
            },
            {
                "name": "always_fail_name",
                "context": {
                    "request_id": trace_data.spans[2].context.request_id,
                    "span_id": trace_data.spans[2].context.span_id,
                },
                "span_type": "LLM",
                "parent_span_id": trace_data.spans[0].context.span_id,
                "start_time": trace_data.spans[2].start_time,
                "end_time": trace_data.spans[2].end_time,
                "status": {
                    "status_code": "ERROR",
                    "description": "Exception: Error!",
                },
                "inputs": {},
                "outputs": None,
                "attributes": {"delta": 1, "function_name": "always_fail"},
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
