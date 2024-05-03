from mlflow.entities import SpanEvent
from mlflow.exceptions import MlflowException


def test_from_exception():
    exception = MlflowException("test")
    span_event = SpanEvent.from_exception(exception)
    assert span_event.name == "exception"
    assert span_event.attributes["exception.message"] == "test"
    assert span_event.attributes["exception.type"] == "MlflowException"
    assert span_event.attributes["exception.stacktrace"] is not None
