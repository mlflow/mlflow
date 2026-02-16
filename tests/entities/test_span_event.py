import time

import pytest

from mlflow.entities import SpanEvent
from mlflow.exceptions import MlflowException


def test_default_timestamp_is_in_nanoseconds():
    # Record the time before and after creating the event
    before_ns = int(time.time() * 1e9)
    event = SpanEvent(name="test_event")
    after_ns = int(time.time() * 1e9)

    # The event's timestamp should be between before and after in nanoseconds
    assert before_ns <= event.timestamp <= after_ns, (
        f"Timestamp {event.timestamp} is not in expected range "
        f"[{before_ns}, {after_ns}]. It may be using wrong time unit."
    )

    # Additionally, verify it's in the nanosecond range (should be very large)
    # A nanosecond timestamp should be around 1.7e18 (in 2024+)
    # A microsecond timestamp would be around 1.7e15
    assert event.timestamp > 1e18, (
        f"Timestamp {event.timestamp} appears to be in microseconds, "
        f"not nanoseconds (expected > 1e18)"
    )


def test_from_exception():
    exception = MlflowException("test")
    span_event = SpanEvent.from_exception(exception)
    assert span_event.name == "exception"
    assert span_event.attributes["exception.message"] == "test"
    assert span_event.attributes["exception.type"] == "MlflowException"
    assert span_event.attributes["exception.stacktrace"] is not None

    # Verify that the timestamp is in nanoseconds
    current_ns = int(time.time() * 1e9)
    # Allow 1 second tolerance
    assert abs(span_event.timestamp - current_ns) < 1e9, (
        f"Timestamp {span_event.timestamp} is too far from current time {current_ns}. "
        f"It may be using wrong time unit."
    )


@pytest.mark.parametrize(
    "event_attrs",
    [
        {},
        {"simple": "string"},
        {"number": 42, "float": 3.14, "bool": True},
        {"list": [1, 2, 3], "dict": {"nested": "value"}},
    ],
)
def test_span_event_to_otel_proto_conversion(event_attrs):
    # Create span event
    event = SpanEvent(
        name="test_event",
        timestamp=1234567890,
        attributes=event_attrs,
    )

    # Convert to OTel proto
    otel_proto_event = event.to_otel_proto()

    # Verify fields
    assert otel_proto_event.name == "test_event"
    assert otel_proto_event.time_unix_nano == 1234567890

    # Verify attributes
    from mlflow.tracing.utils.otlp import _decode_otel_proto_anyvalue

    decoded_attrs = {}
    for attr in otel_proto_event.attributes:
        decoded_attrs[attr.key] = _decode_otel_proto_anyvalue(attr.value)

    assert decoded_attrs == event_attrs
