import json
import time
from dataclasses import dataclass
from typing import Dict

from opentelemetry.sdk.trace import ReadableSpan


def create_mock_otel_span(
    trace_id, span_id, name="test_span", parent_id=None, start_time=None, end_time=None
):
    """
    Create a mock OpenTelemetry span for testing purposes.

    OpenTelemetry doesn't allow creating a span outside of a tracer. So here we create a mock span
    that extends ReadableSpan (data object) and exposes the necessary attributes for testing.
    """
    if start_time is None:
        start_time = time.time_ns()
    if end_time is None:
        end_time = time.time_ns()

    @dataclass
    class _MockSpanContext:
        trace_id: str
        span_id: str

    class _MockOTelSpan(ReadableSpan):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._parent = kwargs.get("parent", None)
            self._attributes = {}

        def set_attribute(self, key, value):
            self._attributes[key] = value

        def set_status(self, status):
            self._status = status

    return _MockOTelSpan(
        name=name,
        context=_MockSpanContext(trace_id, span_id),
        parent=_MockSpanContext(trace_id, parent_id) if parent_id else None,
        start_time=start_time,
        end_time=end_time,
    )


def deser_attributes(attributes: Dict[str, str]):
    """
    Deserialize the attribute values from JSON strings.
    """
    return {key: json.loads(value) for key, value in attributes.items()}
