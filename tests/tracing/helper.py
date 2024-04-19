import time
from dataclasses import dataclass
from typing import Optional

import opentelemetry.trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan


def create_mock_otel_span(
    trace_id: int,
    span_id: int,
    name: str = "test_span",
    parent_id: Optional[int] = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
):
    """
    Create a mock OpenTelemetry span for testing purposes.

    OpenTelemetry doesn't allow creating a span outside of a tracer. So here we create a mock span
    that extends ReadableSpan (data object) and exposes the necessary attributes for testing.
    """

    @dataclass
    class _MockSpanContext:
        trace_id: str
        span_id: str

    class _MockOTelSpan(trace_api.Span, ReadableSpan):
        def __init__(
            self,
            name,
            context,
            parent,
            start_time=None,
            end_time=None,
            status=trace_api.Status(trace_api.StatusCode.UNSET),
        ):
            self._name = name
            self._parent = parent
            self._context = context
            self._start_time = start_time if start_time is not None else int(time.time() * 1e9)
            self._end_time = end_time
            self._status = status
            self._attributes = {}

        # NB: The following methods are defined as abstract method in the Span class.
        def set_attributes(self, attributes):
            self._attributes.update(attributes)

        def set_attribute(self, key, value):
            self._attributes[key] = value

        def set_status(self, status):
            self._status = status

        def add_event():
            pass

        def get_span_context(self):
            return self._context

        def is_recording(self):
            return self._end_time is None

        def update_name(self, name):
            self.name = name

        def end():
            pass

        def record_exception():
            pass

    return _MockOTelSpan(
        name=name,
        context=_MockSpanContext(trace_id, span_id),
        parent=_MockSpanContext(trace_id, parent_id) if parent_id else None,
        start_time=start_time,
        end_time=end_time,
    )
