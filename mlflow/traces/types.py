from dataclasses import dataclass
from enum import Enum
import logging
from time import time_ns
from typing import Any, Dict, List, Optional

from opentelemetry import trace as trace_api


_logger = logging.getLogger(__name__)


class SpanType(str, Enum):
    LLM = "LLM"
    UNKNOWN = "UNKNOWN"


@dataclass
class SpanContext:
    trace_id: str
    parent_span_id: Optional[str] = None


class Event:
    # TBA
    pass

@dataclass
class Span:
    span_id: str
    name: str
    context: SpanContext
    span_type: SpanType
    start_time: int
    end_time: int
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    attributes: Optional[Dict[str, Any]] = None
    events: Optional[List[Event]] = None



class MLflowSpanWrapper():
    def __init__(self, span: trace_api.Span, span_type: SpanType = SpanType.UNKNOWN):
        self._span = span
        self._span_type = span_type
        self._inputs = None
        self._outputs = None

    # Override the OTel's span end hook to pass this wrapper to processor/exporter
    # https://github.com/open-telemetry/opentelemetry-python/blob/216411f03a3a067177a0b927b668a87a60cf8797/opentelemetry-sdk/src/opentelemetry/sdk/trace/__init__.py#L909
    def end(self):
        with self._span._lock:
            if self._span._start_time is None:
                raise RuntimeError("Calling end() on a not started span.")
            if self._span._end_time is not None:
                _logger.warning("Calling end() on an ended span.")
                return

            self._span._end_time = time_ns()

        self._span._span_processor.on_end(self)

    @property
    def trace_id(self):
        return self._span.context.trace_id

    @property
    def name(self):
        return self._span.name

    @property
    def start_time(self):
        return self._span._start_time

    @property
    def end_time(self):
        return self._span._end_time

    @property
    def context(self):
        return self._span.get_span_context()

    @property
    def parent_span_id(self):
        if self._span.parent is None:
            return None
        return self._span.parent.span_id

    def set_inputs(self, inputs: Dict[str, Any]):
        self._inputs = inputs

    def set_outputs(self, outputs: Dict[str, Any]):
        self._outputs = outputs

    def set_attributes(self, attributes: Dict[str, Any]):
        self._span.set_attributes(attributes)

    def set_attribute(self, key: str, value: Any):
        self._span.set_attribute(key, value)

    def add_event(self, event: Event):
        self._span.add_event(event)

    def to_mlflow_span(self):
        return Span(
            span_id=self._span.get_span_context().span_id,
            name=self._span.name,
            context=SpanContext(
                trace_id=self.trace_id,
                parent_span_id=self.parent_span_id,
            ),
            span_type=self._span_type,
            start_time=self._span._start_time,
            end_time=self._span._end_time,
            inputs=self._inputs,
            outputs=self._outputs,
            attributes=self._span.attributes,
        )

    def is_root(self):
        return self._span.parent is None


@dataclass
class TraceInfo:
    trace_id: str
    trace_name: str
    start_time: int
    end_time: Optional[int]
	# model metadata
    model_uri: Optional[str] = None
    run_id: Optional[str] = None
    app_version_id: Optional[str] = None


@dataclass
class TraceData:
    spans: List[Span]


@dataclass
class Trace:
    trace_info: TraceInfo
    trace_data: TraceData
