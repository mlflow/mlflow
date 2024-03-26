from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

from opentelemetry import trace as trace_api


@dataclass
class Trace:
    """A trace object. (TODO: Add conceptual guide for tracing.)

    Args:
        trace_info: A lightweight object that contains the metadata of a trace.
        trace_data: A container object that holds the spans data of a trace.
    """

    trace_info: TraceInfo
    trace_data: TraceData

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    def _repr_mimebundle_(self, include=None, exclude=None):
        """
        This method is used to trigger custom display logic in IPython notebooks.
        See https://ipython.readthedocs.io/en/stable/config/integrating.html#MyObject
        for more details.

        At the moment, the only supported MIME type is "application/databricks.mlflow.trace",
        which contains a JSON representation of the Trace object. This object is deserialized
        in Databricks notebooks to display the Trace object in a nicer UI.
        """
        return {
            "application/databricks.mlflow.trace": self.to_json(),
            "text/plain": self.__repr__(),
        }


@dataclass
class TraceInfo:
    """A lightweight object that contains the metadata of a trace.

    Args:
        trace_id: Unique identifier of the trace.
        experiment_id: The ID of the experiment that contains the trace.
        start_time: Start time of the trace in microseconds, inherited from the root span.
        end_time: End time of the trace in microseconds, inherited from the root span.
        status: Status of the trace, inherited from the root span.
        attributes: Arbitrary string key-value pairs of other trace attributes such as
            name, source, root-level inputs and outputs, etc.
        tags: String key-value pairs to attach labels to the trace.
    """

    trace_id: str
    experiment_id: str
    start_time: int
    end_time: int
    status: Status
    attributes: Dict[str, str]
    tags: Dict[str, str]


@dataclass
class TraceData:
    """A container object that holds the spans data of a trace.

    Args:
        spans: List of spans that are part of the trace.
    """

    spans: List[Span]


@dataclass
class Span:
    """A span object. (TODO: Add conceptual guide for span vs trace.)

    Args:
        name: Name of the span.
        context: SpanContext object that contains the trace_id and span_id.
        parent_span_id: Id of the parent span. If None, the span is the root span.
        span_type: Type of the span. Can be a pre-defined enum or a custom string.
        status: Status of the span.
        start_time: Start time of the span in microseconds.
        end_time: End time of the span in microseconds.
        inputs: Inputs data of the span. Optional.
        outputs: Outputs data of the span. Optional.
        attributes: Arbitrary key-value pairs of the span attributes. Optional.
        events: List of events that happened during the span. Optional.
    """

    name: str
    context: SpanContext
    parent_span_id: Optional[str]
    span_type: Union[SpanType, str]
    status: Status
    start_time: int
    end_time: int
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    attributes: Optional[Dict[str, Any]] = None
    events: Optional[List[Event]] = None


@dataclass
class SpanContext:
    """
    Following OpenTelemetry spec, trace_id and span_id are packed into SpanContext object.
    This design is TBD: the motivation in the original spec is to restrict the
    access to other Span fields and also allow lighter serialization and deserialization
    for the purpose of trace propagation. However, since we don't have a clear use case for
    this, we may want to just flatten this into the Span object.
    https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/trace/api.md#spancontext

    Args:
        trace_id: Unique identifier of the trace.
        span_id: Unique identifier of the span.
    """

    trace_id: str
    span_id: str


# Not using enum as we want to allow custom span type string.
class SpanType:
    """
    Predefined set of span types.
    """

    LLM = "LLM"
    CHAIN = "CHAIN"
    AGENT = "AGENT"
    TOOL = "TOOL"
    CHAT_MODEL = "CHAT_MODEL"
    RETRIEVER = "RETRIEVER"
    PARSER = "PARSER"
    EMBEDDING = "EMBEDDING"
    RERANKER = "RERANKER"
    UNKNOWN = "UNKNOWN"


@dataclass
class Status:
    """
    Status of the span or the trace.

    Args:
        status_code: The status code of the span or the trace.
        description: Description of the status. Optional.
    """

    status_code: StatusCode
    description: str = ""


# NB: Using the OpenTelemetry native StatusCode values here, because span's set_status
#     method only accepts a StatusCode enum in their definition.
#     https://github.com/open-telemetry/opentelemetry-python/blob/8ed71b15fb8fc9534529da8ce4a21e686248a8f3/opentelemetry-sdk/src/opentelemetry/sdk/trace/__init__.py#L949
#     Working around this is possible, but requires some hack to handle automatic status
#     propagation mechanism, so here we just use the native object that meets our
#     current requirements at least. Nevertheless, declaring the new class extending
#     the OpenTelemetry Status class so users code doesn't have to import the OTel's
#     StatusCode object, which makes future migration easier.
class StatusCode:
    UNSET = trace_api.StatusCode.UNSET
    OK = trace_api.StatusCode.OK
    ERROR = trace_api.StatusCode.ERROR


@dataclass
class Event:
    """
    Point of time event that happened during the span.

    Args:
        name: Name of the event.
        timestamp: Point of time when the event happened in microseconds.
        attributes: Arbitrary key-value pairs of the event attributes.
    """

    name: str
    timestamp: Optional[int]
    attributes: Optional[Dict[str, Any]] = None
