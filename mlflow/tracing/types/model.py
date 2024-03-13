from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class Trace:
    """TBA
    """
    trace_info: TraceInfo
    trace_data: TraceData


@dataclass
class TraceInfo:
    """A lightweight object that contains the metadata of a trace.

    Args:
        trace_id: Unique identifier of the trace.
        name: Name of the trace. Inherited from the root span name at the time of trace creation.
        start_time: Start time of the trace in microseconds, inherited from the root span.
        end_time: End time of the trace in microseconds, inherited from the root span.
        status: Status of the trace, inherited from the root span.
        inputs: Serialized and truncated input data of the root span.
        outputs: Serialized and truncated output data of the root span. The full data can be
            found in the root span object in TraceData.
        metadata: Key-value pairs set at the time of trace creation and immutable afterwords.
        tags: Key-value pairs that can be mutated by the user to attach additional information to the trace.
        source: The entity (model_id, app_version_id, etc) from which the trace is generated.
    """
    trace_id: str
    name: str
    start_time: int
    end_time: int
    status: Status
    inputs: str
    outputs: str
    metadata: Dict[str, Any]
    tags: Dict[str, Union[str, float]]
    source: Optional[str] = None


@dataclass
class TraceData:
    """A container object that holds the spans data of a trace.

    Args:
        spans: List of spans that are part of the trace.
    """
    spans: List[Span]


@dataclass
class Span:
    """TBA
    """
    name: str
    context: SpanContext
    parent_span_id: Optional[str]
    # Type of the span can be either a pre-defined enum or a custom string.
    span_type: Union[SpanType, str]
    status: Status
    # Start and end time of the span in microseconds
    start_time: int
    end_time: int
    # Inputs and outputs data of the span.
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    # Arbitrary key-value pairs of the span attributes.
    attributes: Optional[Dict[str, Any]] = None
    # Point of time events that happened during the span.
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
    from opentelemetry import trace as trace_api

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
