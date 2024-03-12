from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class Trace:
    trace_info: "TraceInfo"
    trace_data: "TraceData"


@dataclass
class TraceInfo:
    trace_id: str
    # Trace name is inherited from the root span name at the time of trace creation.
    # This may be edited by the user later.
    name: str
    # Start and end time of the trace, inherited from the root span.
    start_time: int
    end_time: int
    # Status of the trace, inherited from the root span.
    status: "Status"
    # Input and output data of the root span, but serialized and truncated to fixed
    # length for the efficient storage and retrieval.
    inputs: str
    outputs: str
    # Metadata should only be set by the system and immutable.
    metadata: Dict[str, Any]
    # Tags can be mutated by the user to attach additional information to the trace.
    tags: Dict[str, Union[str, float]]
    # Save the entity (model_id, app_version_id, etc) from which the trace is generated
    source: Optional[str] = None


@dataclass
class TraceData:
    spans: List["Span"]


@dataclass
class Span:
    name: str
    context: "SpanContext"
    parent_span_id: Optional[str]
    # Type of the span can be either a pre-defined enum or a custom string.
    span_type: Union["SpanType", str]
    status: "Status"
    # Start and end time of the span in microseconds
    start_time: int
    end_time: int
    # Inputs and outputs data of the span.
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    # Arbitrary key-value pairs of the span attributes.
    attributes: Optional[Dict[str, Any]] = None
    # Point of time events that happened during the span.
    events: Optional[List["Event"]] = None


@dataclass
class SpanContext:
    """
    Following OpenTelemetry spec, trace_id and span_id are packed into SpanContext object.
    This design is TBD: the motivation in the original spec is to restrict the
    access to other Span fields and also allow lighter serialization and deserialization
    for the purpose of trace propagation. However, since we don't have a clear use case for
    this, we may want to just flatten this into the Span object.
    https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/trace/api.md#spancontext
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
    """

    status_code: "StatusCode"
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
    """

    name: str
    timestamp: Optional[int]
    attributes: Optional[Dict[str, Any]] = None
