import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from opentelemetry.util.types import AttributeValue

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.span_context import SpanContext
from mlflow.entities.span_event import SpanEvent


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
class Span(_MlflowObject):
    """A span object. OpenTelemetry compatible but defines subset of fields.

    Args:
        name: Name of the span.
        context: SpanContext object that contains the trace_id and span_id.
        parent_id: Id of the parent span. If None, the span is the root span.
        status: Status of the span.
        start_time: Start time of the span in microseconds.
        end_time: End time of the span in microseconds.
        span_type: Type of the span. Can be a pre-defined enum or a custom string.
        attributes: Arbitrary key-value pairs of the span attributes. Optional.
        events: List of events that happened during the span. Optional.
    """

    name: str
    context: SpanContext
    parent_id: Optional[str]
    status_code: str
    status_message: Optional[str]
    start_time: int
    end_time: int
    attributes: Dict[str, AttributeValue] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)
