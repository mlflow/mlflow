from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.entities.span_context import SpanContext
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatus


@dataclass
class Span(_MLflowObject):
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
    status: SpanStatus
    start_time: int
    end_time: int
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


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
