from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.span_event import SpanEvent
from mlflow.protos import databricks_trace_server_pb2 as pb


class SpanKind(str, Enum):
    SPAN_KIND_UNSPECIFIED = "SPAN_KIND_UNSPECIFIED"
    SPAN_KIND_INTERNAL = "SPAN_KIND_INTERNAL"
    SPAN_KIND_SERVER = "SPAN_KIND_SERVER"
    SPAN_KIND_CLIENT = "SPAN_KIND_CLIENT"
    SPAN_KIND_PRODUCER = "SPAN_KIND_PRODUCER"
    SPAN_KIND_CONSUMER = "SPAN_KIND_CONSUMER"

    def to_proto(self) -> pb.Span.SpanKind:
        return pb.Span.SpanKind.Value(self)

    @classmethod
    def from_proto(cls, proto: int) -> "SpanKind":
        return SpanKind(pb.Span.SpanKind.Name(proto))


class SpanStatusCode(str, Enum):
    STATUS_CODE_UNSET = "STATUS_CODE_UNSET"
    STATUS_CODE_OK = "STATUS_CODE_OK"
    STATUS_CODE_ERROR = "STATUS_CODE_ERROR"

    def to_proto(self) -> pb.Span.SpanStatusCode:
        if self == SpanStatusCode.STATUS_CODE_UNSET:
            return pb.Span.Status.SpanStatusCode.STATUS_CODE_UNSET
        elif self == SpanStatusCode.STATUS_CODE_OK:
            return pb.Span.Status.SpanStatusCode.STATUS_CODE_OK
        elif self == SpanStatusCode.STATUS_CODE_ERROR:
            return pb.Span.STATUS_CODE_ERROR
        raise ValueError(f"Unknown SpanStatusCode: {self}")

    @classmethod
    def from_proto(cls, proto: pb.Span.SpanStatusCode) -> "SpanStatusCode":
        if proto == pb.Span.Status.SpanStatusCode.STATUS_CODE_UNSET:
            return cls.STATUS_CODE_UNSET
        elif proto == pb.Span.Status.SpanStatusCode.STATUS_CODE_OK:
            return cls.STATUS_CODE_OK
        elif proto == pb.Span.Status.SpanStatusCode.STATUS_CODE_ERROR:
            return cls.STATUS_CODE_ERROR
        raise ValueError(f"Unknown SpanStatusCode: {proto}")


@dataclass
class SpanStatus(_MlflowObject):
    message: str
    code: SpanStatusCode


@dataclass
class Span(_MlflowObject):
    trace_id: str
    span_id: str
    trace_state: str
    parent_span_id: Optional[str]
    flags: int
    name: str
    kind: SpanKind
    start_time_unix_nano: int
    end_time_unix_nano: int
    attributes: dict[str, Any] = field(default_factory=dict)
    dropped_attributes_count: int = 0
    events: list[SpanEvent] = field(default_factory=list)
    dropped_events_count: int = 0
    status: Optional[SpanStatus] = None

    def to_proto(self) -> pb.Span:
        return pb.Span(
            trace_id=self.trace_id,
            span_id=self.span_id,
            trace_state=self.trace_state,
            parent_span_id=self.parent_span_id or "",
            flags=self.flags,
            name=self.name,
            kind=self.kind.to_proto(),
            start_time_unix_nano=self.start_time_unix_nano,
            end_time_unix_nano=self.end_time_unix_nano,
            attributes=self.attributes,
            dropped_attributes_count=self.dropped_attributes_count,
            events=[event.to_proto() for event in self.events],
            dropped_events_count=self.dropped_events_count,
            status=self.status.to_proto() if self.status else None,
        )
