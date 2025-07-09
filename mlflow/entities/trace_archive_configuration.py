from dataclasses import dataclass
from typing import Any

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.trace_location import TraceLocation
from mlflow.protos.databricks_trace_server_pb2 import TraceDestination as ProtoTraceDestination


@dataclass
class TraceArchiveConfiguration(_MlflowObject):
    """Information about where traces are stored/archived.

    Args:
        trace_location: The location of the trace source, represented as a
            :py:class:`~mlflow.entities.TraceLocation` object. This uniquely
            identifies a TraceArchiveConfiguration.
        spans_table_name: The full qualified name of the spans table in the format
            `catalog.schema.table`.
        events_table_name: The full qualified name of the events table in the format
            `catalog.schema.table`.
        spans_schema_version: The schema version of the spans table.
        events_schema_version: The schema version of the events table.
    """

    trace_location: TraceLocation
    spans_table_name: str
    events_table_name: str
    spans_schema_version: str
    events_schema_version: str

    def to_dict(self) -> dict[str, Any]:
        """Convert the TraceArchiveConfiguration object to a dictionary."""
        return {
            "trace_location": self.trace_location.to_dict(),
            "spans_table_name": self.spans_table_name,
            "events_table_name": self.events_table_name,
            "spans_schema_version": self.spans_schema_version,
            "events_schema_version": self.events_schema_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TraceArchiveConfiguration":
        """Create a TraceArchiveConfiguration object from a dictionary."""
        d = d.copy()
        if trace_location := d.get("trace_location"):
            d["trace_location"] = TraceLocation.from_dict(trace_location)
        return cls(**d)

    def to_proto(self) -> ProtoTraceDestination:
        """Convert the TraceArchiveConfiguration object to a protobuf message."""
        return ProtoTraceDestination(
            trace_location=self.trace_location.to_proto(),
            spans_table_name=self.spans_table_name,
            events_table_name=self.events_table_name,
            spans_schema_version=self.spans_schema_version,
            events_schema_version=self.events_schema_version,
        )

    @classmethod
    def from_proto(cls, proto: ProtoTraceDestination) -> "TraceArchiveConfiguration":
        """Create a TraceArchiveConfiguration object from a protobuf message."""
        return cls(
            trace_location=TraceLocation.from_proto(proto.trace_location),
            spans_table_name=proto.spans_table_name,
            events_table_name=proto.events_table_name,
            spans_schema_version=proto.spans_schema_version,
            events_schema_version=proto.events_schema_version,
        )