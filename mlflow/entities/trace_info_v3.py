from dataclasses import dataclass, field

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.assessment import Assessment
from mlflow.entities.trace_location import TraceLocation


@dataclass
class TraceInfoV3(_MlflowObject):
    """Metadata about a trace.

    Args:
        request_id: id of the trace.
        experiment_id: id of the experiment.
        timestamp_ms: start time of the trace, in milliseconds.
        execution_time_ms: duration of the trace, in milliseconds.
        status: status of the trace.
        request_metadata: Key-value pairs associated with the trace. Request metadata are designed
            for immutable values like run ID associated with the trace.
        tags: Tags associated with the trace. Tags are designed for mutable values like trace name,
            that can be updated by the users after the trace is created, unlike request_metadata.
    """

    trace_id: str
    client_request_id: str
    trace_location: TraceLocation
    request: str
    response: str
    request_time: int
    execution_duration: int
    state: str
    trace_metadata: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    assessments: list[Assessment] = field(default_factory=list)

    def to_proto(self):
        pass

    @classmethod
    def from_proto(self, proto):
        pass
