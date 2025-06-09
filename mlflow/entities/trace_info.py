import json
from dataclasses import dataclass, field
from typing import Any, Optional

from google.protobuf.duration_pb2 import Duration
from google.protobuf.json_format import MessageToDict
from google.protobuf.timestamp_pb2 import Timestamp

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.assessment import Assessment
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.entities.trace_status import TraceStatus
from mlflow.protos.service_pb2 import TraceInfoV3 as ProtoTraceInfoV3
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION, TRACE_SCHEMA_VERSION_KEY, TraceMetadataKey


@dataclass
class TraceInfo(_MlflowObject):
    """Metadata about a trace, such as its ID, location, timestamp, etc.

    Args:
        trace_id: The primary identifier for the trace.
        trace_location: The location where the trace is stored, represented as
            a :py:class:`~mlflow.entities.TraceLocation` object. MLflow currently
            support MLflow Experiment or Databricks Inference Table as a trace location.
        request_time: Start time of the trace, in milliseconds.
        state: State of the trace, represented as a :py:class:`~mlflow.entities.TraceState`
            enum. Can be one of [`OK`, `ERROR`, `IN_PROGRESS`, `STATE_UNSPECIFIED`].
        request_preview: Request to the model/agent, equivalent to the input of the root,
            span but JSON-encoded and can be truncated.
        response_preview: Response from the model/agent, equivalent to the output of the
            root span but JSON-encoded and can be truncated.
        client_request_id: Client supplied request ID associated with the trace. This
            could be used to identify the trace/request from an external system that
            produced the trace, e.g., a session ID in a web application.
        execution_duration: Duration of the trace, in milliseconds.
        trace_metadata: Key-value pairs associated with the trace. They are designed
            for immutable values like run ID associated with the trace.
        tags: Tags associated with the trace. They are designed for mutable values,
            that can be updated after the trace is created via MLflow UI or API.
        assessments: List of assessments associated with the trace.
    """

    trace_id: str
    trace_location: TraceLocation
    request_time: int
    state: TraceState
    request_preview: Optional[str] = None
    response_preview: Optional[str] = None
    client_request_id: Optional[str] = None
    execution_duration: Optional[int] = None
    trace_metadata: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    assessments: list[Assessment] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the TraceInfoV3 object to a dictionary."""
        res = MessageToDict(self.to_proto(), preserving_proto_field_name=True)
        if self.execution_duration is not None:
            res.pop("execution_duration", None)
            res["execution_duration_ms"] = self.execution_duration
        return res

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TraceInfo":
        """Create a TraceInfoV3 object from a dictionary."""
        if "request_id" in d:
            from mlflow.entities.trace_info_v2 import TraceInfoV2

            return TraceInfoV2.from_dict(d).to_v3()

        d = d.copy()
        if assessments := d.get("assessments"):
            d["assessments"] = [Assessment.from_dictionary(a) for a in assessments]

        if trace_location := d.get("trace_location"):
            d["trace_location"] = TraceLocation.from_dict(trace_location)

        if state := d.get("state"):
            d["state"] = TraceState(state)

        if request_time := d.get("request_time"):
            timestamp = Timestamp()
            timestamp.FromJsonString(request_time)
            d["request_time"] = timestamp.ToMilliseconds()

        if (execution_duration := d.pop("execution_duration_ms", None)) is not None:
            d["execution_duration"] = execution_duration

        return cls(**d)

    def to_proto(self):
        from mlflow.entities.trace_info_v2 import _truncate_request_metadata, _truncate_tags

        request_time = Timestamp()
        request_time.FromMilliseconds(self.request_time)
        execution_duration = None
        if self.execution_duration is not None:
            execution_duration = Duration()
            execution_duration.FromMilliseconds(self.execution_duration)

        return ProtoTraceInfoV3(
            trace_id=self.trace_id,
            client_request_id=self.client_request_id,
            trace_location=self.trace_location.to_proto(),
            request_preview=self.request_preview,
            response_preview=self.response_preview,
            request_time=request_time,
            execution_duration=execution_duration,
            state=self.state.to_proto(),
            trace_metadata=_truncate_request_metadata(self.trace_metadata),
            tags=_truncate_tags(self.tags),
            assessments=[a.to_proto() for a in self.assessments],
        )

    @classmethod
    def from_proto(cls, proto) -> "TraceInfo":
        trace_metadata = dict(proto.trace_metadata)
        # NB: MLflow automatically converts trace metadata and spans to V3 format, even if the
        # trace was originally created in V2 format with an earlier version of MLflow. Accordingly,
        # we also update the `TRACE_SCHEMA_VERSION_KEY` in the trace metadata to V3 for consistency
        trace_metadata[TRACE_SCHEMA_VERSION_KEY] = str(TRACE_SCHEMA_VERSION)

        return cls(
            trace_id=proto.trace_id,
            client_request_id=(
                proto.client_request_id if proto.HasField("client_request_id") else None
            ),
            trace_location=TraceLocation.from_proto(proto.trace_location),
            request_preview=proto.request_preview,
            response_preview=proto.response_preview,
            request_time=proto.request_time.ToMilliseconds(),
            execution_duration=(
                proto.execution_duration.ToMilliseconds()
                if proto.HasField("execution_duration")
                else None
            ),
            state=TraceState.from_proto(proto.state),
            trace_metadata=trace_metadata,
            tags=dict(proto.tags),
            assessments=[Assessment.from_proto(a) for a in proto.assessments],
        )

    # Aliases for backward compatibility with V2 format
    @property
    def request_id(self) -> str:
        """Deprecated. Use `trace_id` instead."""
        return self.trace_id

    @property
    def experiment_id(self) -> Optional[str]:
        """
        An MLflow experiment ID associated with the trace, if the trace is stored
        in MLflow tracking server. Otherwise, None.
        """
        return (
            self.trace_location.mlflow_experiment
            and self.trace_location.mlflow_experiment.experiment_id
        )

    @experiment_id.setter
    def experiment_id(self, value: Optional[str]) -> None:
        self.trace_location.mlflow_experiment.experiment_id = value

    @property
    def request_metadata(self) -> dict[str, str]:
        """Deprecated. Use `trace_metadata` instead."""
        return self.trace_metadata

    @property
    def timestamp_ms(self) -> int:
        return self.request_time

    @timestamp_ms.setter
    def timestamp_ms(self, value: int) -> None:
        self.request_time = value

    @property
    def execution_time_ms(self) -> Optional[int]:
        return self.execution_duration

    @execution_time_ms.setter
    def execution_time_ms(self, value: Optional[int]) -> None:
        self.execution_duration = value

    @property
    def status(self) -> TraceStatus:
        """Deprecated. Use `state` instead."""
        return TraceStatus.from_state(self.state)

    @status.setter
    def status(self, value: TraceStatus) -> None:
        self.state = value.to_state()

    @property
    def token_usage(self) -> Optional[dict[str, int]]:
        """
        Returns the aggregated token usage for the trace.

        Returns:
            A dictionary containing the aggregated LLM token usage for the trace.
            - "input_tokens": The total number of input tokens.
            - "output_tokens": The total number of output tokens.
            - "total_tokens": Sum of input and output tokens.

        .. note::

            The token usage tracking is not supported for all LLM providers.
            Refer to the MLflow Tracing documentation for which providers
            support token usage tracking.
        """
        if usage_json := self.trace_metadata.get(TraceMetadataKey.TOKEN_USAGE):
            return json.loads(usage_json)
        return None
