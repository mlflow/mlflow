from dataclasses import dataclass
from typing import Any

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.utils.annotations import experimental


@experimental(version="3.2.0")
@dataclass
class DatabricksTraceDeltaStorageConfig(_MlflowObject):
    """Information about where traces are stored/archived in Databricks.

    Args:
        experiment_id: The ID of the MLflow experiment where traces are archived.
        spans_table_name: The full qualified name of the open telemetry compatible
            spans table in the format
            `catalog.schema.table`.
        events_table_name: The full qualified name of the open telemetry compatible
            events table in the format
            `catalog.schema.table`.
        spans_schema_version: The schema version of the open telemetry compatible spans table.
        events_schema_version: The schema version of the open telemetry compatible events table.
    """

    experiment_id: str
    spans_table_name: str
    events_table_name: str
    spans_schema_version: str
    events_schema_version: str

    def to_dict(self) -> dict[str, Any]:
        """Convert the TraceArchiveConfiguration object to a dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "spans_table_name": self.spans_table_name,
            "events_table_name": self.events_table_name,
            "spans_schema_version": self.spans_schema_version,
            "events_schema_version": self.events_schema_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatabricksTraceDeltaStorageConfig":
        """Create a TraceArchiveConfiguration object from a dictionary."""
        return cls(**d)

    @classmethod
    def from_proto(cls, proto) -> "DatabricksTraceDeltaStorageConfig":
        """Create a DatabricksTraceDeltaStorageConfig object from a proto TraceDestination."""
        from mlflow.exceptions import MlflowException
        from mlflow.protos.databricks_trace_server_pb2 import TraceLocation as ProtoTraceLocation

        # Validate that this is an experiment location
        if proto.trace_location.type != ProtoTraceLocation.TraceLocationType.MLFLOW_EXPERIMENT:
            raise MlflowException(
                f"TraceArchiveConfiguration only supports MLflow experiments, "
                f"but got location type: {proto.trace_location.type}"
            )

        if not proto.trace_location.mlflow_experiment:
            raise MlflowException(
                "TraceArchiveConfiguration requires an MLflow experiment location, "
                "but mlflow_experiment is None"
            )

        return cls(
            experiment_id=proto.trace_location.mlflow_experiment.experiment_id,
            spans_table_name=proto.spans_table_name,
            events_table_name=proto.events_table_name,
            spans_schema_version=proto.spans_schema_version,
            events_schema_version=proto.events_schema_version,
        )
