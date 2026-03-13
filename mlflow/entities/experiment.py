from __future__ import annotations

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.entities.trace_location import UnityCatalog
from mlflow.protos.service_pb2 import Experiment as ProtoExperiment
from mlflow.protos.service_pb2 import ExperimentTag as ProtoExperimentTag
from mlflow.utils.mlflow_tags import (
    MLFLOW_EXPERIMENT_DATABRICKS_TRACE_ANNOTATIONS_TABLE,
    MLFLOW_EXPERIMENT_DATABRICKS_TRACE_DESTINATION_PATH,
    MLFLOW_EXPERIMENT_DATABRICKS_TRACE_LOG_STORAGE_TABLE,
    MLFLOW_EXPERIMENT_DATABRICKS_TRACE_SPAN_STORAGE_TABLE,
)
from mlflow.utils.workspace_utils import resolve_entity_workspace_name


class Experiment(_MlflowObject):
    """
    Experiment object.
    """

    DEFAULT_EXPERIMENT_NAME = "Default"

    def __init__(
        self,
        experiment_id,
        name,
        artifact_location,
        lifecycle_stage,
        tags=None,
        creation_time=None,
        last_update_time=None,
        workspace=None,
        trace_location=None,
    ):
        super().__init__()
        self._experiment_id = experiment_id
        self._name = name
        self._artifact_location = artifact_location
        self._lifecycle_stage = lifecycle_stage
        self._tags = {tag.key: tag.value for tag in (tags or [])}
        self._creation_time = creation_time
        self._last_update_time = last_update_time
        self._workspace = resolve_entity_workspace_name(workspace)
        self._trace_location = trace_location

    @property
    def experiment_id(self):
        """String ID of the experiment."""
        return self._experiment_id

    @property
    def name(self):
        """String name of the experiment."""
        return self._name

    def _set_name(self, new_name):
        self._name = new_name

    @property
    def artifact_location(self):
        """String corresponding to the root artifact URI for the experiment."""
        return self._artifact_location

    @property
    def lifecycle_stage(self):
        """Lifecycle stage of the experiment. Can either be 'active' or 'deleted'."""
        return self._lifecycle_stage

    @property
    def tags(self):
        """Tags that have been set on the experiment."""
        return self._tags

    def _add_tag(self, tag):
        self._tags[tag.key] = tag.value

    @property
    def creation_time(self):
        return self._creation_time

    def _set_creation_time(self, creation_time):
        self._creation_time = creation_time

    @property
    def last_update_time(self):
        return self._last_update_time

    def _set_last_update_time(self, last_update_time):
        self._last_update_time = last_update_time

    @property
    def trace_location(self) -> UnityCatalog | None:
        """Trace storage location, if configured."""
        if self._trace_location is None:
            self._trace_location = self._resolve_trace_location_from_tags()
        return self._trace_location

    @trace_location.setter
    def trace_location(self, trace_location):
        self._trace_location = trace_location

    def _resolve_trace_location_from_tags(self) -> UnityCatalog | None:
        destination_path = self._tags.get(MLFLOW_EXPERIMENT_DATABRICKS_TRACE_DESTINATION_PATH)
        if not destination_path:
            return None

        match destination_path.split("."):
            case [catalog, schema, table_prefix]:
                location = UnityCatalog(catalog, schema, table_prefix)
                location._otel_spans_table_name = self._tags.get(
                    MLFLOW_EXPERIMENT_DATABRICKS_TRACE_SPAN_STORAGE_TABLE
                )
                location._otel_logs_table_name = self._tags.get(
                    MLFLOW_EXPERIMENT_DATABRICKS_TRACE_LOG_STORAGE_TABLE
                )
                location._annotations_table_name = self._tags.get(
                    MLFLOW_EXPERIMENT_DATABRICKS_TRACE_ANNOTATIONS_TABLE
                )
                return location
            case _:
                return None

    @property
    def workspace(self):
        """Workspace that owns the experiment, if known."""
        return self._workspace

    @classmethod
    def from_proto(cls, proto):
        experiment = cls(
            proto.experiment_id,
            proto.name,
            proto.artifact_location,
            proto.lifecycle_stage,
            # `creation_time` and `last_update_time` were added in MLflow 1.29.0. Experiments
            # created before this version don't have these fields and `proto.creation_time` and
            # `proto.last_update_time` default to 0. We should only set `creation_time` and
            # `last_update_time` if they are non-zero.
            creation_time=proto.creation_time or None,
            last_update_time=proto.last_update_time or None,
            workspace=None,
        )
        for proto_tag in proto.tags:
            experiment._add_tag(ExperimentTag.from_proto(proto_tag))
        return experiment

    def to_proto(self):
        experiment = ProtoExperiment()
        experiment.experiment_id = self.experiment_id
        experiment.name = self.name
        experiment.artifact_location = self.artifact_location
        experiment.lifecycle_stage = self.lifecycle_stage
        if self.creation_time:
            experiment.creation_time = self.creation_time
        if self.last_update_time:
            experiment.last_update_time = self.last_update_time
        experiment.tags.extend([
            ProtoExperimentTag(key=key, value=val) for key, val in self._tags.items()
        ])
        return experiment
