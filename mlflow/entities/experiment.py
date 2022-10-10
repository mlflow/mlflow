from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.protos.service_pb2 import (
    Experiment as ProtoExperiment,
    ExperimentTag as ProtoExperimentTag,
)


class Experiment(_MLflowObject):
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
    ):
        super().__init__()
        self._experiment_id = experiment_id
        self._name = name
        self._artifact_location = artifact_location
        self._lifecycle_stage = lifecycle_stage
        self._tags = {tag.key: tag.value for tag in (tags or [])}
        self._creation_time = creation_time
        self._last_update_time = last_update_time

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
        experiment.tags.extend(
            [ProtoExperimentTag(key=key, value=val) for key, val in self._tags.items()]
        )
        return experiment
