from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.protos.service_pb2 import Experiment as ProtoExperiment,\
    ExperimentTag as ProtoExperimentTag


class Experiment(_MLflowObject):
    """
    Experiment object.
    """
    DEFAULT_EXPERIMENT_NAME = "Default"

    def __init__(self, experiment_id, name, artifact_location, lifecycle_stage, tags=None):
        super(Experiment, self).__init__()
        self._experiment_id = experiment_id
        self._name = name
        self._artifact_location = artifact_location
        self._lifecycle_stage = lifecycle_stage
        self._tags = {tag.key: tag.value for tag in (tags or [])}

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

    @classmethod
    def from_proto(cls, proto):
        experiment = cls(proto.experiment_id,
                         proto.name,
                         proto.artifact_location,
                         proto.lifecycle_stage)
        for proto_tag in proto.tags:
            experiment._add_tag(ExperimentTag.from_proto(proto_tag))
        return experiment

    def to_proto(self):
        experiment = ProtoExperiment()
        experiment.experiment_id = self.experiment_id
        experiment.name = self.name
        experiment.artifact_location = self.artifact_location
        experiment.lifecycle_stage = self.lifecycle_stage
        experiment.tags.extend(
            [ProtoExperimentTag(key=key, value=val) for key, val in self._tags.items()])
        return experiment
