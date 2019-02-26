from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.protos.service_pb2 import Experiment as ProtoExperiment


class Experiment(_MLflowObject):
    """
    Experiment object.
    """
    DEFAULT_EXPERIMENT_ID = 0
    DEFAULT_EXPERIMENT_NAME = "Default"

    def __init__(self, experiment_id, name, artifact_location, lifecycle_stage):
        super(Experiment, self).__init__()
        self._experiment_id = experiment_id
        self._name = name
        self._artifact_location = artifact_location
        self._lifecycle_stage = lifecycle_stage

    @property
    def experiment_id(self):
        """Integer ID of the experiment."""
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

    @classmethod
    def from_proto(cls, proto):
        return cls(proto.experiment_id, proto.name, proto.artifact_location, proto.lifecycle_stage)

    def to_proto(self):
        proto = ProtoExperiment()
        proto.experiment_id = self.experiment_id
        proto.name = self.name
        proto.artifact_location = self.artifact_location
        proto.lifecycle_stage = self.lifecycle_stage
        return proto
