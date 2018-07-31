from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.protos.service_pb2 import Experiment as ProtoExperiment


class Experiment(_MLflowObject):
    """
    Experiment object for python client. Backend stores will hydrate this object in APIs.
    """
    DEFAULT_EXPERIMENT_ID = 0

    def __init__(self, experiment_id, name, artifact_location):
        super(Experiment, self).__init__()
        self._experiment_id = experiment_id
        self._name = name
        self._artifact_location = artifact_location

    @property
    def experiment_id(self):
        return self._experiment_id

    @property
    def name(self):
        return self._name

    @property
    def artifact_location(self):
        return self._artifact_location

    @classmethod
    def from_proto(cls, proto):
        return cls(proto.experiment_id, proto.name, proto.artifact_location)

    def to_proto(self):
        proto = ProtoExperiment()
        proto.experiment_id = self.experiment_id
        proto.name = self.name
        proto.artifact_location = self.artifact_location
        return proto

    @classmethod
    def _properties(cls):
        # TODO: Hard coding this list of props for now. There has to be a clearer way...
        return ["experiment_id", "name", "artifact_location"]
