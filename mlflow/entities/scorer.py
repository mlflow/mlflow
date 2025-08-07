from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.service_pb2 import Scorer as ProtoScorer


class Scorer(_MlflowObject):
    """Scorer object associated with an experiment."""

    def __init__(self, experiment_id, scorer_name, scorer_version, serialized_scorer):
        self._experiment_id = experiment_id
        self._scorer_name = scorer_name
        self._scorer_version = scorer_version
        self._serialized_scorer = serialized_scorer

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def experiment_id(self):
        """Integer ID of the experiment this scorer belongs to."""
        return self._experiment_id

    @property
    def scorer_name(self):
        """String name of the scorer."""
        return self._scorer_name

    @property
    def scorer_version(self):
        """Integer version of the scorer."""
        return self._scorer_version

    @property
    def serialized_scorer(self):
        """String containing the serialized scorer data."""
        return self._serialized_scorer

    @classmethod
    def from_proto(cls, proto):
        """Create a Scorer from a protobuf message."""
        return cls(
            experiment_id=proto.experiment_id,
            scorer_name=proto.scorer_name,
            scorer_version=proto.scorer_version,
            serialized_scorer=proto.serialized_scorer,
        )

    def to_proto(self):
        """Convert this Scorer to a protobuf message."""
        proto = ProtoScorer()
        proto.experiment_id = self.experiment_id
        proto.scorer_name = self.scorer_name
        proto.scorer_version = self.scorer_version
        proto.serialized_scorer = self.serialized_scorer
        return proto

    def __repr__(self):
        return f"<Scorer(experiment_id={self.experiment_id}, scorer_name='{self.scorer_name}', scorer_version={self.scorer_version})>" 