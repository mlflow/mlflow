import json
from functools import lru_cache

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.service_pb2 import Scorer as ProtoScorer


class ScorerVersion(_MlflowObject):
    """ScorerVersion object associated with an experiment."""

    def __init__(
        self,
        experiment_id: str,
        scorer_name: str,
        scorer_version: int,
        serialized_scorer: str,
        creation_time: int,
    ):
        self._experiment_id = experiment_id
        self._scorer_name = scorer_name
        self._scorer_version = scorer_version
        self._serialized_scorer = serialized_scorer
        self._creation_time = creation_time

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
    @lru_cache(maxsize=1)
    def serialized_scorer(self):
        """SerializedScorer object containing the metadata and function code for the scorer."""
        from mlflow.genai.scorers.base import SerializedScorer

        return SerializedScorer(**json.loads(self._serialized_scorer))

    @property
    def creation_time(self):
        """BigInteger creation time of the scorer version."""
        return self._creation_time

    @classmethod
    def from_proto(cls, proto):
        """Create a ScorerVersion from a protobuf message."""
        return cls(
            experiment_id=proto.experiment_id,
            scorer_name=proto.scorer_name,
            scorer_version=proto.scorer_version,
            serialized_scorer=proto.serialized_scorer,
            creation_time=proto.creation_time,
        )

    def to_proto(self):
        """Convert this ScorerVersion to a protobuf message."""
        proto = ProtoScorer()
        proto.experiment_id = self.experiment_id
        proto.scorer_name = self.scorer_name
        proto.scorer_version = self.scorer_version
        proto.serialized_scorer = self._serialized_scorer
        proto.creation_time = self.creation_time
        return proto

    def __repr__(self):
        return (
            f"<ScorerVersion(experiment_id={self.experiment_id}, "
            f"scorer_name='{self.scorer_name}', "
            f"scorer_version={self.scorer_version})>"
        )
