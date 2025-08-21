import json
from functools import lru_cache

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.service_pb2 import Scorer as ProtoScorer


class ScorerVersion(_MlflowObject):
    """
    A versioned scorer entity that represents a specific version of a scorer within an MLflow
    experiment.

    Each ScorerVersion instance is uniquely identified by the combination of:
    - experiment_id: The experiment containing the scorer
    - scorer_name: The name of the scorer
    - scorer_version: The version number of the scorer

    The class provides access to both the metadata (name, version, creation time) and the actual
    scorer implementation through the serialized_scorer property, which deserializes the stored
    scorer data into a usable SerializedScorer object.

    Args:
        experiment_id (str): The ID of the experiment this scorer belongs to.
        scorer_name (str): The name identifier for the scorer.
        scorer_version (int): The version number of this scorer instance.
        serialized_scorer (str): JSON-serialized string containing the scorer's metadata and code.
        creation_time (int): Unix timestamp (in milliseconds) when this version was created.

    Example:
        .. code-block:: python

            from mlflow.entities.scorer import ScorerVersion

            # Create a ScorerVersion instance
            scorer_version = ScorerVersion(
                experiment_id="123",
                scorer_name="accuracy_scorer",
                scorer_version=2,
                serialized_scorer='{"name": "accuracy_scorer", "call_source": "..."}',
                creation_time=1640995200000,
            )

            # Access scorer metadata
            print(f"Scorer: {scorer_version.scorer_name} v{scorer_version.scorer_version}")
            print(f"Created: {scorer_version.creation_time}")
    """

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
        """
        The ID of the experiment this scorer belongs to.

        Returns:
            str: The id of the experiment that this scorer version belongs to.
        """
        return self._experiment_id

    @property
    def scorer_name(self):
        """
        The name identifier for the scorer.

        Returns:
            str: The human-readable name used to identify and reference this scorer.
        """
        return self._scorer_name

    @property
    def scorer_version(self):
        """
        The version number of this scorer instance.

        Returns:
            int: The sequential version number, starting from 1. Higher versions represent
                 newer saved scorers with the same name.
        """
        return self._scorer_version

    @property
    @lru_cache(maxsize=1)
    def serialized_scorer(self):
        """
        The deserialized scorer object containing metadata and function code.

        This property automatically deserializes the stored JSON string into a
        SerializedScorer object that contains all the information needed to
        reconstruct and execute the scorer function.

        The result is cached using LRU caching to avoid repeated deserialization
        when the same ScorerVersion instance is accessed multiple times.

        Returns:
            SerializedScorer: A `SerializedScorer` object with metadata, function code,
                              and configuration information.

        Note:
            The `SerializedScorer` object construction is lazy,
            it only happens when this property is first accessed.
        """
        from mlflow.genai.scorers.base import SerializedScorer

        return SerializedScorer(**json.loads(self._serialized_scorer))

    @property
    def creation_time(self):
        """
        The timestamp when this scorer version was created.

        Returns:
            int: Unix timestamp in milliseconds representing when this specific
                 version of the scorer was registered in MLflow.
        """
        return self._creation_time

    @classmethod
    def from_proto(cls, proto):
        """
        Create a ScorerVersion instance from a protobuf message.

        This class method is used internally by MLflow to reconstruct ScorerVersion
        objects from serialized protobuf data, typically when retrieving scorers
        from remote tracking servers or deserializing stored data.

        Args:
            proto: A protobuf message containing scorer version data.

        Returns:
            ScorerVersion: A new ScorerVersion instance populated with data from the protobuf.

        Note:
            This method is primarily used internally by MLflow's tracking infrastructure
            and should not typically be called directly by users.
        """
        return cls(
            experiment_id=proto.experiment_id,
            scorer_name=proto.scorer_name,
            scorer_version=proto.scorer_version,
            serialized_scorer=proto.serialized_scorer,
            creation_time=proto.creation_time,
        )

    def to_proto(self):
        """
        Convert this ScorerVersion instance to a protobuf message.

        This method serializes the ScorerVersion data into a protobuf format
        for transmission over the network or storage in binary format. It's
        primarily used internally by MLflow's tracking infrastructure.

        Returns:
            ProtoScorer: A protobuf message containing the serialized scorer version data.

        Note:
            This method is primarily used internally by MLflow's tracking infrastructure
            and should not typically be called directly by users.
        """
        proto = ProtoScorer()
        proto.experiment_id = self.experiment_id
        proto.scorer_name = self.scorer_name
        proto.scorer_version = self.scorer_version
        proto.serialized_scorer = self._serialized_scorer
        proto.creation_time = self.creation_time
        return proto

    def __repr__(self):
        """
        Return a string representation of the ScorerVersion instance.

        Returns:
            str: A human-readable string showing the key identifying information
                 of this scorer version (experiment_id, scorer_name, and scorer_version).
        """
        return (
            f"<ScorerVersion(experiment_id={self.experiment_id}, "
            f"scorer_name='{self.scorer_name}', "
            f"scorer_version={self.scorer_version})>"
        )
