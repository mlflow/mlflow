from mlflow.entities._mlflow_object import _MlflowObject


class ScorerPresetVersion(_MlflowObject):
    """
    A versioned scorer preset entity representing a specific version of a preset
    within an MLflow experiment. Each version is an immutable snapshot of
    (scorer_id, scorer_version) pairs, identified by a hash digest.
    """

    def __init__(
        self,
        experiment_id: str,
        preset_name: str,
        version: str,
        scorer_refs: list[tuple[str, int]],
        creation_time: int,
        preset_id: str | None = None,
        serialized_scorers: list[str] | None = None,
    ):
        self._experiment_id = experiment_id
        self._preset_name = preset_name
        self._version = version
        self._scorer_refs = scorer_refs
        self._creation_time = creation_time
        self._preset_id = preset_id
        self._serialized_scorers = serialized_scorers or []

    @property
    def experiment_id(self):
        return self._experiment_id

    @property
    def preset_name(self):
        return self._preset_name

    @property
    def version(self):
        return self._version

    @property
    def scorer_refs(self) -> list[tuple[str, int]]:
        return self._scorer_refs

    @property
    def creation_time(self):
        return self._creation_time

    @property
    def preset_id(self):
        return self._preset_id

    @property
    def serialized_scorers(self) -> list[str]:
        return self._serialized_scorers

    @classmethod
    def from_proto(cls, proto):
        return cls(
            experiment_id=str(proto.experiment_id),
            preset_name=proto.preset_name,
            version=proto.version,
            scorer_refs=[(ref.scorer_id, ref.scorer_version) for ref in proto.scorer_refs],
            creation_time=proto.creation_time,
            preset_id=proto.preset_id if proto.HasField("preset_id") else None,
            serialized_scorers=list(proto.serialized_scorers) if proto.serialized_scorers else [],
        )

    def to_proto(self):
        from mlflow.protos.service_pb2 import ScorerPreset as ProtoScorerPreset
        from mlflow.protos.service_pb2 import ScorerPresetRef as ProtoScorerPresetRef

        proto = ProtoScorerPreset()
        proto.experiment_id = int(self.experiment_id)
        proto.preset_name = self.preset_name
        proto.version = self.version
        for scorer_id, scorer_version in self.scorer_refs:
            ref = ProtoScorerPresetRef()
            ref.scorer_id = scorer_id
            ref.scorer_version = scorer_version
            proto.scorer_refs.append(ref)
        proto.creation_time = self.creation_time
        if self.preset_id is not None:
            proto.preset_id = self.preset_id
        for s in self._serialized_scorers:
            proto.serialized_scorers.append(s)
        return proto

    def __repr__(self):
        return (
            f"<ScorerPresetVersion(experiment_id={self.experiment_id}, "
            f"preset_name='{self.preset_name}', version='{self.version}')>"
        )
