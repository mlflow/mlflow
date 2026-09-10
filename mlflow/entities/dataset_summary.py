from mlflow.protos.service_pb2 import DatasetSummary


class _DatasetSummary:
    """
    DatasetSummary object.

    This is used to return a list of dataset summaries across one or more experiments in the UI.
    """

    def __init__(self, experiment_id: str, name: str, digest: str, context: str | None) -> None:
        self._experiment_id = experiment_id
        self._name = name
        self._digest = digest
        self._context = context

    def __eq__(self, other: object) -> bool:
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def context(self) -> str | None:
        return self._context

    def to_dict(self) -> dict[str, str | None]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "digest": self.digest,
            "context": self.context,
        }

    def to_proto(self) -> DatasetSummary:
        dataset_summary = DatasetSummary()
        dataset_summary.experiment_id = self.experiment_id
        dataset_summary.name = self.name
        dataset_summary.digest = self.digest
        if self.context:
            dataset_summary.context = self.context
        return dataset_summary

    @classmethod
    def from_proto(cls, proto: DatasetSummary) -> "_DatasetSummary":
        return cls(
            experiment_id=proto.experiment_id,
            name=proto.name,
            digest=proto.digest,
            context=proto.context,
        )
