from typing import Any

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.service_pb2 import Dataset as ProtoDataset


class Dataset(_MlflowObject):
    """Dataset object associated with an experiment."""

    def __init__(
        self,
        name: str,
        digest: str,
        source_type: str,
        source: str,
        schema: str | None = None,
        profile: str | None = None,
    ) -> None:
        self._name = name
        self._digest = digest
        self._source_type = source_type
        self._source = source
        self._schema = schema
        self._profile = profile

    def __eq__(self, other: object) -> bool:
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def name(self) -> str:
        """String name of the dataset."""
        return self._name

    @property
    def digest(self) -> str:
        """String digest of the dataset."""
        return self._digest

    @property
    def source_type(self) -> str:
        """String source_type of the dataset."""
        return self._source_type

    @property
    def source(self) -> str:
        """String source of the dataset."""
        return self._source

    @property
    def schema(self) -> str | None:
        """String schema of the dataset."""
        return self._schema

    @property
    def profile(self) -> str | None:
        """String profile of the dataset."""
        return self._profile

    def to_proto(self) -> ProtoDataset:
        dataset = ProtoDataset()
        dataset.name = self.name
        dataset.digest = self.digest
        dataset.source_type = self.source_type
        dataset.source = self.source
        if self.schema:
            dataset.schema = self.schema
        if self.profile:
            dataset.profile = self.profile
        return dataset

    @classmethod
    def from_proto(cls, proto: ProtoDataset) -> "Dataset":
        return cls(
            proto.name,
            proto.digest,
            proto.source_type,
            proto.source,
            proto.schema if proto.HasField("schema") else None,
            proto.profile if proto.HasField("profile") else None,
        )

    def to_dictionary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "digest": self.digest,
            "source_type": self.source_type,
            "source": self.source,
            "schema": self.schema,
            "profile": self.profile,
        }
