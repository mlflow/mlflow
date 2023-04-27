from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.protos.service_pb2 import Dataset as ProtoDataset
from mlflow.utils.annotations import experimental

from typing import Optional


@experimental
class Dataset(_MLflowObject):
    """Dataset object associated with an experiment."""

    def __init__(
        self,
        name: str,
        digest: str,
        source_type: str,
        source: str,
        schema: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> None:
        self._name = name
        self._digest = digest
        self._source_type = source_type
        self._source = source
        self._schema = schema
        self._profile = profile

    def __eq__(self, other: _MLflowObject) -> bool:
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
    def schema(self) -> str:
        """String schema of the dataset."""
        return self._schema

    @property
    def profile(self) -> str:
        """String profile of the dataset."""
        return self._profile

    def to_proto(self):
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
    def from_proto(cls, proto):
        return cls(
            proto.name, proto.digest, proto.source_type, proto.source, proto.schema, proto.profile
        )
