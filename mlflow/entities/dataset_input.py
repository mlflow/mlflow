from typing import List, Optional

from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.entities.dataset import Dataset
from mlflow.entities.input_tag import InputTag
from mlflow.protos.service_pb2 import DatasetInput as ProtoDatasetInput
from mlflow.utils.annotations import experimental


@experimental
class DatasetInput(_MLflowObject):
    """DatasetInput object associated with an experiment."""

    def __init__(self, dataset: Dataset, tags: Optional[List[InputTag]] = None) -> None:
        self._dataset = dataset
        self._tags = tags or []

    def __eq__(self, other: _MLflowObject) -> bool:
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    def _add_tag(self, tag: InputTag) -> None:
        self._tags.append(tag)

    @property
    def tags(self) -> List[InputTag]:
        """Array of input tags."""
        return self._tags

    @property
    def dataset(self) -> Dataset:
        """Dataset."""
        return self._dataset

    def to_proto(self):
        dataset_input = ProtoDatasetInput()
        dataset_input.tags.extend([tag.to_proto() for tag in self.tags])
        dataset_input.dataset.MergeFrom(self.dataset.to_proto())
        return dataset_input

    @classmethod
    def from_proto(cls, proto):
        dataset_input = cls(Dataset.from_proto(proto.dataset))
        for input_tag in proto.tags:
            dataset_input._add_tag(InputTag.from_proto(input_tag))
        return dataset_input
