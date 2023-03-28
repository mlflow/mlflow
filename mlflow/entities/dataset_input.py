from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.protos.service_pb2 import DatasetInput as ProtoDatasetInput


class DatasetInput(_MLflowObject):
    """DatasetInput object associated with an experiment."""

    def __init__(self, tags, dataset):
        self._tags = (tags,)
        self._dataset = dataset

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def tags(self):
        """Array of input tags."""
        return self._tags

    @property
    def dataset(self):
        """Dataset."""
        return self._dataset

    def to_proto(self):
        dataset_input = ProtoDatasetInput()
        dataset_input.tags = self.tags
        dataset_input.dataset = self.dataset
        return dataset_input

    @classmethod
    def from_proto(cls, proto):
        return cls(proto.tags, proto.dataset)
