from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.protos.service_pb2 import RunInputs as ProtoRunInputs
from mlflow.entities.dataset_input import DatasetInput

from typing import List, Dict, Any


class RunInputs(_MLflowObject):
    """RunInputs object."""

    def __init__(self, dataset_inputs: List[DatasetInput]) -> None:
        self._dataset_inputs = dataset_inputs

    def __eq__(self, other: _MLflowObject) -> bool:
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def dataset_inputs(self) -> List[DatasetInput]:
        """Array of dataset inputs."""
        return self._dataset_inputs

    def to_proto(self):
        run_inputs = ProtoRunInputs()
        run_inputs.dataset_inputs.extend(
            [dataset_input.to_proto() for dataset_input in self.dataset_inputs]
        )
        return run_inputs

    def to_dictionary(self) -> Dict[Any, Any]:
        return {
            "dataset_inputs": self.dataset_inputs,
        }

    @classmethod
    def from_proto(cls, proto):
        dataset_inputs = [
            DatasetInput.from_proto(dataset_input) for dataset_input in proto.dataset_inputs
        ]
        return cls(dataset_inputs)
