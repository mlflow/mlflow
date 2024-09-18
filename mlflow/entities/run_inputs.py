from typing import Any, Dict, List

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.model_input import ModelInput
from mlflow.protos.service_pb2 import RunInputs as ProtoRunInputs


class RunInputs(_MlflowObject):
    """RunInputs object."""

    def __init__(self, dataset_inputs: List[DatasetInput], model_inputs: List[ModelInput]) -> None:
        self._dataset_inputs = dataset_inputs
        self._model_inputs = model_inputs

    def __eq__(self, other: _MlflowObject) -> bool:
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def dataset_inputs(self) -> List[DatasetInput]:
        """Array of dataset inputs."""
        return self._dataset_inputs

    @property
    def model_inputs(self) -> List[ModelInput]:
        """Array of model inputs."""
        return self._model_inputs

    def to_proto(self):
        run_inputs = ProtoRunInputs()
        run_inputs.dataset_inputs.extend(
            [dataset_input.to_proto() for dataset_input in self.dataset_inputs]
        )
        # TODO: Support proto conversion for model inputs
        # run_inputs.model_inputs.extend(
        #     [model_input.to_proto() for model_input in self.model_inputs]
        # )
        return run_inputs

    def to_dictionary(self) -> Dict[Any, Any]:
        return {
            "dataset_inputs": self.dataset_inputs,
            "model_inputs": self.model_inputs,
        }

    @classmethod
    def from_proto(cls, proto):
        dataset_inputs = [
            DatasetInput.from_proto(dataset_input) for dataset_input in proto.dataset_inputs
        ]
        # TODO: Support proto conversion for model inputs
        # model_inputs = [
        #     ModelInput.from_proto(model_input) for model_input in proto.model_inputs
        # ]
        return cls(dataset_inputs, [])
