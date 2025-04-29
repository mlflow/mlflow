from typing import Any, Optional

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.logged_model_input import LoggedModelInput
from mlflow.protos.service_pb2 import RunInputs as ProtoRunInputs


class RunInputs(_MlflowObject):
    """RunInputs object."""

    def __init__(
        self,
        dataset_inputs: list[DatasetInput],
        model_inputs: Optional[list[LoggedModelInput]] = None,
    ) -> None:
        self._dataset_inputs = dataset_inputs
        self._model_inputs = model_inputs or []

    def __eq__(self, other: _MlflowObject) -> bool:
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def dataset_inputs(self) -> list[DatasetInput]:
        """Array of dataset inputs."""
        return self._dataset_inputs

    @property
    def model_inputs(self) -> list[LoggedModelInput]:
        """Array of model inputs."""
        return self._model_inputs

    def to_proto(self):
        run_inputs = ProtoRunInputs()
        run_inputs.dataset_inputs.extend(
            [dataset_input.to_proto() for dataset_input in self.dataset_inputs]
        )
        run_inputs.model_inputs.extend(
            [model_input.to_proto() for model_input in self.model_inputs]
        )
        return run_inputs

    def to_dictionary(self) -> dict[str, Any]:
        return {
            "model_inputs": self.model_inputs,
            "dataset_inputs": [d.to_dictionary() for d in self.dataset_inputs],
        }

    @classmethod
    def from_proto(cls, proto):
        dataset_inputs = [
            DatasetInput.from_proto(dataset_input) for dataset_input in proto.dataset_inputs
        ]
        model_inputs = [
            LoggedModelInput.from_proto(model_input) for model_input in proto.model_inputs
        ]
        return cls(dataset_inputs, model_inputs)
