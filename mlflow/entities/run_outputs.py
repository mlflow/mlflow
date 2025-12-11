from typing import Any

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.logged_model_output import LoggedModelOutput
from mlflow.protos.service_pb2 import RunOutputs as ProtoRunOutputs


class RunOutputs(_MlflowObject):
    """RunOutputs object."""

    def __init__(self, model_outputs: list[LoggedModelOutput]) -> None:
        self._model_outputs = model_outputs

    def __eq__(self, other: _MlflowObject) -> bool:
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def model_outputs(self) -> list[LoggedModelOutput]:
        """Array of model outputs."""
        return self._model_outputs

    def to_proto(self):
        run_outputs = ProtoRunOutputs()
        run_outputs.model_outputs.extend(
            [model_output.to_proto() for model_output in self.model_outputs]
        )

        return run_outputs

    def to_dictionary(self) -> dict[Any, Any]:
        return {
            "model_outputs": [model_output.to_dictionary() for model_output in self.model_outputs],
        }

    @classmethod
    def from_proto(cls, proto):
        model_outputs = [
            LoggedModelOutput.from_proto(model_output) for model_output in proto.model_outputs
        ]

        return cls(model_outputs)
