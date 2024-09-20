from typing import Any, Dict, List

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.model_output import ModelOutput


class RunOutputs(_MlflowObject):
    """RunOutputs object."""

    def __init__(self, model_outputs: List[ModelOutput]) -> None:
        self._model_outputs = model_outputs

    def __eq__(self, other: _MlflowObject) -> bool:
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def model_outputs(self) -> List[ModelOutput]:
        """Array of model outputs."""
        return self._model_outputs

    def to_dictionary(self) -> Dict[Any, Any]:
        return {
            "model_outputs": self.model_outputs,
        }
