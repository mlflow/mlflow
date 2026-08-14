from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.service_pb2 import ModelOutput


class LoggedModelOutput(_MlflowObject):
    """ModelOutput object associated with a Run."""

    def __init__(self, model_id: str, step: int) -> None:
        self._model_id = model_id
        self._step = step

    def __eq__(self, other: _MlflowObject) -> bool:
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def model_id(self) -> str:
        """Model ID"""
        return self._model_id

    @property
    def step(self) -> str:
        """Step at which the model was logged"""
        return self._step

    def to_proto(self):
        return ModelOutput(model_id=self.model_id, step=self.step)

    def to_dictionary(self) -> dict[str, str | int]:
        return {"model_id": self.model_id, "step": self.step}

    @classmethod
    def from_proto(cls, proto):
        return cls(proto.model_id, proto.step)
