from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.service_pb2 import ModelOutput as ProtoModelOutput


class LoggedModelOutput(_MlflowObject):
    """ModelOutput object associated with a Run."""

    def __init__(self, model_id: str, step: int) -> None:
        self._model_id = model_id
        self._step = step

    def __eq__(self, other: object) -> bool:
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def model_id(self) -> str:
        """Model ID"""
        return self._model_id

    @property
    def step(self) -> int:
        """Step at which the model was logged"""
        return self._step

    def to_proto(self) -> ProtoModelOutput:
        return ProtoModelOutput(model_id=self.model_id, step=self.step)

    def to_dictionary(self) -> dict[str, str | int]:
        return {"model_id": self.model_id, "step": self.step}

    @classmethod
    def from_proto(cls, proto: ProtoModelOutput) -> "LoggedModelOutput":
        return cls(proto.model_id, proto.step)
