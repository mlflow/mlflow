from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.service_pb2 import ModelInput as ProtoModelInput


class LoggedModelInput(_MlflowObject):
    """ModelInput object associated with a Run."""

    def __init__(self, model_id: str):
        self._model_id = model_id

    def __eq__(self, other: _MlflowObject) -> bool:
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def model_id(self) -> str:
        """Model ID."""
        return self._model_id

    def to_proto(self):
        return ProtoModelInput(model_id=self._model_id)

    @classmethod
    def from_proto(cls, proto):
        return cls(proto.model_id)
