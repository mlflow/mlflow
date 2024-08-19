from mlflow.entities._mlflow_object import _MlflowObject


class ModelOutput(_MlflowObject):
    """ModelOutput object associated with a Run."""

    def __init__(self, model_id: str):
        self._model_id = model_id

    def __eq__(self, other: _MlflowObject) -> bool:
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def model_id(self) -> str:
        """Model ID"""
        return self._model_id
