from mlflow.entities._mlflow_object import _MlflowObject


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
