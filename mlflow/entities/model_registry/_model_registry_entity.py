from abc import abstractmethod

from mlflow.entities._mlflow_object import _MlflowObject


class _ModelRegistryEntity(_MlflowObject):
    @classmethod
    @abstractmethod
    def from_proto(cls, proto):
        pass

    def __eq__(self, other):
        return dict(self) == dict(other)
