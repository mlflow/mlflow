from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.protos.service_pb2 import (
    SignedUrl as ProtoSignedUrl
)


class SignedUri(_MLflowObject):

    def __init__(self, name, uri):
        super().__init__()
        self._name = name
        self._uri = uri

    @property
    def name(self):
        return self._name

    def _set_name(self, new_name):
        self._name = new_name

    @property
    def uri(self):
        return self._uri

    def _set_uri(self, uri):
        self._uri = uri

    @classmethod
    def from_proto(cls, proto):
        experiment = cls(
            proto.name, proto.uri
        )
        return experiment

    def to_proto(self):
        experiment = ProtoSignedUrl()
        experiment.name = self.name
        experiment.uri = self.uri
        return experiment
