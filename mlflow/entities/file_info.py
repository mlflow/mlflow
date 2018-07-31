from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.protos.service_pb2 import FileInfo as ProtoFileInfo


class FileInfo(_MLflowObject):
    def __init__(self, path, is_dir, file_size):
        self._path = path
        self._is_dir = is_dir
        self._bytes = file_size

    @property
    def path(self):
        return self._path

    @property
    def is_dir(self):
        return self._is_dir

    @property
    def file_size(self):
        return self._bytes

    def to_proto(self):
        proto = ProtoFileInfo()
        proto.path = self.path
        proto.is_dir = self.is_dir
        if self.file_size:
            proto.file_size = self.file_size
        return proto

    @classmethod
    def from_proto(cls, proto):
        return cls(proto.path, proto.is_dir, proto.file_size)

    @classmethod
    def _properties(cls):
        # TODO: Hard coding this list of props for now. There has to be a clearer way...
        return ["path", "is_dir", "file_size"]
