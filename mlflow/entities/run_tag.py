from mlflow.entities.base_tag import BaseTag
from mlflow.protos.service_pb2 import RunTag as ProtoRunTag


class RunTag(BaseTag):
    """Tag object associated with a run."""

    def to_proto(self):
        param = ProtoRunTag()
        param.key = self.key
        param.value = self.value
        return param
