from mlflow.entities.base_tag import BaseTag
from mlflow.protos.service_pb2 import InputTag as ProtoInputTag
from mlflow.utils.annotations import experimental


@experimental
class InputTag(BaseTag):
    """Tag object associated with an experiment."""

    def to_proto(self):
        param = ProtoInputTag()
        param.key = self.key
        param.value = self.value
        return param
