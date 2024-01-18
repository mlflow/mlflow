from mlflow.entities.base_tag import BaseTag
from mlflow.protos.service_pb2 import ExperimentTag as ProtoExperimentTag


class ExperimentTag(BaseTag):
    """Tag object associated with an experiment."""

    def to_proto(self):
        param = ProtoExperimentTag()
        param.key = self.key
        param.value = self.value
        return param
