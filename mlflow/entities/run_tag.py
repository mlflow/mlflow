from mlflow.entities.base_tag import BaseTag
from mlflow.protos.service_pb2 import RunTag as ProtoRunTag


class RunTag(BaseTag):
    """Tag object associated with a run."""

    def to_proto(self):
        param = ProtoRunTag()
        param.key = self.key
        param.value = self.value
        return param

    @property
    def key(self):
        """
        String name of the tag.
        To be compatible with _MLflowObject._get_properties_helper
        """
        return self._key

    @property
    def value(self):
        """
        String value of the tag.
        To be compatible with _MLflowObject._get_properties_helper
        """
        return self._value
