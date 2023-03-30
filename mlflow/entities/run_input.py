from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.protos.service_pb2 import RunInputs as ProtoRunInputs

# from mlflow.protos.service_pb2 import DatasetInput as ProtoDatasetInput


class RunInput(_MLflowObject):
    """RunInput object."""

    def __init__(self, dataset_inputs):
        self._dataset_inputs = dataset_inputs

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def dataset_inputs(self):
        """Array of dataset inputs."""
        return self._dataset_inputs

    def to_proto(self):
        run_inputs = ProtoRunInputs()
        run_inputs.dataset_inputs.extend(
            [dataset_input.to_proto() for dataset_input in self.dataset_inputs]
        )
        return run_inputs

    def to_dictionary(self):
        return {
            "dataset_inputs": self.dataset_inputs,
        }

    @classmethod
    def from_proto(cls, proto):
        return cls(proto.dataset_inputs)
