from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.entities.run_data import RunData
from mlflow.entities.run_info import RunInfo
from mlflow.protos.service_pb2 import Run as ProtoRun


class Run(_MLflowObject):
    """
    Run object.
    """

    def __init__(self, run_info, run_data):
        if run_info is None:
            raise Exception("run_info cannot be None")
        self._info = run_info
        self._data = run_data

    @property
    def info(self):
        """
        The run metadata, such as the run id, start time, and status.

        :rtype: :py:class:`mlflow.entities.RunInfo`
        """
        return self._info

    @property
    def data(self):
        """
        The run data, including metrics, parameters, and tags.

        :rtype: :py:class:`mlflow.entities.RunData`
        """
        return self._data

    def to_proto(self):
        run = ProtoRun()
        run.info.MergeFrom(self.info.to_proto())
        if self.data:
            run.data.MergeFrom(self.data.to_proto())
        return run

    @classmethod
    def from_proto(cls, proto):
        return cls(RunInfo.from_proto(proto.info), RunData.from_proto(proto.data))

    @classmethod
    def from_dictionary(cls, the_dict):
        if "info" not in the_dict or "data" not in the_dict:
            raise Exception("Malformed input '%s'. Run cannot be constructed." % str(the_dict))
        the_info = RunInfo.from_dictionary(the_dict.get("info"))
        the_data = RunData.from_dictionary(the_dict.get("data"))
        return cls(the_info, the_data)

    def to_dictionary(self):
        return {"info": dict(self.info), "data": self.data.to_dictionary()}
