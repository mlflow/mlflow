from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.entities.run_data import RunData
from mlflow.entities.run_info import RunInfo
from mlflow.protos.service_pb2 import Run as ProtoRun


class Run(_MLflowObject):
    """
    Run object for python client. Backend stores will hydrate this object in APIs.
    """

    def __init__(self, run_info, run_data):
        if run_info is None:
            raise Exception("run_info cannot be None")
        self._info = run_info
        self._data = run_data

    @property
    def info(self):
        return self._info

    @property
    def data(self):
        return self._data

    def to_proto(self):
        run = ProtoRun()
        run.info.MergeFrom(self.info.to_proto())
        if self.data:
            run.data.MergeFrom(self.data.to_proto())
        return run

    @classmethod
    def from_proto(cls, proto):
        return cls(proto.info, proto.data)

    @classmethod
    def from_dictionary(cls, the_dict):
        if "info" not in the_dict or "data" not in the_dict:
            raise Exception("Malformed input '%s'. Run cannot be constructed." % str(the_dict))
        the_info = RunInfo.from_dictionary(the_dict.get("info"))
        the_data = RunData.from_dictionary(the_dict.get("data"))
        return cls(the_info, the_data)

    def to_dictionary(self):
        return {"info": dict(self.info), "data": dict(self.data)}

    def __iter__(self):
        the_dict = self.to_dictionary()
        for k in the_dict:
            yield k, the_dict[k]

    @classmethod
    def _properties(cls):
        # This method should never get called since __iter__ from base class has been overridden.
        raise NotImplementedError
