from typing import Any, Dict, Optional

from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.entities.run_data import RunData
from mlflow.entities.run_info import RunInfo
from mlflow.entities.run_inputs import RunInputs
from mlflow.exceptions import MlflowException
from mlflow.protos.service_pb2 import Run as ProtoRun


class Run(_MLflowObject):
    """
    Run object.
    """

    def __init__(
        self, run_info: RunInfo, run_data: RunData, run_inputs: Optional[RunInputs] = None
    ) -> None:
        if run_info is None:
            raise MlflowException("run_info cannot be None")
        self._info = run_info
        self._data = run_data
        self._inputs = run_inputs

    @property
    def info(self) -> RunInfo:
        """
        The run metadata, such as the run id, start time, and status.

        :rtype: :py:class:`mlflow.entities.RunInfo`
        """
        return self._info

    @property
    def data(self) -> RunData:
        """
        The run data, including metrics, parameters, and tags.

        :rtype: :py:class:`mlflow.entities.RunData`
        """
        return self._data

    @property
    def inputs(self) -> RunInputs:
        """
        The run inputs, including dataset inputs

        :rtype: :py:class:`mlflow.entities.RunData`
        """
        return self._inputs

    @property
    def system_metrics_monitor(self):
        return getattr(self, "_system_metrics_monitor", None)

    @system_metrics_monitor.setter
    def system_metrics_monitor(self, system_metrics_monitor):
        self._system_metrics_monitor = system_metrics_monitor

    def to_proto(self):
        run = ProtoRun()
        run.info.MergeFrom(self.info.to_proto())
        if self.data:
            run.data.MergeFrom(self.data.to_proto())
        if self.inputs:
            run.inputs.MergeFrom(self.inputs.to_proto())
        return run

    @classmethod
    def from_proto(cls, proto):
        return cls(
            RunInfo.from_proto(proto.info),
            RunData.from_proto(proto.data),
            RunInputs.from_proto(proto.inputs),
        )

    def to_dictionary(self) -> Dict[Any, Any]:
        run_dict = {
            "info": dict(self.info),
        }
        if self.data:
            run_dict["data"] = self.data.to_dictionary()
        if self.inputs:
            run_dict["inputs"] = self.inputs.to_dictionary()
        return run_dict
