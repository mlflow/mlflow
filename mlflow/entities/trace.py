from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


@dataclass
class Trace(_MlflowObject):
    """A trace object. (TODO: Add conceptual guide for tracing.)

    Args:
        info: A lightweight object that contains the metadata of a trace.
        data: A container object that holds the spans data of a trace.

    :meta private:
    """

    info: TraceInfo
    data: TraceData

    def to_dict(self) -> Dict[str, Any]:
        return {"info": self.info.to_dict(), "data": self.data.to_dict()}

    def to_json(self) -> str:
        from mlflow.tracing.utils import TraceJSONEncoder

        return json.dumps(self.to_dict(), cls=TraceJSONEncoder)

    @classmethod
    def from_dict(cls, trace_dict: Dict[str, Any]) -> Trace:
        info = trace_dict.get("info")
        data = trace_dict.get("data")
        if info is None or data is None:
            raise MlflowException(
                "Unable to parse Trace from dictionary. Expected keys: 'info' and 'data'. "
                "Received keys: %s" % list(trace_dict.keys()),
                error_code=INVALID_PARAMETER_VALUE,
            )
        return cls(
            info=TraceInfo.from_dict(info),
            data=TraceData.from_dict(data),
        )

    @classmethod
    def from_json(cls, trace_json: str) -> Trace:
        try:
            trace_dict = json.loads(trace_json)
        except json.JSONDecodeError as e:
            raise MlflowException(
                f"Unable to parse trace JSON: {trace_json}. Error: {e}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return cls.from_dict(trace_dict)

    def _repr_mimebundle_(self, include=None, exclude=None):
        """
        This method is used to trigger custom display logic in IPython notebooks.
        See https://ipython.readthedocs.io/en/stable/config/integrating.html#MyObject
        for more details.

        At the moment, the only supported MIME type is "application/databricks.mlflow.trace",
        which contains a JSON representation of the Trace object. This object is deserialized
        in Databricks notebooks to display the Trace object in a nicer UI.
        """
        return {
            "application/databricks.mlflow.trace": self.to_json(),
            "text/plain": self.__repr__(),
        }

    def to_pandas_dataframe_row(self) -> Dict[str, Any]:
        return {
            "request_id": self.info.request_id,
            "timestamp_ms": self.info.timestamp_ms,
            "status": self.info.status,
            "execution_time_ms": self.info.execution_time_ms,
            "request": self.data.request,
            "response": self.data.response,
            "request_metadata": self.info.request_metadata,
            "spans": [span.to_dict() for span in self.data.spans],
            "tags": self.info.tags,
        }
