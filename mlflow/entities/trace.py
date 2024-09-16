from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


@dataclass
class Trace(_MlflowObject):
    """A trace object.

    Args:
        info: A lightweight object that contains the metadata of a trace.
        data: A container object that holds the spans data of a trace.
    """

    info: TraceInfo
    data: TraceData

    def __repr__(self) -> str:
        return f"Trace(request_id={self.info.request_id})"

    def to_dict(self) -> Dict[str, Any]:
        return {"info": self.info.to_dict(), "data": self.data.to_dict()}

    def to_json(self, pretty=False) -> str:
        from mlflow.tracing.utils import TraceJSONEncoder

        return json.dumps(self.to_dict(), cls=TraceJSONEncoder, indent=2 if pretty else None)

    @classmethod
    def from_dict(cls, trace_dict: Dict[str, Any]) -> Trace:
        info = trace_dict.get("info")
        data = trace_dict.get("data")
        if info is None or data is None:
            raise MlflowException(
                "Unable to parse Trace from dictionary. Expected keys: 'info' and 'data'. "
                f"Received keys: {list(trace_dict.keys())}",
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

    def _serialize_for_mimebundle(self):
        # databricks notebooks will use the request ID to
        # fetch the trace from the backend. including the
        # full JSON can cause notebooks to exceed size limits
        return json.dumps(self.info.request_id)

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
            "application/databricks.mlflow.trace": self._serialize_for_mimebundle(),
            "text/plain": repr(self),
        }

    def to_pandas_dataframe_row(self) -> Dict[str, Any]:
        return {
            "request_id": self.info.request_id,
            "trace": self,
            "timestamp_ms": self.info.timestamp_ms,
            "status": self.info.status,
            "execution_time_ms": self.info.execution_time_ms,
            "request": self.data.request,
            "response": self.data.response,
            "request_metadata": self.info.request_metadata,
            "spans": [span.to_dict() for span in self.data.spans],
            "tags": self.info.tags,
        }

    @staticmethod
    def pandas_dataframe_columns() -> List[str]:
        return [
            "request_id",
            "trace",
            "timestamp_ms",
            "status",
            "execution_time_ms",
            "request",
            "response",
            "request_metadata",
            "spans",
            "tags",
        ]
