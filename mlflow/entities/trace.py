from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo


@dataclass
class Trace(_MlflowObject):
    """A trace object. (TODO: Add conceptual guide for tracing.)

    Args:
        info: A lightweight object that contains the metadata of a trace.
        data: A container object that holds the spans data of a trace.
    """

    info: TraceInfo
    data: TraceData

    def to_json(self) -> str:
        from mlflow.tracing.utils import TraceJSONEncoder

        return json.dumps(
            {"info": asdict(self.info), "data": self.data.to_dict()}, cls=TraceJSONEncoder
        )

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
