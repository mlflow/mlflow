from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from packaging.version import Version

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
        return json.dumps(asdict(self), cls=_TraceJSONEncoder)

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


class _TraceJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for serializing Trace objects.

    Trace may contain types that require custom serialization logic, such as Pydantic models,
    non-JSON-serializable types, etc.
    """

    def default(self, obj):
        try:
            import pydantic

            if isinstance(obj, pydantic.BaseModel):
                # NB: Pydantic 2.0+ has a different API for model serialization
                if Version(pydantic.VERSION) >= Version("2.0"):
                    return obj.model_dump()
                else:
                    return obj.dict()
        except ImportError:
            pass

        try:
            return super().default(obj)
        except TypeError:
            return str(obj)
