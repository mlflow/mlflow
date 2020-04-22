import json
import numpy as np
from typing import Dict, Any, List

from mlflow.exceptions import MlflowException
from mlflow.types import DataType


class ColSpec(object):
    """
    Specification of a column used in model signature.
    Declares data type and optionally a name.
    """

    def __init__(self, type: DataType, name: str = None):  # pylint: disable=redefined-builtin
        self.name = name
        try:
            self.type = DataType[type] if isinstance(type, str) else type
        except KeyError:
            raise MlflowException("Unsupported type '{0}', expected instance of DataType or "
                                  "one of {1}".format(type, [t.name for t in DataType]))
        if not isinstance(self.type, DataType):
            raise TypeError("Expected mlflow.models.signature.Datatype or str for the 'type' "
                            "argument, but got {}".format(self.type.__class__))

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize into a jsonable dictionary.
        :return: dictionary representation of the column spec.
        """
        return {"name": self.name, "type": self.type.name}

    def __eq__(self, other) -> bool:
        names_eq = (self.name is None and other.name is None) or self.name == other.name
        return names_eq and self.type == other.type

    def __repr__(self) -> str:
        return "{name}: {type}".format(name=self.name, type=self.type)


class Schema(object):
    """
    Schema specifies column types (:py:class:`DataType`) in a dataset.

    Schema is a list of column specification :py:class:`ColSpec`. Columns can be named and must
    specify their data type. The list of supported types is defined in :py:class:`DataType` enum.
    """

    def __init__(self, cols: List[ColSpec]):
        self._cols = cols

    @property
    def columns(self) -> List[ColSpec]:
        return self._cols

    def column_names(self) -> List[str]:
        return [x.name or i for i, x in enumerate(self.columns)]

    def column_types(self) -> List[DataType]:
        return [x.type for x in self._cols]

    def numpy_types(self) -> List[np.dtype]:
        return [x.type.to_numpy() for x in self.columns]

    def to_json(self) -> str:
        return json.dumps([x.to_dict() for x in self.columns])

    @classmethod
    def from_json(cls, json_str: str):
        return cls([ColSpec(**x) for x in json.loads(json_str)])

    def __eq__(self, other) -> bool:
        if isinstance(other, Schema):
            return self.columns == other.columns
        else:
            return False

    def __repr__(self) -> str:
        return repr(self.columns)

