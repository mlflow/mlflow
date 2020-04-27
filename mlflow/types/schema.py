import json
from enum import Enum

import numpy as np
from typing import Dict, Any, List, Union, Optional

from mlflow.exceptions import MlflowException


class DataType(Enum):
    """
    MLflow data types.
    """
    def __new__(cls, value, numpy_type):
        res = object.__new__(cls)
        res._value_ = value
        res._numpy_type = numpy_type
        return res

    boolean = (1, np.bool)
    """Logical data (True, False) ."""
    integer = (2, np.int32)
    """32b signed integer numbers."""
    long = (3, np.int64)
    """64b signed integer numbers. """
    float = (4, np.float32)
    """32b floating point numbers. """
    double = (5, np.float64)
    """64b floating point numbers. """
    string = (6, np.str)
    """Text data."""
    binary = (7, np.bytes_)
    """Sequence of raw bytes."""

    def __repr__(self):
        return self.name

    def __str(self):
        return self.name

    def to_numpy(self) -> np.dtype:
        """Get equivalent numpy data type. """
        return self._numpy_type


class ColSpec(object):
    """
    Specification of name and type of a single column in a dataset.
    """
    def __init__(self, type: DataType,  # pylint: disable=redefined-builtin
                 name: Optional[str] = None):
        self._name = name
        try:
            self._type = DataType[type] if isinstance(type, str) else type
        except KeyError:
            raise MlflowException("Unsupported type '{0}', expected instance of DataType or "
                                  "one of {1}".format(type, [t.name for t in DataType]))
        if not isinstance(self.type, DataType):
            raise TypeError("Expected mlflow.models.signature.Datatype or str for the 'type' "
                            "argument, but got {}".format(self.type.__class__))

    @property
    def type(self) -> DataType:
        """The column data type."""
        return self._type

    @property
    def name(self) -> Optional[str]:
        """The column name or None if the columns is unnamed."""
        return self._name

    def to_dict(self) -> Dict[str, Any]:
        if self.name is None:
            return {"type": self.type.name}
        else:
            return {"name": self.name, "type": self.type.name}

    def __eq__(self, other) -> bool:
        names_eq = (self.name is None and other.name is None) or self.name == other.name
        return names_eq and self.type == other.type

    def __repr__(self) -> str:
        if self.name is None:
            return repr(self.type)
        else:
            return "{name}: {type}".format(name=repr(self.name), type=repr(self.type))


class Schema(object):
    """
    Specification of types and column names in a dataset.

    Schema is represented as a list of :py:class:`ColSpec`. The columns in a schema can be named,
    with unique non empty name for every column, or unnamed with implicit integer index defined by
    their list indices. Combination of named and unnamed columns is not allowed.
    """
    def __init__(self, cols: List[ColSpec]):
        if not (all(map(lambda x: x.name is None, cols))
                or all(map(lambda x: x.name is not None, cols))):
            raise MlflowException("Creating Schema with a combination of named and unnamed columns "
                                  "is not allowed. Got column names {}".format(
                                    [x.name for x in cols]))
        self._cols = cols

    @property
    def columns(self) -> List[ColSpec]:
        """The list of columns that defines this schema."""
        return self._cols

    def column_names(self) -> List[Union[str, int]]:
        """Get list of column names or range of indices if the schema has no column names."""
        return [x.name or i for i, x in enumerate(self.columns)]

    def column_types(self) -> List[DataType]:
        """ Get column types of the columns in the dataset."""
        return [x.type for x in self._cols]

    def numpy_types(self) -> List[np.dtype]:
        """ Convenience shortcut to get the datatypes as numpy types."""
        return [x.type.to_numpy() for x in self.columns]

    def to_json(self) -> str:
        """Serialize into json string."""
        return json.dumps([x.to_dict() for x in self.columns])

    def to_dict(self) -> List[Dict[str, Any]]:
        """Serialize into a jsonable dictionary."""
        return [x.to_dict() for x in self.columns]

    @classmethod
    def from_json(cls, json_str: str):
        """ Deserialize from a json string."""
        return cls([ColSpec(**x) for x in json.loads(json_str)])

    def __eq__(self, other) -> bool:
        if isinstance(other, Schema):
            return self.columns == other.columns
        else:
            return False

    def __repr__(self) -> str:
        return repr(self.columns)
