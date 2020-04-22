"""
The :py:mod:`mlflow.types` module defines mlflow data types and provides utilities such as type
inference from python types.

The types defined here can be used by other mlflow components to describe interface independent of
other frameworks or languages.  The data types can be organized in Schema that declares a sequence
of optionally named typed columns.
"""

from enum import Enum
import numpy as np


class DataType(Enum):
    """
    MLflow element data types definition.
    """

    def __new__(cls, value, numpy_type):
        res = object.__new__(cls)
        res._value_ = value
        res._numpy_type = numpy_type
        return res

    boolean = (1, np.bool)
    integer = (2, np.int32)
    long = (3, np.int64)
    float = (4, np.float32)
    double = (5, np.float64)
    string = (6, np.str)
    binary = (7, np.bytes_)

    def __repr__(self):
        return self.name

    def __str(self):
        return self.name

    def to_numpy(self) -> np.dtype:
        return self._numpy_type
