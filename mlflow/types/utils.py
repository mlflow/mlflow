from typing import Any
import warnings

import numpy as np
import pandas as pd


from mlflow.exceptions import MlflowException
from mlflow.types import DataType
from mlflow.types.schema import Schema, ColSpec


class TensorsNotSupportedException(MlflowException):
    def __init__(self, msg):
        super().__init__(
            "Multidimensional arrays (aka tensors) are not supported. " "{}".format(msg)
        )


def _infer_schema(data: Any) -> Schema:
    """
    Infer an MLflow schema from a dataset.

    This method captures the column names and data types from the user data. The signature
    represents model input and output as data frames with (optionally) named columns and data
    type specified as one of types defined in :py:class:`DataType`. This method will raise
    an exception if the user data contains incompatible types or is not passed in one of the
    supported formats (containers).

    The input should be one of these:
      - pandas.DataFrame or pandas.Series
      - dictionary of { name -> numpy.ndarray}
      - numpy.ndarray
      - pyspark.sql.DataFrame

    The element types should be mappable to one of :py:class:`mlflow.models.signature.DataType`.

    NOTE: Multidimensional (>2d) arrays (aka tensors) are not supported at this time.

    :param data: Dataset to infer from.

    :return: Schema
    """

    if isinstance(data, dict):
        res = []
        for col in data.keys():
            ary = data[col]
            if not isinstance(ary, np.ndarray):
                raise TypeError("Data in the dictionary must be of type numpy.ndarray")
            dims = len(ary.shape)
            if dims == 1:
                res.append(ColSpec(type=_infer_numpy_array(ary), name=col))
            else:
                raise TensorsNotSupportedException(
                    "Data in the dictionary must be 1-dimensional, "
                    "got shape {}".format(ary.shape)
                )
        schema = Schema(res)
    elif isinstance(data, pd.Series):
        schema = Schema([ColSpec(type=_infer_numpy_array(data.values))])
    elif isinstance(data, pd.DataFrame):
        schema = Schema(
            [ColSpec(type=_infer_numpy_array(data[col].values), name=col) for col in data.columns]
        )
    elif isinstance(data, np.ndarray):
        if len(data.shape) > 2:
            raise TensorsNotSupportedException(
                "Attempting to infer schema from numpy array with " "shape {}".format(data.shape)
            )
        if data.dtype == np.object:
            data = pd.DataFrame(data).infer_objects()
            schema = Schema(
                [ColSpec(type=_infer_numpy_array(data[col].values)) for col in data.columns]
            )
        elif len(data.shape) == 1:
            schema = Schema([ColSpec(type=_infer_numpy_dtype(data.dtype))])
        elif len(data.shape) == 2:
            schema = Schema([ColSpec(type=_infer_numpy_dtype(data.dtype))] * data.shape[1])
    elif _is_spark_df(data):
        schema = Schema(
            [
                ColSpec(type=_infer_spark_type(field.dataType), name=field.name)
                for field in data.schema.fields
            ]
        )
    else:
        raise TypeError(
            "Expected one of (pandas.DataFrame, numpy array, "
            "dictionary of (name -> numpy.ndarray), pyspark.sql.DataFrame) "
            "but got '{}'".format(type(data))
        )
    if any([t in (DataType.integer, DataType.long) for t in schema.column_types()]):
        warnings.warn(
            "Hint: Inferred schema contains integer column(s). Integer columns in "
            "Python cannot represent missing values. If your input data contains "
            "missing values at inference time, it will be encoded as floats and will "
            "cause a schema enforcement error. The best way to avoid this problem is "
            "to infer the model schema based on a realistic data sample (training "
            "dataset) that includes missing values. Alternatively, you can declare "
            "integer columns as doubles (float64) whenever these columns may have "
            "missing values. See `Handling Integers With Missing Values "
            "<https://www.mlflow.org/docs/latest/models.html#"
            "handling-integers-with-missing-values>`_ for more details.",
            stacklevel=2,
        )
    return schema


def _infer_numpy_dtype(dtype: np.dtype) -> DataType:
    if not isinstance(dtype, np.dtype):
        raise TypeError("Expected numpy.dtype, got '{}'.".format(type(dtype)))
    if dtype.kind == "b":
        return DataType.boolean
    elif dtype.kind == "i" or dtype.kind == "u":
        if dtype.itemsize < 4 or (dtype.kind == "i" and dtype.itemsize == 4):
            return DataType.integer
        elif dtype.itemsize < 8 or (dtype.kind == "i" and dtype.itemsize == 8):
            return DataType.long
    elif dtype.kind == "f":
        if dtype.itemsize <= 4:
            return DataType.float
        elif dtype.itemsize <= 8:
            return DataType.double

    elif dtype.kind == "U":
        return DataType.string
    elif dtype.kind == "S":
        return DataType.binary
    elif dtype.kind == "O":
        raise Exception(
            "Can not infer np.object without looking at the values, call "
            "_map_numpy_array instead."
        )
    raise MlflowException("Unsupported numpy data type '{0}', kind '{1}'".format(dtype, dtype.kind))


def _infer_numpy_array(col: np.ndarray) -> DataType:
    if not isinstance(col, np.ndarray):
        raise TypeError("Expected numpy.ndarray, got '{}'.".format(type(col)))
    if len(col.shape) > 1:
        raise MlflowException("Expected 1d array, got array with shape {}".format(col.shape))

    class IsInstanceOrNone(object):
        def __init__(self, *args):
            self.classes = args
            self.seen_instances = 0

        def __call__(self, x):
            if x is None:
                return True
            elif any(map(lambda c: isinstance(x, c), self.classes)):
                self.seen_instances += 1
                return True
            else:
                return False

    if col.dtype.kind == "O":
        is_binary_test = IsInstanceOrNone(bytes, bytearray)
        if all(map(is_binary_test, col)) and is_binary_test.seen_instances > 0:
            return DataType.binary
        is_string_test = IsInstanceOrNone(str)
        if all(map(is_string_test, col)) and is_string_test.seen_instances > 0:
            return DataType.string
        # NB: bool is also instance of int => boolean test must precede integer test.
        is_boolean_test = IsInstanceOrNone(bool)
        if all(map(is_boolean_test, col)) and is_boolean_test.seen_instances > 0:
            return DataType.boolean
        is_long_test = IsInstanceOrNone(int)
        if all(map(is_long_test, col)) and is_long_test.seen_instances > 0:
            return DataType.long
        is_double_test = IsInstanceOrNone(float)
        if all(map(is_double_test, col)) and is_double_test.seen_instances > 0:
            return DataType.double
        else:
            raise MlflowException(
                "Unable to map 'np.object' type to MLflow DataType. np.object can"
                "be mapped iff all values have identical data type which is one "
                "of (string, (bytes or byterray),  int, float)."
            )
    else:
        return _infer_numpy_dtype(col.dtype)


def _infer_spark_type(x) -> DataType:
    import pyspark.sql.types

    if isinstance(x, pyspark.sql.types.NumericType):
        if isinstance(x, pyspark.sql.types.IntegralType):
            if isinstance(x, pyspark.sql.types.LongType):
                return DataType.long
            else:
                return DataType.integer
        elif isinstance(x, pyspark.sql.types.FloatType):
            return DataType.float
        elif isinstance(x, pyspark.sql.types.DoubleType):
            return DataType.double
    elif isinstance(x, pyspark.sql.types.BooleanType):
        return DataType.boolean
    elif isinstance(x, pyspark.sql.types.StringType):
        return DataType.string
    elif isinstance(x, pyspark.sql.types.BinaryType):
        return DataType.binary
    else:
        raise Exception(
            "Unsupported Spark Type '{}', MLflow schema is only supported for scalar "
            "Spark types.".format(type(x))
        )


def _is_spark_df(x) -> bool:
    try:
        import pyspark.sql.dataframe

        return isinstance(x, pyspark.sql.dataframe.DataFrame)
    except ImportError:
        return False
