from typing import Any
import warnings

import numpy as np
import pandas as pd
from typing import Optional

from mlflow.exceptions import MlflowException
from mlflow.types import DataType
from mlflow.types.schema import Schema, ColSpec, TensorSpec


class TensorsNotSupportedException(MlflowException):
    def __init__(self, msg):
        super().__init__(
            "Multidimensional arrays (aka tensors) are not supported. " "{}".format(msg)
        )


def _get_tensor_shape(data, variable_dimension: Optional[int] = 0) -> tuple:
    """
    Infer the shape of the inputted data.

    This method creates the shape of the tensor to store in the TensorSpec. The variable dimension
    is assumed to be the first dimension by default. This assumption can be overridden by inputting
    a different variable dimension or `None` to represent that the input tensor does not contain a
    variable dimension.

    :param data: Dataset to infer from.
    :param variable_dimension: An optional integer representing a variable dimension.
    :return: tuple : Shape of the inputted data (including a variable dimension)
    """
    from scipy.sparse import csr_matrix, csc_matrix

    if not isinstance(data, (np.ndarray, csr_matrix, csc_matrix)):
        raise TypeError("Expected numpy.ndarray or csc/csr matrix, got '{}'.".format(type(data)))
    variable_input_data_shape = data.shape
    if variable_dimension is not None:
        try:
            variable_input_data_shape = list(variable_input_data_shape)
            variable_input_data_shape[variable_dimension] = -1
        except IndexError:
            raise MlflowException(
                "The specified variable_dimension {0} is out of bounds with"
                "respect to the number of dimensions {1} in the input dataset".format(
                    variable_dimension, data.ndim
                )
            )
    return tuple(variable_input_data_shape)


def clean_tensor_type(dtype: np.dtype):
    """
    This method strips away the size information stored in flexible datatypes such as np.str_ and
    np.bytes_. Other numpy dtypes are returned unchanged.

    :param dtype: Numpy dtype of a tensor
    :return: dtype: Cleaned numpy dtype
    """
    if not isinstance(dtype, np.dtype):
        raise TypeError(
            "Expected `type` to be instance of `{0}`, received `{1}`".format(
                np.dtype, dtype.__class__
            )
        )

    # Special casing for np.str_ and np.bytes_
    if dtype.char == "U":
        return np.dtype("str")
    elif dtype.char == "S":
        return np.dtype("bytes")
    return dtype


def _infer_schema(data: Any) -> Schema:
    """
    Infer an MLflow schema from a dataset.

    Data inputted as a numpy array or a dictionary is represented by :py:class:`TensorSpec`.
    All other inputted data types are specified by :py:class:`ColSpec`.

    A `TensorSpec` captures the data shape (default variable axis is 0), the data type (numpy.dtype)
    and an optional name for each individual tensor of the dataset.
    A `ColSpec` captures the data type (defined in :py:class:`DataType`) and an optional name for
    each individual column of the dataset.

    This method will raise an exception if the user data contains incompatible types or is not
    passed in one of the supported formats (containers).

    The input should be one of these:
      - pandas.DataFrame or pandas.Series
      - dictionary of { name -> numpy.ndarray}
      - numpy.ndarray
      - pyspark.sql.DataFrame
      - csc/csr matrix

    The element types should be mappable to one of :py:class:`mlflow.models.signature.DataType` for
    dataframes and to one of numpy types for tensors.

    :param data: Dataset to infer from.

    :return: Schema
    """
    from scipy.sparse import csr_matrix, csc_matrix

    if isinstance(data, dict):
        res = []
        for name in data.keys():
            ndarray = data[name]
            if not isinstance(ndarray, np.ndarray):
                raise TypeError("Data in the dictionary must be of type numpy.ndarray")
            res.append(
                TensorSpec(
                    type=clean_tensor_type(ndarray.dtype),
                    shape=_get_tensor_shape(ndarray),
                    name=name,
                )
            )
        schema = Schema(res)
    elif isinstance(data, pd.Series):
        schema = Schema([ColSpec(type=_infer_pandas_column(data))])
    elif isinstance(data, pd.DataFrame):
        schema = Schema(
            [ColSpec(type=_infer_pandas_column(data[col]), name=col) for col in data.columns]
        )
    elif isinstance(data, np.ndarray):
        schema = Schema(
            [TensorSpec(type=clean_tensor_type(data.dtype), shape=_get_tensor_shape(data))]
        )
    elif isinstance(data, (csc_matrix, csr_matrix)):
        schema = Schema(
            [TensorSpec(type=clean_tensor_type(data.data.dtype), shape=_get_tensor_shape(data))]
        )
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
    if not schema.is_tensor_spec() and any(
        [t in (DataType.integer, DataType.long) for t in schema.column_types()]
    ):
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


def _infer_numpy_dtype(dtype) -> DataType:
    supported_types = np.dtype

    # noinspection PyBroadException
    try:
        from pandas.core.dtypes.base import ExtensionDtype

        supported_types = (np.dtype, ExtensionDtype)
    except ImportError:
        # This version of pandas does not support extension types
        pass
    if not isinstance(dtype, supported_types):
        raise TypeError(
            "Expected numpy.dtype or pandas.ExtensionDtype, got '{}'.".format(type(dtype))
        )

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
    elif dtype.kind == "M":
        return DataType.datetime
    raise MlflowException("Unsupported numpy data type '{0}', kind '{1}'".format(dtype, dtype.kind))


def _infer_pandas_column(col: pd.Series) -> DataType:
    if not isinstance(col, pd.Series):
        raise TypeError("Expected pandas.Series, got '{}'.".format(type(col)))
    if len(col.values.shape) > 1:
        raise MlflowException("Expected 1d array, got array with shape {}".format(col.shape))

    class IsInstanceOrNone:
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
        col = col.infer_objects()
    if col.dtype.kind == "O":
        # NB: Objects can be either binary or string. Pandas may consider binary data to be a string
        # so we need to check for binary first.
        is_binary_test = IsInstanceOrNone(bytes, bytearray)
        if all(map(is_binary_test, col)) and is_binary_test.seen_instances > 0:
            return DataType.binary
        elif pd.api.types.is_string_dtype(col):
            return DataType.string
        else:
            raise MlflowException(
                "Unable to map 'np.object' type to MLflow DataType. np.object can"
                "be mapped iff all values have identical data type which is one "
                "of (string, (bytes or byterray),  int, float)."
            )
    else:
        # NB: The following works for numpy types as well as pandas extension types.
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
    # NB: Spark differentiates date and timestamps, so we coerce both to TimestampType.
    elif isinstance(x, (pyspark.sql.types.DateType, pyspark.sql.types.TimestampType)):
        return DataType.datetime
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
