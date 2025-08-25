import logging
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pydantic

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types import DataType
from mlflow.types.schema import (
    HAS_PYSPARK,
    AnyType,
    Array,
    ColSpec,
    Map,
    Object,
    ParamSchema,
    ParamSpec,
    Property,
    Schema,
    SparkMLVector,
    TensorSpec,
)

MULTIPLE_TYPES_ERROR_MSG = (
    "Expected all values in the list to be of the same type. To specify a model signature "
    "with a list containing elements of multiple types, define the signature manually "
    "using the Array(AnyType()) type from mlflow.models.schema."
)
_logger = logging.getLogger(__name__)


class TensorsNotSupportedException(MlflowException):
    def __init__(self, msg):
        super().__init__(f"Multidimensional arrays (aka tensors) are not supported. {msg}")


def _get_tensor_shape(data, variable_dimension: int | None = 0) -> tuple[int, ...]:
    """Infer the shape of the inputted data.

    This method creates the shape of the tensor to store in the TensorSpec. The variable dimension
    is assumed to be the first dimension by default. This assumption can be overridden by inputting
    a different variable dimension or `None` to represent that the input tensor does not contain a
    variable dimension.

    Args:
        data: Dataset to infer from.
        variable_dimension: An optional integer representing a variable dimension.

    Returns:
        tuple: Shape of the inputted data (including a variable dimension)
    """
    from scipy.sparse import csc_matrix, csr_matrix

    if not isinstance(data, (np.ndarray, csr_matrix, csc_matrix)):
        raise TypeError(f"Expected numpy.ndarray or csc/csr matrix, got '{type(data)}'.")
    variable_input_data_shape = data.shape
    if variable_dimension is not None:
        try:
            variable_input_data_shape = list(variable_input_data_shape)
            variable_input_data_shape[variable_dimension] = -1
        except IndexError:
            raise MlflowException(
                f"The specified variable_dimension {variable_dimension} is out of bounds with "
                f"respect to the number of dimensions {data.ndim} in the input dataset"
            )
    return tuple(variable_input_data_shape)


def clean_tensor_type(dtype: np.dtype):
    """
    This method strips away the size information stored in flexible datatypes such as np.str_ and
    np.bytes_. Other numpy dtypes are returned unchanged.

    Args:
        dtype: Numpy dtype of a tensor

    Returns:
        dtype: Cleaned numpy dtype
    """
    if not isinstance(dtype, np.dtype):
        raise TypeError(
            f"Expected `type` to be instance of `{np.dtype}`, received `{dtype.__class__}`"
        )

    # Special casing for np.str_ and np.bytes_
    if dtype.char == "U":
        return np.dtype("str")
    elif dtype.char == "S":
        return np.dtype("bytes")
    return dtype


def _infer_colspec_type(data: Any) -> DataType | Array | Object | AnyType:
    """
    Infer an MLflow Colspec type from the dataset.

    Args:
        data: data to infer from.

    Returns:
        Object
    """
    dtype = _infer_datatype(data)

    if dtype is None:
        raise MlflowException(
            f"Numpy array must include at least one non-empty item. Invalid input `{data}`."
        )

    return dtype


class InvalidDataForSignatureInferenceError(MlflowException):
    def __init__(self, message):
        super().__init__(message=message, error_code=INVALID_PARAMETER_VALUE)


def _infer_datatype(data: Any) -> DataType | Array | Object | AnyType | None:
    """
    Infer the datatype of input data.
    Data type and inferred schema type mapping:
        - dict -> Object
        - list -> Array
        - numpy.ndarray -> Array
        - scalar -> DataType
        - None, empty dictionary/list -> AnyType

    .. Note::
        Empty numpy arrays are inferred as None to keep the backward compatibility, as numpy
        arrays are used by some traditional ML flavors.
        e.g. numpy.array([]) -> None, numpy.array([[], []]) -> None
        While empty lists are inferred as AnyType instead of None after the support of AnyType.
        e.g. [] -> AnyType, [[], []] -> Array(Any)
    """
    if isinstance(data, pydantic.BaseModel):
        raise InvalidDataForSignatureInferenceError(
            message="MLflow does not support inferring model signature from input example "
            "with Pydantic objects. To use Pydantic objects, define your PythonModel's "
            "`predict` method with a Pydantic type hint, and model signature will be automatically "
            "inferred when logging the model. e.g. "
            "`def predict(self, model_input: list[PydanticType])`. Check "
            "https://mlflow.org/docs/latest/model/python_model.html#type-hint-usage-in-pythonmodel "
            "for more details."
        )

    if _is_none_or_nan(data) or (isinstance(data, (list, dict)) and not data):
        return AnyType()

    if isinstance(data, dict):
        properties = []
        for k, v in data.items():
            dtype = _infer_datatype(v)
            if dtype is None:
                raise MlflowException("Dictionary value must not be an empty numpy array.")
            properties.append(
                Property(name=k, dtype=dtype, required=not isinstance(dtype, AnyType))
            )
        return Object(properties=properties)

    if isinstance(data, (list, np.ndarray)):
        return _infer_array_datatype(data)

    return _infer_scalar_datatype(data)


def _infer_array_datatype(data: list[Any] | np.ndarray) -> Array | None:
    """Infer schema from an array. This tries to infer type if there is at least one
    non-null item in the list, assuming the list has a homogeneous type. However,
    if the list is empty or all items are null, returns None as a sign of undetermined.

    E.g.
        ["a", "b"] => Array(string)
        ["a", None] => Array(string)
        [["a", "b"], []] => Array(Array(string))
        [["a", "b"], None] => Array(Array(string))
        [] => None
        [None] => Array(Any)

    Args:
        data: data to infer from.

    Returns:
        Array(dtype) or None if undetermined
    """
    result = None
    for item in data:
        dtype = _infer_datatype(item)

        # Skip item with undetermined type
        if dtype is None:
            continue

        if result is None:
            result = Array(dtype)
        elif isinstance(result.dtype, (Array, Object, Map, AnyType)):
            try:
                result = Array(result.dtype._merge(dtype))
            except MlflowException as e:
                raise MlflowException.invalid_parameter_value(MULTIPLE_TYPES_ERROR_MSG) from e
        elif isinstance(result.dtype, DataType):
            if not isinstance(dtype, AnyType) and dtype != result.dtype:
                raise MlflowException.invalid_parameter_value(MULTIPLE_TYPES_ERROR_MSG)
        else:
            raise MlflowException.invalid_parameter_value(
                f"{dtype} is not a valid type for an item of a list or numpy array."
            )
    return result


# datetime is not included here
SCALAR_TO_DATATYPE_MAPPING = {
    bool: DataType.boolean,
    np.bool_: DataType.boolean,
    int: DataType.long,
    np.int64: DataType.long,
    np.int32: DataType.integer,
    float: DataType.double,
    np.float64: DataType.double,
    np.float32: DataType.float,
    str: DataType.string,
    np.str_: DataType.string,
    object: DataType.string,
    bytes: DataType.binary,
    np.bytes_: DataType.binary,
    bytearray: DataType.binary,
}


def _infer_scalar_datatype(data) -> DataType:
    if data_type := SCALAR_TO_DATATYPE_MAPPING.get(type(data)):
        return data_type
    if DataType.check_type(DataType.datetime, data):
        return DataType.datetime
    if HAS_PYSPARK:
        for data_type in DataType.all_types():
            if isinstance(data, type(data_type.to_spark())):
                return data_type
    raise MlflowException.invalid_parameter_value(
        f"Data {data} is not one of the supported DataType"
    )


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
      - pandas.DataFrame
      - pandas.Series
      - numpy.ndarray
      - dictionary of (name -> numpy.ndarray)
      - pyspark.sql.DataFrame
      - scipy.sparse.csr_matrix/csc_matrix
      - DataType
      - List[DataType]
      - Dict[str, Union[DataType, List, Dict]]
      - List[Dict[str, Union[DataType, List, Dict]]]

    The last two formats are used to represent complex data structures. For example,

        Input Data:
            [
                {
                    'text': 'some sentence',
                    'ids': ['id1'],
                    'dict': {'key': 'value'}
                },
                {
                    'text': 'some sentence',
                    'ids': ['id1', 'id2'],
                    'dict': {'key': 'value', 'key2': 'value2'}
                },
            ]

        The corresponding pandas DataFrame representation should look like this:

                    output         ids                                dict
            0  some sentence  [id1, id2]                    {'key': 'value'}
            1  some sentence  [id1, id2]  {'key': 'value', 'key2': 'value2'}

        The inferred schema should look like this:

            Schema([
                ColSpec(type=DataType.string, name='output'),
                ColSpec(type=Array(dtype=DataType.string), name='ids'),
                ColSpec(
                    type=Object([
                        Property(name='key', dtype=DataType.string),
                        Property(name='key2', dtype=DataType.string, required=False)
                    ]),
                    name='dict')]
                ),
            ])

    The element types should be mappable to one of :py:class:`mlflow.models.signature.DataType` for
    dataframes and to one of numpy types for tensors.

    Args:
        data: Dataset to infer from.

    Returns:
        Schema
    """
    from scipy.sparse import csc_matrix, csr_matrix

    # To keep backward compatibility with < 2.9.0, an empty list is inferred as string.
    #   ref: https://github.com/mlflow/mlflow/pull/10125#discussion_r1372751487
    if isinstance(data, list) and data == []:
        return Schema([ColSpec(DataType.string)])

    if isinstance(data, list) and all(isinstance(value, dict) for value in data):
        col_data_mapping = defaultdict(list)
        for item in data:
            for k, v in item.items():
                col_data_mapping[k].append(v)
        requiredness = {}
        for col in col_data_mapping:
            # if col exists in item but its value is None, then it is not required
            requiredness[col] = all(item.get(col) is not None for item in data)

        schema = Schema(
            [
                ColSpec(_infer_colspec_type(values).dtype, name=name, required=requiredness[name])
                for name, values in col_data_mapping.items()
            ]
        )

    elif isinstance(data, dict):
        # dictionary of (name -> numpy.ndarray)
        if all(isinstance(values, np.ndarray) for values in data.values()):
            schema = Schema(
                [
                    TensorSpec(
                        type=clean_tensor_type(ndarray.dtype),
                        shape=_get_tensor_shape(ndarray),
                        name=name,
                    )
                    for name, ndarray in data.items()
                ]
            )
        # Dict[str, Union[DataType, List, Dict]]
        else:
            if any(not isinstance(key, str) for key in data):
                raise MlflowException("The dictionary keys are not all strings.")
            schema = Schema(
                [
                    ColSpec(
                        _infer_colspec_type(value),
                        name=name,
                        required=_infer_required(value),
                    )
                    for name, value in data.items()
                ]
            )
    # pandas.Series
    elif isinstance(data, pd.Series):
        name = getattr(data, "name", None)
        schema = Schema(
            [
                ColSpec(
                    type=_infer_pandas_column(data),
                    name=name,
                    required=_infer_required(data),
                )
            ]
        )
    # pandas.DataFrame
    elif isinstance(data, pd.DataFrame):
        schema = Schema(
            [
                ColSpec(
                    type=_infer_pandas_column(data[col]),
                    name=col,
                    required=_infer_required(data[col]),
                )
                for col in data.columns
            ]
        )
    # numpy.ndarray
    elif isinstance(data, np.ndarray):
        schema = Schema(
            [TensorSpec(type=clean_tensor_type(data.dtype), shape=_get_tensor_shape(data))]
        )
    # scipy.sparse.csr_matrix/csc_matrix
    elif isinstance(data, (csc_matrix, csr_matrix)):
        schema = Schema(
            [TensorSpec(type=clean_tensor_type(data.data.dtype), shape=_get_tensor_shape(data))]
        )
    # pyspark.sql.DataFrame
    elif _is_spark_df(data):
        schema = Schema(
            [
                ColSpec(
                    type=_infer_spark_type(field.dataType, data, field.name),
                    name=field.name,
                    # Avoid setting required field for spark dataframe
                    # as the default value for spark df nullable is True
                    # which counterparts to default required=True in ColSpec
                )
                for field in data.schema.fields
            ]
        )
    elif isinstance(data, list):
        # Assume list as a single column
        # List[DataType]
        # e.g. ['some sentence', 'some sentence'] -> Schema([ColSpec(type=DataType.string)])
        # The corresponding pandas DataFrame representation should be pd.DataFrame(data)
        # We set required=True as unnamed optional inputs is not allowed
        schema = Schema([ColSpec(_infer_colspec_type(data).dtype)])
    else:
        # DataType
        # e.g. "some sentence" -> Schema([ColSpec(type=DataType.string)])
        try:
            # We set required=True as unnamed optional inputs is not allowed
            schema = Schema([ColSpec(_infer_colspec_type(data))])
        except MlflowException as e:
            raise MlflowException.invalid_parameter_value(
                "Failed to infer schema. Expected one of the following types:\n"
                "- pandas.DataFrame\n"
                "- pandas.Series\n"
                "- numpy.ndarray\n"
                "- dictionary of (name -> numpy.ndarray)\n"
                "- pyspark.sql.DataFrame\n"
                "- scipy.sparse.csr_matrix\n"
                "- scipy.sparse.csc_matrix\n"
                "- DataType\n"
                "- List[DataType]\n"
                "- Dict[str, Union[DataType, List, Dict]]\n"
                "- List[Dict[str, Union[DataType, List, Dict]]]\n"
                f"but got '{data}'.\n"
                f"Error: {e}",
            )
    if not schema.is_tensor_spec() and any(
        t in (DataType.integer, DataType.long) for t in schema.input_types()
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
            "handling-integers-with-missing-values>`_ for more details."
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
        raise TypeError(f"Expected numpy.dtype or pandas.ExtensionDtype, got '{type(dtype)}'.")

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
            "Can not infer object without looking at the values, call _map_numpy_array instead."
        )
    elif dtype.kind == "M":
        return DataType.datetime
    raise MlflowException(f"Unsupported numpy data type '{dtype}', kind '{dtype.kind}'")


def _is_none_or_nan(x):
    if isinstance(x, float):
        return np.isnan(x)
    # NB: We can't use pd.isna() because the input can be a series.
    return x is None or x is pd.NA or x is pd.NaT


def _infer_required(col) -> bool:
    if isinstance(col, (list, pd.Series)):
        return not any(_is_none_or_nan(x) for x in col)
    return not _is_none_or_nan(col)


def _infer_pandas_column(col: pd.Series) -> DataType:
    if not isinstance(col, pd.Series):
        raise TypeError(f"Expected pandas.Series, got '{type(col)}'.")
    if len(col.values.shape) > 1:
        raise MlflowException(f"Expected 1d array, got array with shape {col.shape}")

    if col.dtype.kind == "O":
        col = col.infer_objects()
    if col.dtype.kind == "O":
        try:
            # We convert pandas Series into list and infer the schema.
            # The real schema for internal field should be the Array's dtype
            arr_type = _infer_colspec_type(col.to_list())
            return arr_type.dtype
        except Exception as e:
            # For backwards compatibility, we fall back to string
            # if the provided array is of string type
            # This is for diviner test where df field is ('key2', 'key1', 'key0')
            if pd.api.types.is_string_dtype(col):
                return DataType.string
            raise MlflowException(f"Failed to infer schema for pandas.Series {col}. Error: {e}")
    else:
        # NB: The following works for numpy types as well as pandas extension types.
        return _infer_numpy_dtype(col.dtype)


def _infer_spark_type(x, data=None, col_name=None) -> DataType:
    import pyspark.sql.types
    from pyspark.ml.linalg import VectorUDT
    from pyspark.sql.functions import col, collect_list

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
    elif isinstance(x, pyspark.sql.types.ArrayType):
        return Array(_infer_spark_type(x.elementType))
    elif isinstance(x, pyspark.sql.types.StructType):
        return Object(
            properties=[
                Property(
                    name=f.name,
                    dtype=_infer_spark_type(f.dataType),
                    required=not f.nullable,
                )
                for f in x.fields
            ]
        )
    elif isinstance(x, pyspark.sql.types.MapType):
        if data is None or col_name is None:
            raise MlflowException("Cannot infer schema for MapType without data and column name.")
        # Map MapType to StructType
        # Note that MapType assumes all values are of same type,
        # if they're not then spark picks the first item's type
        # and tries to convert rest to that type.
        # e.g.
        # >>> spark.createDataFrame([{"col": {"a": 1, "b": "b"}}]).show()
        # +-------------------+
        # |                col|
        # +-------------------+
        # |{a -> 1, b -> null}|
        # +-------------------+
        if isinstance(x.valueType, pyspark.sql.types.MapType):
            raise MlflowException(
                "Please construct spark DataFrame with schema using StructType "
                "for dictionary/map fields, MLflow schema inference only supports "
                "scalar, array and struct types."
            )

        merged_keys = (
            data.selectExpr(f"map_keys({col_name}) as keys")
            .agg(collect_list(col("keys")).alias("merged_keys"))
            .head()
            .merged_keys
        )
        keys = {key for sublist in merged_keys for key in sublist}
        return Object(
            properties=[
                Property(
                    name=k,
                    dtype=_infer_spark_type(x.valueType),
                )
                for k in keys
            ]
        )
    elif isinstance(x, VectorUDT):
        return SparkMLVector()

    else:
        raise MlflowException.invalid_parameter_value(
            f"Unsupported Spark Type '{type(x)}' for MLflow schema."
        )


def _is_spark_df(x) -> bool:
    try:
        import pyspark.sql.dataframe

        if isinstance(x, pyspark.sql.dataframe.DataFrame):
            return True
    except ImportError:
        return False
    # For spark 4.0
    try:
        import pyspark.sql.connect.dataframe

        return isinstance(x, pyspark.sql.connect.dataframe.DataFrame)
    except ImportError:
        return False


def _validate_input_dictionary_contains_only_strings_and_lists_of_strings(data) -> None:
    # isinstance(True, int) is True
    invalid_keys = [
        key for key in data.keys() if not isinstance(key, (str, int)) or isinstance(key, bool)
    ]
    if invalid_keys:
        raise MlflowException(
            f"The dictionary keys are not all strings or indexes. Invalid keys: {invalid_keys}"
        )
    if any(isinstance(value, np.ndarray) for value in data.values()) and not all(
        isinstance(value, np.ndarray) for value in data.values()
    ):
        raise MlflowException("The dictionary values are not all numpy.ndarray.")

    invalid_values = [
        key
        for key, value in data.items()
        if (isinstance(value, list) and not all(isinstance(item, (str, bytes)) for item in value))
        or (not isinstance(value, (np.ndarray, list, str, bytes)))
    ]
    if invalid_values:
        raise MlflowException.invalid_parameter_value(
            "Invalid values in dictionary. If passing a dictionary containing strings, all "
            "values must be either strings or lists of strings. If passing a dictionary containing "
            "numeric values, the data must be enclosed in a numpy.ndarray. The following keys "
            f"in the input dictionary are invalid: {invalid_values}",
        )


def _is_list_str(type_hint: Any) -> bool:
    return type_hint in [
        List[str],  # noqa: UP006
        list[str],
    ]


def _is_list_dict_str(type_hint: Any) -> bool:
    return type_hint in [
        List[Dict[str, str]],  # noqa: UP006
        list[Dict[str, str]],  # noqa: UP006
        List[dict[str, str]],  # noqa: UP006
        list[dict[str, str]],
    ]


def _get_array_depth(l: Any) -> int:
    if isinstance(l, np.ndarray):
        return l.ndim
    if isinstance(l, list):
        return max(_get_array_depth(item) for item in l) + 1 if l else 1
    return 0


def _infer_type_and_shape(value):
    if isinstance(value, (list, np.ndarray)):
        ndim = _get_array_depth(value)
        if ndim != 1:
            raise MlflowException.invalid_parameter_value(
                f"Expected parameters to be 1D array or scalar, got {ndim}D array",
            )
        if all(DataType.check_type(DataType.datetime, v) for v in value):
            return DataType.datetime, (-1,)
        value_type = _infer_numpy_dtype(np.array(value).dtype)
        return value_type, (-1,)
    elif DataType.check_type(DataType.datetime, value):
        return DataType.datetime, None
    elif np.isscalar(value):
        try:
            value_type = _infer_numpy_dtype(np.array(value).dtype)
            return value_type, None
        except (Exception, MlflowException) as e:
            raise MlflowException.invalid_parameter_value(
                f"Failed to infer schema for parameter {value}: {e!r}"
            )
    elif isinstance(value, dict):
        # reuse _infer_schema to infer schema for dict, wrapping it in a dictionary is
        # necessary to make sure value is inferred as Object
        schema = _infer_schema({"value": value})
        object_type = schema.inputs[0].type
        return object_type, None
    raise MlflowException.invalid_parameter_value(
        f"Expected parameters to be 1D array or scalar, got {type(value).__name__}",
    )


def _infer_param_schema(parameters: dict[str, Any]):
    if not isinstance(parameters, dict):
        raise MlflowException.invalid_parameter_value(
            f"Expected parameters to be dict, got {type(parameters).__name__}",
        )

    param_specs = []
    invalid_params = []
    for name, value in parameters.items():
        try:
            value_type, shape = _infer_type_and_shape(value)
            param_specs.append(
                ParamSpec(name=name, dtype=value_type, default=deepcopy(value), shape=shape)
            )
        except Exception as e:
            invalid_params.append((name, value, e))

    if invalid_params:
        raise MlflowException.invalid_parameter_value(
            f"Failed to infer schema for parameters: {invalid_params}",
        )

    return ParamSchema(param_specs)
