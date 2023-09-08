import logging
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types import DataType
from mlflow.types.schema import ColSpec, ParamSchema, ParamSpec, Schema, TensorSpec

_logger = logging.getLogger(__name__)


class TensorsNotSupportedException(MlflowException):
    def __init__(self, msg):
        super().__init__(f"Multidimensional arrays (aka tensors) are not supported. {msg}")


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

    :param dtype: Numpy dtype of a tensor
    :return: dtype: Cleaned numpy dtype
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
      - dictionary of { name -> [str, List[str]}
      - numpy.ndarray
      - pyspark.sql.DataFrame
      - csc/csr matrix
      - str
      - List[str]
      - List[Dict[str, Union[str, List[str]]]]
      - Dict[str, Union[str, List[str]]]
      - bytes

    The element types should be mappable to one of :py:class:`mlflow.models.signature.DataType` for
    dataframes and to one of numpy types for tensors.

    :param data: Dataset to infer from.

    :return: Schema
    """
    from scipy.sparse import csc_matrix, csr_matrix

    if isinstance(data, dict) and all(isinstance(values, np.ndarray) for values in data.values()):
        res = []
        for name in data.keys():
            ndarray = data[name]
            res.append(
                TensorSpec(
                    type=clean_tensor_type(ndarray.dtype),
                    shape=_get_tensor_shape(ndarray),
                    name=name,
                )
            )
        schema = Schema(res)
    elif isinstance(data, pd.Series):
        name = getattr(data, "name", None)
        schema = Schema([ColSpec(type=_infer_pandas_column(data), name=name)])
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
    elif isinstance(data, dict):
        _validate_input_dictionary_contains_only_strings_and_lists_of_strings(data)
        schema = Schema([ColSpec(type=DataType.string, name=name) for name in data.keys()])
    elif isinstance(data, str):
        schema = Schema([ColSpec(type=DataType.string)])
    elif isinstance(data, bytes):
        schema = Schema([ColSpec(type=DataType.binary)])
    elif isinstance(data, list) and all(isinstance(element, str) for element in data):
        schema = Schema([ColSpec(type=DataType.string)])
    elif (
        isinstance(data, list)
        and all(isinstance(element, dict) for element in data)
        and all(isinstance(key, str) for d in data for key in d)
        # NB: We allow both str and List[str] as values in the dictionary
        # e.g. [{'output': 'some sentence', 'ids': ['id1', 'id2']}]
        and all(
            isinstance(value, str)
            or (isinstance(value, list) and all(isinstance(v, str) for v in value))
            for d in data
            for value in d.values()
        )
    ):
        first_keys = data[0].keys()
        if all(d.keys() == first_keys for d in data):
            schema = Schema([ColSpec(type=DataType.string, name=name) for name in first_keys])
        else:
            raise MlflowException(
                "The list of dictionaries supplied has inconsistent keys among "
                "each dictionary in the list. Please validate the uniformity "
                "in the key naming for each dictionary.",
                error_code=INVALID_PARAMETER_VALUE,
            )
    else:
        raise TypeError(
            "Expected one of the following types:\n"
            "- pandas.DataFrame\n"
            "- pandas.Series\n"
            "- numpy.ndarray\n"
            "- dictionary of (name -> numpy.ndarray)\n"
            "- pyspark.sql.DataFrame\n",
            "- scipy.sparse.csr_matrix\n"
            "- scipy.sparse.csc_matrix\n"
            "- str\n"
            "- List[str]\n"
            "- List[Dict[str, Union[str, List[str]]]]\n"
            "- Dict[str, Union[str, List[str]]]\n"
            "- bytes\n"
            f"but got '{type(data)}'",
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


def _infer_pandas_column(col: pd.Series) -> DataType:
    if not isinstance(col, pd.Series):
        raise TypeError(f"Expected pandas.Series, got '{type(col)}'.")
    if len(col.values.shape) > 1:
        raise MlflowException(f"Expected 1d array, got array with shape {col.shape}")

    class IsInstanceOrNone:
        def __init__(self, *args):
            self.classes = args
            self.seen_instances = 0

        def __call__(self, x):
            if x is None:
                return True
            elif any(isinstance(x, c) for c in self.classes):
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
                "Unable to map 'object' type to MLflow DataType. object can "
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
            f"Unsupported Spark Type '{type(x)}', MLflow schema is only supported for scalar "
            "Spark types."
        )


def _is_spark_df(x) -> bool:
    try:
        import pyspark.sql.dataframe

        return isinstance(x, pyspark.sql.dataframe.DataFrame)
    except ImportError:
        return False


def _validate_input_dictionary_contains_only_strings_and_lists_of_strings(data) -> None:
    invalid_keys = []
    invalid_values = []
    value_type = None
    for key, value in data.items():
        if not value_type:
            value_type = type(value)
        if isinstance(key, bool):
            invalid_keys.append(key)
        elif not isinstance(key, (str, int)):
            invalid_keys.append(key)
        if isinstance(value, list) and not all(isinstance(item, (str, bytes)) for item in value):
            invalid_values.append(key)
        elif not isinstance(value, (np.ndarray, list, str, bytes)):
            invalid_values.append(key)
        elif isinstance(value, np.ndarray) or value_type == np.ndarray:
            if not isinstance(value, value_type):
                invalid_values.append(key)
    if invalid_values:
        raise MlflowException(
            "Invalid values in dictionary. If passing a dictionary containing strings, all "
            "values must be either strings or lists of strings. If passing a dictionary containing "
            "numeric values, the data must be enclosed in a numpy.ndarray. The following keys "
            f"in the input dictionary are invalid: {invalid_values}",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if invalid_keys:
        raise MlflowException(
            f"The dictionary keys are not all strings or indexes. Invalid keys: {invalid_keys}"
        )


def _is_all_string(x):
    return all(isinstance(v, str) for v in x)


def _validate_is_all_string(x):
    if not _is_all_string(x):
        raise MlflowException(f"Expected all values to be string, got {x}", INVALID_PARAMETER_VALUE)


def _validate_all_keys_string(d):
    keys = list(d.keys())
    if not _is_all_string(keys):
        raise MlflowException(
            f"Expected example to be dict with string keys, got {keys}",
            INVALID_PARAMETER_VALUE,
        )


def _validate_all_values_string(d):
    values = list(d.values())
    if not _is_all_string(values):
        raise MlflowException(
            f"Expected example to be dict with string values, got {values}", INVALID_PARAMETER_VALUE
        )


def _validate_keys_match(d, expected_keys):
    if d.keys() != expected_keys:
        raise MlflowException(
            f"Expected example to be dict with keys {list(expected_keys)}, got {list(d.keys())}",
            INVALID_PARAMETER_VALUE,
        )


def _validate_num_items(d, num_items):
    actual_num_items = len(d)
    if actual_num_items != num_items:
        raise MlflowException(
            f"Expected example to be dict with {num_items} items, got {actual_num_items}",
            INVALID_PARAMETER_VALUE,
        )


def _validate_has_items(d):
    num_items = len(d)
    if num_items == 0:
        raise MlflowException(
            f"Expected example to be dict with at least one item, got {num_items}",
            INVALID_PARAMETER_VALUE,
        )


def _validate_is_dict(d):
    if not isinstance(d, dict):
        raise MlflowException(
            f"Expected each item in example to be dict, got {type(d).__name__}",
            INVALID_PARAMETER_VALUE,
        )


def _validate_non_empty(examples):
    num_items = len(examples)
    if num_items == 0:
        raise MlflowException(
            f"Expected examples to be non-empty list, got {num_items}",
            INVALID_PARAMETER_VALUE,
        )


def _validate_is_list(examples):
    if not isinstance(examples, list):
        raise MlflowException(
            f"Expected examples to be list, got {type(examples).__name__}",
            INVALID_PARAMETER_VALUE,
        )


def _validate_dict_examples(examples, num_items=None):
    examples_iter = iter(examples)
    first_example = next(examples_iter)
    _validate_is_dict(first_example)
    _validate_has_items(first_example)
    if num_items is not None:
        _validate_num_items(first_example, num_items)
    _validate_all_keys_string(first_example)
    _validate_all_values_string(first_example)
    first_keys = first_example.keys()

    for example in examples_iter:
        _validate_is_dict(example)
        _validate_has_items(example)
        if num_items is not None:
            _validate_num_items(example, num_items)
        _validate_all_keys_string(example)
        _validate_all_values_string(example)
        _validate_keys_match(example, first_keys)


def _infer_schema_from_type_hint(type_hint, examples=None):
    has_examples = examples is not None
    if has_examples:
        _validate_is_list(examples)
        _validate_non_empty(examples)

    if type_hint == List[str]:
        if has_examples:
            _validate_is_all_string(examples)
        return Schema([ColSpec(type="string", name=None)])
    elif type_hint == List[Dict[str, str]]:
        if has_examples:
            _validate_dict_examples(examples)
            return Schema([ColSpec(type="string", name=name) for name in examples[0]])
        else:
            _logger.warning(f"Could not infer schema for {type_hint} because example is missing")
            return Schema([ColSpec(type="string", name=None)])
    else:
        _logger.info("Unsupported type hint: %s, skipping schema inference", type_hint)
        return None


def _infer_type_and_shape(value):
    if isinstance(value, (list, np.ndarray)):
        ndim = np.array(value).ndim
        if ndim != 1:
            raise MlflowException.invalid_parameter_value(
                f"Expected parameters to be 1D array or scalar, got {ndim}D array",
            )
        if all(DataType.is_datetime(v) for v in value):
            return DataType.datetime, (-1,)
        value_type = _infer_numpy_dtype(np.array(value).dtype)
        return value_type, (-1,)
    elif DataType.is_datetime(value):
        return DataType.datetime, None
    elif np.isscalar(value):
        try:
            value_type = _infer_numpy_dtype(np.array(value).dtype)
            return value_type, None
        except (Exception, MlflowException) as e:
            raise MlflowException.invalid_parameter_value(
                f"Failed to infer schema for parameter {value}: {e!r}"
            )
    raise MlflowException.invalid_parameter_value(
        f"Expected parameters to be 1D array or scalar, got {type(value).__name__}",
    )


def _infer_param_schema(parameters: Dict[str, Any]):
    if not isinstance(parameters, dict):
        raise MlflowException.invalid_parameter_value(
            f"Expected parameters to be dict, got {type(parameters).__name__}",
        )

    param_specs = []
    invalid_params = []
    for name, value in parameters.items():
        try:
            value_type, shape = _infer_type_and_shape(value)
            param_specs.append(ParamSpec(name=name, dtype=value_type, default=value, shape=shape))
        except Exception as e:
            invalid_params.append((name, value, e))

    if invalid_params:
        raise MlflowException.invalid_parameter_value(
            f"Failed to infer schema for parameters: {invalid_params}",
        )

    return ParamSchema(param_specs)
