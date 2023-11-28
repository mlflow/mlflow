import decimal
import json
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import PIL

from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
from mlflow.models import Model
from mlflow.store.artifact.utils.models import get_model_name_and_version
from mlflow.types import DataType, ParamSchema, ParamSpec, Schema, TensorSpec
from mlflow.types.utils import TensorsNotSupportedException, _infer_param_schema, clean_tensor_type
from mlflow.utils.annotations import experimental
from mlflow.utils.proto_json_utils import (
    NumpyEncoder,
    dataframe_from_parsed_json,
    parse_tf_serving_input,
)
from mlflow.utils.uri import get_databricks_profile_uri_from_artifact_uri

try:
    from scipy.sparse import csc_matrix, csr_matrix

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

INPUT_EXAMPLE_PATH = "artifact_path"
EXAMPLE_DATA_KEY = "inputs"
EXAMPLE_PARAMS_KEY = "params"

ModelInputExample = Union[
    pd.DataFrame, np.ndarray, dict, list, "csr_matrix", "csc_matrix", str, bytes, tuple
]

PyFuncInput = Union[
    pd.DataFrame, pd.Series, np.ndarray, "csc_matrix", "csr_matrix", List[Any], Dict[str, Any], str
]
PyFuncOutput = Union[pd.DataFrame, pd.Series, np.ndarray, list, str]

_logger = logging.getLogger(__name__)


class _Example:
    """
    Represents an input example for MLflow model.

    Contains jsonable data that can be saved with the model and meta data about the exported format
    that can be saved with :py:class:`Model <mlflow.models.Model>`.

    The _Example is created from example data provided by user. The example(s) can be provided as
    pandas.DataFrame, numpy.ndarray, python dictionary or python list. The assumption is that the
    example contains jsonable elements (see storage format section below).

    NOTE: If the example is 1 dimensional (e.g. dictionary of str -> scalar, or a list of scalars),
    the assumption is that it is a single row of data (rather than a single column).

    Metadata:

    The _Example metadata contains the following information:
        - artifact_path: Relative path to the serialized example within the model directory.
        - type: Type of example data provided by the user. E.g. dataframe, ndarray.
        - One of the following metadata based on the `type`:
            - pandas_orient: For dataframes, this attribute specifies how is the dataframe encoded
                             in json. For example, "split" value signals that the data is stored as
                             object with columns and data attributes.
            - format: For tensors, this attribute specifies the standard being used to store an
                      input example. MLflow uses a JSON-formatted string representation of T
                      F serving input.

    Storage Format:

    The examples are stored as json for portability and readability. Therefore, the contents of the
    example(s) must be jsonable. MLflow will make the following conversions automatically on behalf
    of the user:

        - binary values: :py:class:`bytes` or :py:class:`bytearray` are converted to base64
          encoded strings.
        - numpy types: Numpy types are converted to the corresponding python types or their closest
          equivalent.
        - csc/csr matrix: similar to 2 dims numpy array, csc/csr matrix are converted to
          corresponding python types or their closest equivalent.
    """

    def __init__(self, input_example: ModelInputExample):
        def _is_scalar(x):
            return np.isscalar(x) or x is None

        def _validate_params(params):
            try:
                _infer_param_schema(params)
            except MlflowException:
                _logger.warning(f"Invalid params found in input example: {params}")
                raise

        def _is_ndarray(x):
            return isinstance(x, np.ndarray) or (
                isinstance(x, dict) and all(isinstance(ary, np.ndarray) for ary in x.values())
            )

        def _is_sparse_matrix(x):
            if not HAS_SCIPY:
                # we can safely assume that if no scipy is installed,
                # the user won't log scipy sparse matrices
                return False
            return isinstance(x, (csc_matrix, csr_matrix))

        def _handle_ndarray_nans(x: np.ndarray):
            if np.issubdtype(x.dtype, np.number):
                return np.where(np.isnan(x), None, x)
            else:
                return x

        def _handle_ndarray_input(input_array: Union[np.ndarray, dict]):
            if isinstance(input_array, dict):
                result = {}
                for name in input_array.keys():
                    result[name] = _handle_ndarray_nans(input_array[name]).tolist()
                return {"inputs": result}
            else:
                return {"inputs": _handle_ndarray_nans(input_array).tolist()}

        def _handle_sparse_matrix(x: Union["csr_matrix", "csc_matrix"]):
            return {
                "data": _handle_ndarray_nans(x.data).tolist(),
                "indices": x.indices.tolist(),
                "indptr": x.indptr.tolist(),
                "shape": list(x.shape),
            }

        def _handle_dataframe_nans(df: pd.DataFrame):
            return df.where(df.notnull(), None)

        def _coerce_to_pandas_df(input_ex):
            if isinstance(input_ex, dict):
                if all(_is_scalar(x) for x in input_ex.values()):
                    input_ex = pd.DataFrame([input_ex])
                elif all(isinstance(x, (str, list)) for x in input_ex.values()):
                    for value in input_ex.values():
                        if isinstance(value, list) and not all(_is_scalar(x) for x in value):
                            raise TypeError(
                                "List values within dictionaries must be of scalar type."
                            )
                    input_ex = pd.DataFrame(input_ex)
                else:
                    raise TypeError(
                        "Data in the dictionary must be scalar or of type numpy.ndarray"
                    )
            elif isinstance(input_ex, list):
                for i, x in enumerate(input_ex):
                    if isinstance(x, np.ndarray) and len(x.shape) > 1:
                        raise TensorsNotSupportedException(f"Row '{i}' has shape {x.shape}")
                if all(_is_scalar(x) for x in input_ex):
                    input_ex = pd.DataFrame([input_ex], columns=range(len(input_ex)))
                else:
                    input_ex = pd.DataFrame(input_ex)
            elif isinstance(input_ex, (str, bytes)):
                input_ex = pd.DataFrame([input_ex])
            elif not isinstance(input_ex, pd.DataFrame):
                try:
                    import pyspark.sql.dataframe

                    if isinstance(input_example, pyspark.sql.dataframe.DataFrame):
                        raise MlflowException(
                            "Examples can not be provided as Spark Dataframe. "
                            "Please make sure your example is of a small size and "
                            "turn it into a pandas DataFrame."
                        )
                except ImportError:
                    pass
                input_ex = None
            return input_ex

        def _handle_dataframe_input(df):
            result = _handle_dataframe_nans(df).to_dict(orient="split")
            # Do not include row index
            del result["index"]
            if all(df.columns == range(len(df.columns))):
                # No need to write default column index out
                del result["columns"]
            return result

        example_filename = "input_example.json"
        self.info = {
            INPUT_EXAMPLE_PATH: example_filename,
        }
        # Avoid changing the variable passed in
        input_example = deepcopy(input_example)
        if _contains_params(input_example):
            input_example, self._inference_params = input_example
            _validate_params(self._inference_params)
            self.info[EXAMPLE_PARAMS_KEY] = "true"
        else:
            self._inference_params = None

        if _is_ndarray(input_example):
            self._inference_data = input_example
            self.data = _handle_ndarray_input(input_example)
            self.info.update(
                {
                    "type": "ndarray",
                    "format": "tf-serving",
                }
            )
        elif _is_sparse_matrix(input_example):
            self._inference_data = input_example
            self.data = _handle_sparse_matrix(input_example)
            if isinstance(input_example, csc_matrix):
                example_type = "sparse_matrix_csc"
            else:
                example_type = "sparse_matrix_csr"
            self.info.update(
                {
                    "type": example_type,
                }
            )
        else:
            self._inference_data = _coerce_to_pandas_df(input_example)
            if self._inference_data is None:
                raise TypeError(
                    "Expected one of the following types:\n"
                    "- pandas.DataFrame\n"
                    "- numpy.ndarray\n"
                    "- dictionary of (name -> numpy.ndarray)\n"
                    "- scipy.sparse.csr_matrix\n"
                    "- scipy.sparse.csc_matrix\n"
                    "- dict\n"
                    "- list\n"
                    "- str\n"
                    "- bytes\n"
                    f"but got '{type(input_example)}'",
                )
            self.data = _handle_dataframe_input(self._inference_data)
            orient = "split" if "columns" in self.data else "values"
            self.info.update(
                {
                    "type": "dataframe",
                    "pandas_orient": orient,
                }
            )

    def save(self, parent_dir_path: str):
        """Save the example as json at ``parent_dir_path``/`self.info['artifact_path']`."""
        if self._inference_params is not None:
            data = {EXAMPLE_DATA_KEY: self.data, EXAMPLE_PARAMS_KEY: self._inference_params}
        else:
            data = self.data
        with open(os.path.join(parent_dir_path, self.info[INPUT_EXAMPLE_PATH]), "w") as f:
            json.dump(data, f, cls=NumpyEncoder)

    @property
    def inference_data(self):
        """
        Returns the input example in a form that PyFunc wrapped models can score.
        """
        return self._inference_data

    @property
    def inference_params(self):
        """
        Returns the params dictionary that PyFunc wrapped models can use for scoring.
        """
        return self._inference_params


def _contains_params(input_example):
    # For tuple input, we assume the first item is input_example data
    # and the second item is params dictionary.
    return (
        isinstance(input_example, tuple)
        and len(input_example) == 2
        and isinstance(input_example[1], dict)
    )


def _save_example(mlflow_model: Model, input_example: ModelInputExample, path: str):
    """
    Save example to a file on the given path and updates passed Model with example metadata.

    The metadata is a dictionary with the following fields:
      - 'artifact_path': example path relative to the model directory.
      - 'type': Type of example. Currently the supported values are 'dataframe' and 'ndarray'
      -  One of the following metadata based on the `type`:
            - 'pandas_orient': Used to store dataframes. Determines the json encoding for dataframe
                               examples in terms of pandas orient convention. Defaults to 'split'.
            - 'format: Used to store tensors. Determines the standard used to store a tensor input
                       example. MLflow uses a JSON-formatted string representation of TF serving
                       input.
    :param mlflow_model: Model metadata that will get updated with the example metadata.
    :param path: Where to store the example file. Should be model the model directory.
    """
    example = _Example(input_example)
    example.save(path)
    mlflow_model.saved_input_example_info = example.info


def _get_mlflow_model_input_example_dict(mlflow_model: Model, path: str):
    """
    Read input_example dictionary from the model artifact path. Returns None if there is no
    example metadata.
    :param mlflow_model: Model metadata.
    :param path: Path to the model directory.
    :return: Input example or None if the model has no example.
    """
    if mlflow_model.saved_input_example_info is None:
        return None
    example_type = mlflow_model.saved_input_example_info["type"]
    if example_type not in ["dataframe", "ndarray", "sparse_matrix_csc", "sparse_matrix_csr"]:
        raise MlflowException(f"This version of mlflow can not load example of type {example_type}")
    path = os.path.join(path, mlflow_model.saved_input_example_info["artifact_path"])
    with open(path) as handle:
        return json.load(handle)


def _read_example(mlflow_model: Model, path: str):
    """
    Read example from a model directory. Returns None if there is no example metadata (i.e. the
    model was saved without example). Raises FileNotFoundError if there is model metadata but the
    example file is missing.

    :param mlflow_model: Model metadata.
    :param path: Path to the model directory.
    :return: Input example data or None if the model has no example.
    """
    input_example = _get_mlflow_model_input_example_dict(mlflow_model, path)
    if input_example is None:
        return None

    example_type = mlflow_model.saved_input_example_info["type"]
    input_schema = mlflow_model.signature.inputs if mlflow_model.signature is not None else None
    if mlflow_model.saved_input_example_info.get(EXAMPLE_PARAMS_KEY, None):
        input_example = input_example[EXAMPLE_DATA_KEY]
    if example_type == "ndarray":
        return _read_tensor_input_from_json(input_example, schema=input_schema)
    if example_type in ["sparse_matrix_csc", "sparse_matrix_csr"]:
        return _read_sparse_matrix_from_json(input_example, example_type)
    return dataframe_from_parsed_json(input_example, pandas_orient="split", schema=input_schema)


def _read_example_params(mlflow_model: Model, path: str):
    """
    Read params of input_example from a model directory. Returns None if there is no params
    in the input_example or the model was saved without example.
    """
    if (
        mlflow_model.saved_input_example_info is None
        or mlflow_model.saved_input_example_info.get(EXAMPLE_PARAMS_KEY, None) is None
    ):
        return None
    input_example_dict = _get_mlflow_model_input_example_dict(mlflow_model, path)
    return input_example_dict[EXAMPLE_PARAMS_KEY]


def _read_tensor_input_from_json(path_or_data, schema=None):
    if isinstance(path_or_data, str) and os.path.exists(path_or_data):
        with open(path_or_data) as handle:
            inp_dict = json.load(handle)
    else:
        inp_dict = path_or_data
    return parse_tf_serving_input(inp_dict, schema)


def _read_sparse_matrix_from_json(path_or_data, example_type):
    if isinstance(path_or_data, str) and os.path.exists(path_or_data):
        with open(path_or_data) as handle:
            matrix_data = json.load(handle)
    else:
        matrix_data = path_or_data
    data = matrix_data["data"]
    indices = matrix_data["indices"]
    indptr = matrix_data["indptr"]
    shape = tuple(matrix_data["shape"])

    if example_type == "sparse_matrix_csc":
        return csc_matrix((data, indices, indptr), shape=shape)
    else:
        return csr_matrix((data, indices, indptr), shape=shape)


def plot_lines(data_series, xlabel, ylabel, legend_loc=None, line_kwargs=None, title=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    if line_kwargs is None:
        line_kwargs = {}

    for label, data_x, data_y in data_series:
        ax.plot(data_x, data_y, label=label, **line_kwargs)

    if legend_loc:
        ax.legend(loc=legend_loc)

    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)

    return fig, ax


def _enforce_tensor_spec(
    values: Union[np.ndarray, "csc_matrix", "csr_matrix"],
    tensor_spec: TensorSpec,
):
    """
    Enforce the input tensor shape and type matches the provided tensor spec.
    """
    expected_shape = tensor_spec.shape
    expected_type = tensor_spec.type
    actual_shape = values.shape
    actual_type = values.dtype if isinstance(values, np.ndarray) else values.data.dtype

    # This logic is for handling "ragged" arrays. The first check is for a standard numpy shape
    # representation of a ragged array. The second is for handling a more manual specification
    # of shape while support an input which is a ragged array.
    if len(expected_shape) == 1 and expected_shape[0] == -1 and expected_type == np.dtype("O"):
        # Sample spec: Tensor('object', (-1,))
        # Will pass on any provided input
        return values
    if (
        len(expected_shape) > 1
        and -1 in expected_shape[1:]
        and len(actual_shape) == 1
        and actual_type == np.dtype("O")
    ):
        # Sample spec: Tensor('float64', (-1, -1, -1, 3))
        # Will pass on inputs which are ragged arrays: shape==(x,), dtype=='object'
        return values

    if len(expected_shape) != len(actual_shape):
        raise MlflowException(
            f"Shape of input {actual_shape} does not match expected shape {expected_shape}."
        )
    for expected, actual in zip(expected_shape, actual_shape):
        if expected == -1:
            continue
        if expected != actual:
            raise MlflowException(
                f"Shape of input {actual_shape} does not match expected shape {expected_shape}."
            )
    if clean_tensor_type(actual_type) != expected_type:
        raise MlflowException(
            f"dtype of input {actual_type} does not match expected dtype {expected_type}"
        )
    return values


def _enforce_mlflow_datatype(name, values: pd.Series, t: DataType):
    """
    Enforce the input column type matches the declared in model input schema.

    The following type conversions are allowed:

    1. object -> string
    2. int -> long (upcast)
    3. float -> double (upcast)
    4. int -> double (safe conversion)
    5. np.datetime64[x] -> datetime (any precision)
    6. object -> datetime

    NB: pandas does not have native decimal data type, when user train and infer
    model from pyspark dataframe that contains decimal type, the schema will be
    treated as float64.
    7. decimal -> double

    Any other type mismatch will raise error.
    """
    if values.dtype == object and t not in (DataType.binary, DataType.string):
        values = values.infer_objects()

    if t == DataType.string and values.dtype == object:
        # NB: the object can contain any type and we currently cannot cast to pandas Strings
        # due to how None is cast
        return values

    # NB: Comparison of pandas and numpy data type fails when numpy data type is on the left hand
    # side of the comparison operator. It works, however, if pandas type is on the left hand side.
    # That is because pandas is aware of numpy.
    if t.to_pandas() == values.dtype or t.to_numpy() == values.dtype:
        # The types are already compatible => conversion is not necessary.
        return values

    if t == DataType.binary and values.dtype.kind == t.binary.to_numpy().kind:
        # NB: bytes in numpy have variable itemsize depending on the length of the longest
        # element in the array (column). Since MLflow binary type is length agnostic, we ignore
        # itemsize when matching binary columns.
        return values

    if t == DataType.datetime and values.dtype.kind == t.to_numpy().kind:
        # NB: datetime values have variable precision denoted by brackets, e.g. datetime64[ns]
        # denotes nanosecond precision. Since MLflow datetime type is precision agnostic, we
        # ignore precision when matching datetime columns.
        return values.astype(np.dtype("datetime64[ns]"))

    if t == DataType.datetime and values.dtype == object:
        # NB: Pyspark date columns get converted to object when converted to a pandas
        # DataFrame. To respect the original typing, we convert the column to datetime.
        try:
            return values.astype(np.dtype("datetime64[ns]"), errors="raise")
        except ValueError as e:
            raise MlflowException(
                "Failed to convert column {name} from type {values.dtype} to {t}."
            ) from e
    if t == DataType.double and values.dtype == decimal.Decimal:
        # NB: Pyspark Decimal column get converted to decimal.Decimal when converted to pandas
        # DataFrame. In order to support decimal data training from spark data frame, we add this
        # conversion even we might lose the precision.
        try:
            return pd.to_numeric(values, errors="raise")
        except ValueError:
            raise MlflowException(
                f"Failed to convert column {name} from type {values.dtype} to {t}."
            )

    numpy_type = t.to_numpy()
    if values.dtype.kind == numpy_type.kind:
        is_upcast = values.dtype.itemsize <= numpy_type.itemsize
    elif values.dtype.kind == "u" and numpy_type.kind == "i":
        is_upcast = values.dtype.itemsize < numpy_type.itemsize
    elif values.dtype.kind in ("i", "u") and numpy_type == np.float64:
        # allow (u)int => double conversion
        is_upcast = values.dtype.itemsize <= 6
    else:
        is_upcast = False

    if is_upcast:
        return values.astype(numpy_type, errors="raise")
    else:
        # NB: conversion between incompatible types (e.g. floats -> ints or
        # double -> float) are not allowed. While supported by pandas and numpy,
        # these conversions alter the values significantly.
        def all_ints(xs):
            return all(pd.isnull(x) or int(x) == x for x in xs)

        hint = ""
        if (
            values.dtype == np.float64
            and numpy_type.kind in ("i", "u")
            and values.hasnans
            and all_ints(values)
        ):
            hint = (
                " Hint: the type mismatch is likely caused by missing values. "
                "Integer columns in python can not represent missing values and are therefore "
                "encoded as floats. The best way to avoid this problem is to infer the model "
                "schema based on a realistic data sample (training dataset) that includes missing "
                "values. Alternatively, you can declare integer columns as doubles (float64) "
                "whenever these columns may have missing values. See `Handling Integers With "
                "Missing Values <https://www.mlflow.org/docs/latest/models.html#"
                "handling-integers-with-missing-values>`_ for more details."
            )

        raise MlflowException(
            f"Incompatible input types for column {name}. "
            f"Can not safely convert {values.dtype} to {numpy_type}.{hint}"
        )


def _enforce_unnamed_col_schema(pf_input: PyFuncInput, input_schema: Schema):
    """Enforce the input columns conform to the model's column-based signature."""
    input_names = pf_input.columns[: len(input_schema.inputs)]
    input_types = input_schema.input_types()
    new_pf_input = pd.DataFrame()
    for i, x in enumerate(input_names):
        new_pf_input[x] = _enforce_mlflow_datatype(x, pf_input[x], input_types[i])
    return new_pf_input


def _enforce_named_col_schema(pf_input: PyFuncInput, input_schema: Schema):
    """Enforce the input columns conform to the model's column-based signature."""
    required_input_names = input_schema.required_input_names()
    input_types = input_schema.input_types_dict()

    new_pf_input = pd.DataFrame()
    for x in required_input_names:
        new_pf_input[x] = _enforce_mlflow_datatype(x, pf_input[x], input_types[x])

    # Upstream validation means that if we're here, pf_input can only be a
    # pandas dataframe or a dict.
    optional_input_names = input_schema.optional_input_names()
    if isinstance(pf_input, pd.DataFrame):
        supplied_optional_inputs = [col for col in pf_input.columns if col in optional_input_names]
    elif isinstance(pf_input, dict):
        supplied_optional_inputs = [col for col in pf_input.keys() if col in optional_input_names]
    else:
        supplied_optional_inputs = []
    # Iterate over supplied optional inputs rather than all optional inputs.
    for x in supplied_optional_inputs:
        new_pf_input[x] = _enforce_mlflow_datatype(x, pf_input[x], input_types[x])
    return new_pf_input


def _reshape_and_cast_pandas_column_values(name, pd_series, tensor_spec):
    if tensor_spec.shape[0] != -1 or -1 in tensor_spec.shape[1:]:
        raise MlflowException(
            "For pandas dataframe input, the first dimension of shape must be a variable "
            "dimension and other dimensions must be fixed, but in model signature the shape "
            f"of {'input ' + name if name else 'the unnamed input'} is {tensor_spec.shape}."
        )

    if np.isscalar(pd_series[0]):
        for shape in [(-1,), (-1, 1)]:
            if tensor_spec.shape == shape:
                return _enforce_tensor_spec(
                    np.array(pd_series, dtype=tensor_spec.type).reshape(shape), tensor_spec
                )
        raise MlflowException(
            f"The input pandas dataframe column '{name}' contains scalar "
            "values, which requires the shape to be (-1,) or (-1, 1), but got tensor spec "
            f"shape of {tensor_spec.shape}.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    elif isinstance(pd_series[0], list) and np.isscalar(pd_series[0][0]):
        # If the pandas column contains list type values,
        # in this case, the shape and type information is lost,
        # so do not enforce the shape and type, instead,
        # reshape the array value list to the required shape, and cast value type to
        # required type.
        reshape_err_msg = (
            f"The value in the Input DataFrame column '{name}' could not be converted to the "
            f"expected shape of: '{tensor_spec.shape}'. Ensure that each of the input list "
            "elements are of uniform length and that the data can be coerced to the tensor "
            f"type '{tensor_spec.type}'"
        )
        try:
            flattened_numpy_arr = np.vstack(pd_series.tolist())
            reshaped_numpy_arr = flattened_numpy_arr.reshape(tensor_spec.shape).astype(
                tensor_spec.type
            )
        except ValueError:
            raise MlflowException(reshape_err_msg, error_code=INVALID_PARAMETER_VALUE)
        if len(reshaped_numpy_arr) != len(pd_series):
            raise MlflowException(reshape_err_msg, error_code=INVALID_PARAMETER_VALUE)
        return reshaped_numpy_arr
    elif isinstance(pd_series[0], np.ndarray):
        reshape_err_msg = (
            f"The value in the Input DataFrame column '{name}' could not be converted to the "
            f"expected shape of: '{tensor_spec.shape}'. Ensure that each of the input numpy "
            "array elements are of uniform length and can be reshaped to above expected shape."
        )
        try:
            # Because numpy array includes precise type information, so we don't convert type
            # here, so that in following schema validation we can have strict type check on
            # numpy array column.
            reshaped_numpy_arr = np.vstack(pd_series.tolist()).reshape(tensor_spec.shape)
        except ValueError:
            raise MlflowException(reshape_err_msg, error_code=INVALID_PARAMETER_VALUE)
        if len(reshaped_numpy_arr) != len(pd_series):
            raise MlflowException(reshape_err_msg, error_code=INVALID_PARAMETER_VALUE)
        return reshaped_numpy_arr
    else:
        raise MlflowException(
            "Because the model signature requires tensor spec input, the input "
            "pandas dataframe values should be either scalar value, python list "
            "containing scalar values or numpy array containing scalar values, "
            "other types are not supported.",
            error_code=INVALID_PARAMETER_VALUE,
        )


def _enforce_tensor_schema(pf_input: PyFuncInput, input_schema: Schema):
    """Enforce the input tensor(s) conforms to the model's tensor-based signature."""

    def _is_sparse_matrix(x):
        if not HAS_SCIPY:
            # we can safely assume that it's not a sparse matrix if scipy is not installed
            return False
        return isinstance(x, (csr_matrix, csc_matrix))

    if input_schema.has_input_names():
        if isinstance(pf_input, dict):
            new_pf_input = {}
            for col_name, tensor_spec in zip(input_schema.input_names(), input_schema.inputs):
                if not isinstance(pf_input[col_name], np.ndarray):
                    raise MlflowException(
                        "This model contains a tensor-based model signature with input names,"
                        " which suggests a dictionary input mapping input name to a numpy"
                        f" array, but a dict with value type {type(pf_input[col_name])} was found.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                new_pf_input[col_name] = _enforce_tensor_spec(pf_input[col_name], tensor_spec)
        elif isinstance(pf_input, pd.DataFrame):
            new_pf_input = {}
            for col_name, tensor_spec in zip(input_schema.input_names(), input_schema.inputs):
                pd_series = pf_input[col_name]
                new_pf_input[col_name] = _reshape_and_cast_pandas_column_values(
                    col_name, pd_series, tensor_spec
                )
        else:
            raise MlflowException(
                "This model contains a tensor-based model signature with input names, which"
                " suggests a dictionary input mapping input name to tensor, or a pandas"
                " DataFrame input containing columns mapping input name to flattened list value"
                f" from tensor, but an input of type {type(pf_input)} was found.",
                error_code=INVALID_PARAMETER_VALUE,
            )
    else:
        tensor_spec = input_schema.inputs[0]
        if isinstance(pf_input, pd.DataFrame):
            num_input_columns = len(pf_input.columns)
            if pf_input.empty:
                raise MlflowException("Input DataFrame is empty.")
            elif num_input_columns == 1:
                new_pf_input = _reshape_and_cast_pandas_column_values(
                    None, pf_input[pf_input.columns[0]], tensor_spec
                )
            else:
                if tensor_spec.shape != (-1, num_input_columns):
                    raise MlflowException(
                        "This model contains a model signature with an unnamed input. Since the "
                        "input data is a pandas DataFrame containing multiple columns, "
                        "the input shape must be of the structure "
                        "(-1, number_of_dataframe_columns). "
                        f"Instead, the input DataFrame passed had {num_input_columns} columns and "
                        f"an input shape of {tensor_spec.shape} with all values within the "
                        "DataFrame of scalar type. Please adjust the passed in DataFrame to "
                        "match the expected structure",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                new_pf_input = _enforce_tensor_spec(pf_input.to_numpy(), tensor_spec)
        elif isinstance(pf_input, np.ndarray) or _is_sparse_matrix(pf_input):
            new_pf_input = _enforce_tensor_spec(pf_input, tensor_spec)
        else:
            raise MlflowException(
                "This model contains a tensor-based model signature with no input names,"
                " which suggests a numpy array input or a pandas dataframe input with"
                f" proper column values, but an input of type {type(pf_input)} was found.",
                error_code=INVALID_PARAMETER_VALUE,
            )
    return new_pf_input


def _enforce_schema(pf_input: PyFuncInput, input_schema: Schema):
    """
    Enforces the provided input matches the model's input schema,

    For signatures with input names, we check there are no missing inputs and reorder the inputs to
    match the ordering declared in schema if necessary. Any extra columns are ignored.

    For column-based signatures, we make sure the types of the input match the type specified in
    the schema or if it can be safely converted to match the input schema.

    For tensor-based signatures, we make sure the shape and type of the input matches the shape
    and type specified in model's input schema.
    """

    def _is_scalar(x):
        return np.isscalar(x) or x is None

    if isinstance(pf_input, pd.Series):
        pf_input = pd.DataFrame(pf_input)
    if not input_schema.is_tensor_spec():
        if isinstance(pf_input, (list, np.ndarray, dict, pd.Series, str, bytes)):
            try:
                if isinstance(pf_input, (str, bytes)) or (
                    isinstance(pf_input, dict) and all(map(_is_scalar, pf_input.values()))
                ):
                    pf_input = pd.DataFrame([pf_input])
                elif isinstance(pf_input, dict) and all(
                    isinstance(value, np.ndarray) and value.dtype.type == np.str_
                    # size & shape constraint makes some data batch inference result not
                    # consistent with serving result.
                    and value.size == 1 and value.shape == ()
                    for value in pf_input.values()
                ):
                    # This check is specifically to handle the serving structural cast for
                    # certain inputs for the transformers implementation. Due to the fact that
                    # specific Pipeline types in transformers support passing input data
                    # of the form Dict[str, str] in which the value is a scalar string, model
                    # serving will cast this entry as a numpy array with shape () and size 1.
                    # This is seen as a scalar input when attempting to create a Pandas DataFrame
                    # from such a numpy structure and requires the array to be encapsulated in a
                    # list in order to prevent a ValueError exception for requiring an index
                    # if passing in all scalar values thrown by Pandas.
                    pf_input = pd.DataFrame([pf_input])
                elif isinstance(pf_input, dict) and all(
                    _is_scalar(value)
                    or (isinstance(value, list) and all(isinstance(elem, str) for elem in value))
                    for value in pf_input.values()
                ):
                    pf_input = pd.DataFrame([pf_input])
                elif isinstance(pf_input, dict) and any(
                    isinstance(value, np.ndarray) and value.ndim > 1 for value in pf_input.values()
                ):
                    # Pandas DataFrames can't be constructed with embedded multi-dimensional
                    # numpy arrays. Accordingly, we convert any multi-dimensional numpy
                    # arrays to lists before constructing a DataFrame. This is safe because ColSpec
                    # model signatures do not support array columns, so subsequent validation logic
                    # will result in a clear "incompatible input types" exception. This is
                    # preferable to a pandas DataFrame construction error
                    pf_input = pd.DataFrame(
                        {
                            key: (
                                value.tolist()
                                if (isinstance(value, np.ndarray) and value.ndim > 1)
                                else value
                            )
                            for key, value in pf_input.items()
                        }
                    )
                else:
                    pf_input = pd.DataFrame(pf_input)
            except Exception as e:
                raise MlflowException(
                    "This model contains a column-based signature, which suggests a DataFrame"
                    " input. There was an error casting the input data to a DataFrame:"
                    f" {e}"
                )
        if not isinstance(pf_input, pd.DataFrame):
            raise MlflowException(
                f"Expected input to be DataFrame or list. Found: {type(pf_input).__name__}"
            )

    if input_schema.has_input_names():
        # make sure there are no missing columns
        input_names = input_schema.required_input_names()
        optional_names = input_schema.optional_input_names()
        expected_required_cols = set(input_names)
        actual_cols = set()
        optional_cols = set(optional_names)
        if len(expected_required_cols) == 1 and isinstance(pf_input, np.ndarray):
            # for schemas with a single column, match input with column
            pf_input = {input_names[0]: pf_input}
            actual_cols = expected_required_cols
        elif isinstance(pf_input, pd.DataFrame):
            actual_cols = set(pf_input.columns)
        elif isinstance(pf_input, dict):
            actual_cols = set(pf_input.keys())
        missing_cols = expected_required_cols - actual_cols
        extra_cols = actual_cols - expected_required_cols - optional_cols
        # Preserve order from the original columns, since missing/extra columns are likely to
        # be in same order.
        missing_cols = [c for c in input_names if c in missing_cols]
        extra_cols = [c for c in actual_cols if c in extra_cols]
        if missing_cols:
            message = f"Model is missing inputs {missing_cols}."
            if extra_cols:
                message += f" Note that there were extra inputs: {extra_cols}"
            raise MlflowException(message)
    elif not input_schema.is_tensor_spec():
        # The model signature does not specify column names => we can only verify column count.
        num_actual_columns = len(pf_input.columns)
        if num_actual_columns < len(input_schema.inputs):
            raise MlflowException(
                "Model inference is missing inputs. The model signature declares "
                "{} inputs  but the provided value only has "
                "{} inputs. Note: the inputs were not named in the signature so we can "
                "only verify their count.".format(len(input_schema.inputs), num_actual_columns)
            )
    if input_schema.is_tensor_spec():
        return _enforce_tensor_schema(pf_input, input_schema)
    elif input_schema.has_input_names():
        return _enforce_named_col_schema(pf_input, input_schema)
    else:
        return _enforce_unnamed_col_schema(pf_input, input_schema)


def validate_schema(data: PyFuncInput, expected_schema: Schema) -> None:
    """
    Validate that the input data has the expected schema.

    :param data: Input data to be validated. Supported types are:

                 - pandas.DataFrame
                 - pandas.Series
                 - numpy.ndarray
                 - scipy.sparse.csc_matrix
                 - scipy.sparse.csr_matrix
                 - List[Any]
                 - Dict[str, Any]
                 - str
    :param expected_schema: Expected :py:class:`Schema <mlflow.types.Schema>` of the input data.
    :raises: A :py:class:`mlflow.exceptions.MlflowException`. when the input data does
             not match the schema.

    .. code-block:: python
        :caption: Example usage of validate_schema

        import mlflow.models

        # Suppose you've already got a model_uri
        model_info = mlflow.models.get_model_info(model_uri)
        # Get model signature directly
        model_signature = model_info.signature
        # validate schema
        mlflow.models.validate_schema(input_data, model_signature.inputs)
    """
    _enforce_schema(data, expected_schema)


@experimental
def add_libraries_to_model(model_uri, run_id=None, registered_model_name=None):
    """
    Given a registered model_uri (e.g. models:/<model_name>/<model_version>), this utility
    re-logs the model along with all the required model libraries back to the Model Registry.
    The required model libraries are stored along with the model as model artifacts. In
    addition, supporting files to the model (e.g. conda.yaml, requirements.txt) are modified
    to use the added libraries.

    By default, this utility creates a new model version under the same registered model specified
    by ``model_uri``. This behavior can be overridden by specifying the ``registered_model_name``
    argument.

    :param model_uri: A registered model uri in the Model Registry of the form
                      models:/<model_name>/<model_version/stage/latest>
    :param run_id: The ID of the run to which the model with libraries is logged. If None, the model
                   with libraries is logged to the source run corresponding to model version
                   specified by ``model_uri``; if the model version does not have a source run, a
                   new run created.
    :param registered_model_name: The new model version (model with its libraries) is
                                  registered under the inputted registered_model_name. If None, a
                                  new version is logged to the existing model in the Model Registry.

    .. note::
        This utility only operates on a model that has been registered to the Model Registry.

    .. note::
        The libraries are only compatible with the platform on which they are added. Cross platform
        libraries are not supported.

    .. code-block:: python
        :caption: Example

        # Create and log a model to the Model Registry

        import pandas as pd
        from sklearn import datasets
        from sklearn.ensemble import RandomForestClassifier
        import mlflow
        import mlflow.sklearn
        from mlflow.models import infer_signature

        with mlflow.start_run():
            iris = datasets.load_iris()
            iris_train = pd.DataFrame(iris.data, columns=iris.feature_names)
            clf = RandomForestClassifier(max_depth=7, random_state=0)
            clf.fit(iris_train, iris.target)
            signature = infer_signature(iris_train, clf.predict(iris_train))
            mlflow.sklearn.log_model(
                clf, "iris_rf", signature=signature, registered_model_name="model-with-libs"
            )

        # model uri for the above model
        model_uri = "models:/model-with-libs/1"

        # Import utility
        from mlflow.models.utils import add_libraries_to_model

        # Log libraries to the original run of the model
        add_libraries_to_model(model_uri)

        # Log libraries to some run_id
        existing_run_id = "21df94e6bdef4631a9d9cb56f211767f"
        add_libraries_to_model(model_uri, run_id=existing_run_id)

        # Log libraries to a new run
        with mlflow.start_run():
            add_libraries_to_model(model_uri)

        # Log libraries to a new registered model named 'new-model'
        with mlflow.start_run():
            add_libraries_to_model(model_uri, registered_model_name="new-model")
    """
    import mlflow
    from mlflow.models.wheeled_model import WheeledModel

    if mlflow.active_run() is None:
        if run_id is None:
            run_id = get_model_version_from_model_uri(model_uri).run_id
        with mlflow.start_run(run_id):
            return WheeledModel.log_model(model_uri, registered_model_name)
    else:
        return WheeledModel.log_model(model_uri, registered_model_name)


def get_model_version_from_model_uri(model_uri):
    """
    Helper function to fetch a model version from a model uri of the form
    models:/<model_name>/<model_version/stage/latest>.
    """
    import mlflow
    from mlflow import MlflowClient

    databricks_profile_uri = (
        get_databricks_profile_uri_from_artifact_uri(model_uri) or mlflow.get_registry_uri()
    )
    client = MlflowClient(registry_uri=databricks_profile_uri)
    (name, version) = get_model_name_and_version(client, model_uri)
    return client.get_model_version(name, version)


def _enforce_params_schema(params: Optional[Dict[str, Any]], schema: Optional[ParamSchema]):
    if schema is None:
        if params in [None, {}]:
            return params
        params_info = (
            f"Ignoring provided params: {list(params.keys())}"
            if isinstance(params, dict)
            else "Ignoring invalid params (not a dictionary)."
        )
        _logger.warning(
            "`params` can only be specified at inference time if the model signature "
            f"defines a params schema. This model does not define a params schema. {params_info}",
        )
        return {}
    params = {} if params is None else params
    if not isinstance(params, dict):
        raise MlflowException.invalid_parameter_value(
            f"Parameters must be a dictionary. Got type '{type(params).__name__}'.",
        )
    if not isinstance(schema, ParamSchema):
        raise MlflowException.invalid_parameter_value(
            "Parameters schema must be an instance of ParamSchema. "
            f"Got type '{type(schema).__name__}'.",
        )
    if any(not isinstance(k, str) for k in params.keys()):
        _logger.warning(
            "Keys in parameters should be of type `str`, but received non-string keys."
            "Converting all keys to string..."
        )
        params = {str(k): v for k, v in params.items()}

    allowed_keys = {param.name for param in schema.params}
    ignored_keys = set(params) - allowed_keys
    if ignored_keys:
        _logger.warning(
            f"Unrecognized params {list(ignored_keys)} are ignored for inference. "
            f"Supported params are: {allowed_keys}. "
            "To enable them, please add corresponding schema in ModelSignature."
        )

    params = {k: params[k] for k in params if k in allowed_keys}

    invalid_params = set()
    for param_spec in schema.params:
        if param_spec.name in params:
            try:
                params[param_spec.name] = ParamSpec.validate_param_spec(
                    params[param_spec.name], param_spec
                )
            except MlflowException as e:
                invalid_params.add((param_spec.name, e.message))
        else:
            params[param_spec.name] = param_spec.default

    if invalid_params:
        raise MlflowException.invalid_parameter_value(
            f"Invalid parameters found: {invalid_params!r}",
        )

    return params
