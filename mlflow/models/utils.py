import decimal
import json
import os
from typing import Union, Any, Dict, List

import numpy as np
import pandas as pd

from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.store.artifact.utils.models import get_model_name_and_version
from mlflow.utils.annotations import experimental
from mlflow.types import DataType, Schema, TensorSpec
from mlflow.types.utils import TensorsNotSupportedException, clean_tensor_type
from mlflow.utils.proto_json_utils import NumpyEncoder, _dataframe_from_json, parse_tf_serving_input
from mlflow.utils.uri import get_databricks_profile_uri_from_artifact_uri

try:
    from scipy.sparse import csr_matrix, csc_matrix

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

ModelInputExample = Union[pd.DataFrame, np.ndarray, dict, list, "csr_matrix", "csc_matrix"]

PyFuncInput = Union[
    pd.DataFrame,
    pd.Series,
    np.ndarray,
    "csc_matrix",
    "csr_matrix",
    List[Any],
    Dict[str, Any],
]
PyFuncOutput = Union[pd.DataFrame, pd.Series, np.ndarray, list]


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
    example(s) must be jsonable. Mlflow will make the following conversions automatically on behalf
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

        def _handle_dataframe_input(input_ex):
            if isinstance(input_ex, dict):
                if all(_is_scalar(x) for x in input_ex.values()):
                    input_ex = pd.DataFrame([input_ex])
                else:
                    raise TypeError(
                        "Data in the dictionary must be scalar or of type numpy.ndarray"
                    )
            elif isinstance(input_ex, list):
                for i, x in enumerate(input_ex):
                    if isinstance(x, np.ndarray) and len(x.shape) > 1:
                        raise TensorsNotSupportedException(
                            "Row '{0}' has shape {1}".format(i, x.shape)
                        )
                if all(_is_scalar(x) for x in input_ex):
                    input_ex = pd.DataFrame([input_ex], columns=range(len(input_ex)))
                else:
                    input_ex = pd.DataFrame(input_ex)
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
                raise TypeError(
                    "Unexpected type of input_example. Expected one of "
                    "(pandas.DataFrame, numpy.ndarray, dict, list), "
                    "got {}".format(type(input_example))
                )
            result = _handle_dataframe_nans(input_ex).to_dict(orient="split")
            # Do not include row index
            del result["index"]
            if all(input_ex.columns == range(len(input_ex.columns))):
                # No need to write default column index out
                del result["columns"]
            return result

        example_filename = "input_example.json"
        if _is_ndarray(input_example):
            self.data = _handle_ndarray_input(input_example)
            self.info = {
                "artifact_path": example_filename,
                "type": "ndarray",
                "format": "tf-serving",
            }
        elif _is_sparse_matrix(input_example):
            self.data = _handle_sparse_matrix(input_example)
            if isinstance(input_example, csc_matrix):
                example_type = "sparse_matrix_csc"
            else:
                example_type = "sparse_matrix_csr"
            self.info = {
                "artifact_path": example_filename,
                "type": example_type,
            }
        else:
            self.data = _handle_dataframe_input(input_example)
            self.info = {
                "artifact_path": example_filename,
                "type": "dataframe",
                "pandas_orient": "split",
            }

    def save(self, parent_dir_path: str):
        """Save the example as json at ``parent_dir_path``/`self.info['artifact_path']`."""
        with open(os.path.join(parent_dir_path, self.info["artifact_path"]), "w") as f:
            json.dump(self.data, f, cls=NumpyEncoder)


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


def _read_example(mlflow_model: Model, path: str):
    """
    Read example from a model directory. Returns None if there is no example metadata (i.e. the
    model was saved without example). Raises FileNotFoundError if there is model metadata but the
    example file is missing.

    :param mlflow_model: Model metadata.
    :param path: Path to the model directory.
    :return: Input example or None if the model has no example.
    """
    if mlflow_model.saved_input_example_info is None:
        return None
    example_type = mlflow_model.saved_input_example_info["type"]
    if example_type not in ["dataframe", "ndarray", "sparse_matrix_csc", "sparse_matrix_csr"]:
        raise MlflowException(
            "This version of mlflow can not load example of type {}".format(example_type)
        )
    input_schema = mlflow_model.signature.inputs if mlflow_model.signature is not None else None
    path = os.path.join(path, mlflow_model.saved_input_example_info["artifact_path"])
    if example_type == "ndarray":
        return _read_tensor_input_from_json(path, schema=input_schema)
    elif example_type in ["sparse_matrix_csc", "sparse_matrix_csr"]:
        return _read_sparse_matrix_from_json(path, example_type)
    else:
        return _dataframe_from_json(path, schema=input_schema, precise_float=True)


def _read_tensor_input_from_json(path, schema=None):
    with open(path, "r") as handle:
        inp_dict = json.load(handle)
        return parse_tf_serving_input(inp_dict, schema)


def _read_sparse_matrix_from_json(path, example_type):
    with open(path, "r") as handle:
        matrix_data = json.load(handle)
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
    actual_shape = values.shape

    actual_type = values.dtype if isinstance(values, np.ndarray) else values.data.dtype

    if len(expected_shape) != len(actual_shape):
        raise MlflowException(
            "Shape of input {0} does not match expected shape {1}.".format(
                actual_shape, expected_shape
            )
        )
    for expected, actual in zip(expected_shape, actual_shape):
        if expected == -1:
            continue
        if expected != actual:
            raise MlflowException(
                "Shape of input {0} does not match expected shape {1}.".format(
                    actual_shape, expected_shape
                )
            )
    if clean_tensor_type(actual_type) != tensor_spec.type:
        raise MlflowException(
            "dtype of input {0} does not match expected dtype {1}".format(
                values.dtype, tensor_spec.type
            )
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
        return values

    if t == DataType.datetime and values.dtype == object:
        # NB: Pyspark date columns get converted to object when converted to a pandas
        # DataFrame. To respect the original typing, we convert the column to datetime.
        try:
            return values.astype(np.datetime64, errors="raise")
        except ValueError:
            raise MlflowException(
                "Failed to convert column {0} from type {1} to {2}.".format(name, values.dtype, t)
            )
    if t == DataType.double and values.dtype == decimal.Decimal:
        # NB: Pyspark Decimal column get converted to decimal.Decimal when converted to pandas
        # DataFrame. In order to support decimal data training from spark data frame, we add this
        # conversion even we might lose the precision.
        try:
            return pd.to_numeric(values, errors="raise")
        except ValueError:
            raise MlflowException(
                "Failed to convert column {0} from type {1} to {2}.".format(name, values.dtype, t)
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
            "Incompatible input types for column {0}. "
            "Can not safely convert {1} to {2}.{3}".format(name, values.dtype, numpy_type, hint)
        )


def _enforce_col_schema(pfInput: PyFuncInput, input_schema: Schema):
    """Enforce the input columns conform to the model's column-based signature."""
    if input_schema.has_input_names():
        input_names = input_schema.input_names()
    else:
        input_names = pfInput.columns[: len(input_schema.inputs)]
    input_types = input_schema.input_types()
    new_pfInput = pd.DataFrame()
    for i, x in enumerate(input_names):
        new_pfInput[x] = _enforce_mlflow_datatype(x, pfInput[x], input_types[i])
    return new_pfInput


def _enforce_tensor_schema(pfInput: PyFuncInput, input_schema: Schema):
    """Enforce the input tensor(s) conforms to the model's tensor-based signature."""

    def _is_sparse_matrix(x):
        if not HAS_SCIPY:
            # we can safely assume that it's not a sparse matrix if scipy is not installed
            return False
        return isinstance(x, (csr_matrix, csc_matrix))

    if input_schema.has_input_names():
        if isinstance(pfInput, dict):
            new_pfInput = dict()
            for col_name, tensor_spec in zip(input_schema.input_names(), input_schema.inputs):
                if not isinstance(pfInput[col_name], np.ndarray):
                    raise MlflowException(
                        "This model contains a tensor-based model signature with input names,"
                        " which suggests a dictionary input mapping input name to a numpy"
                        " array, but a dict with value type {0} was found.".format(
                            type(pfInput[col_name])
                        )
                    )
                new_pfInput[col_name] = _enforce_tensor_spec(pfInput[col_name], tensor_spec)
        elif isinstance(pfInput, pd.DataFrame):
            new_pfInput = dict()
            for col_name, tensor_spec in zip(input_schema.input_names(), input_schema.inputs):
                new_pfInput[col_name] = _enforce_tensor_spec(
                    np.array(pfInput[col_name], dtype=tensor_spec.type), tensor_spec
                )
        else:
            raise MlflowException(
                "This model contains a tensor-based model signature with input names, which"
                " suggests a dictionary input mapping input name to tensor, but an input of"
                " type {0} was found.".format(type(pfInput))
            )
    else:
        if isinstance(pfInput, pd.DataFrame):
            new_pfInput = _enforce_tensor_spec(pfInput.to_numpy(), input_schema.inputs[0])
        elif isinstance(pfInput, np.ndarray) or _is_sparse_matrix(pfInput):
            new_pfInput = _enforce_tensor_spec(pfInput, input_schema.inputs[0])
        else:
            raise MlflowException(
                "This model contains a tensor-based model signature with no input names,"
                " which suggests a numpy array input, but an input of type {0} was"
                " found.".format(type(pfInput))
            )
    return new_pfInput


def _enforce_schema(pfInput: PyFuncInput, input_schema: Schema):
    """
    Enforces the provided input matches the model's input schema,

    For signatures with input names, we check there are no missing inputs and reorder the inputs to
    match the ordering declared in schema if necessary. Any extra columns are ignored.

    For column-based signatures, we make sure the types of the input match the type specified in
    the schema or if it can be safely converted to match the input schema.

    For tensor-based signatures, we make sure the shape and type of the input matches the shape
    and type specified in model's input schema.
    """
    if isinstance(pfInput, pd.Series):
        pfInput = pd.DataFrame(pfInput)
    if not input_schema.is_tensor_spec():
        if isinstance(pfInput, (list, np.ndarray, dict, pd.Series)):
            try:
                pfInput = pd.DataFrame(pfInput)
            except Exception as e:
                raise MlflowException(
                    "This model contains a column-based signature, which suggests a DataFrame"
                    " input. There was an error casting the input data to a DataFrame:"
                    " {0}".format(str(e))
                )
        if not isinstance(pfInput, pd.DataFrame):
            raise MlflowException(
                "Expected input to be DataFrame or list. Found: %s" % type(pfInput).__name__
            )

    if input_schema.has_input_names():
        # make sure there are no missing columns
        input_names = input_schema.input_names()
        expected_cols = set(input_names)
        actual_cols = set()
        if len(expected_cols) == 1 and isinstance(pfInput, np.ndarray):
            # for schemas with a single column, match input with column
            pfInput = {input_names[0]: pfInput}
            actual_cols = expected_cols
        elif isinstance(pfInput, pd.DataFrame):
            actual_cols = set(pfInput.columns)
        elif isinstance(pfInput, dict):
            actual_cols = set(pfInput.keys())
        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols
        # Preserve order from the original columns, since missing/extra columns are likely to
        # be in same order.
        missing_cols = [c for c in input_names if c in missing_cols]
        extra_cols = [c for c in actual_cols if c in extra_cols]
        if missing_cols:
            message = "Model is missing inputs {0}.".format(missing_cols)
            if extra_cols:
                message += " Note that there were extra inputs: {0}".format(extra_cols)
            raise MlflowException(message)
    elif not input_schema.is_tensor_spec():
        # The model signature does not specify column names => we can only verify column count.
        num_actual_columns = len(pfInput.columns)
        if num_actual_columns < len(input_schema.inputs):
            raise MlflowException(
                "Model inference is missing inputs. The model signature declares "
                "{0} inputs  but the provided value only has "
                "{1} inputs. Note: the inputs were not named in the signature so we can "
                "only verify their count.".format(len(input_schema.inputs), num_actual_columns)
            )

    return (
        _enforce_tensor_schema(pfInput, input_schema)
        if input_schema.is_tensor_spec()
        else _enforce_col_schema(pfInput, input_schema)
    )


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
        from mlflow.models.signature import infer_signature

        with mlflow.start_run():
          iris = datasets.load_iris()
          iris_train = pd.DataFrame(iris.data, columns=iris.feature_names)
          clf = RandomForestClassifier(max_depth=7, random_state=0)
          clf.fit(iris_train, iris.target)
          mlflow.sklearn.log_model(clf, "iris_rf", registered_model_name="model-with-libs")

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
    model_version = client.get_model_version(name, version)
    return model_version
