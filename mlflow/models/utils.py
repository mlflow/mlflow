import json
import os
from typing import Union

import numpy as np
import pandas as pd

from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.types.utils import TensorsNotSupportedException
from mlflow.utils.proto_json_utils import NumpyEncoder, _dataframe_from_json, parse_tf_serving_input
from scipy.sparse import csr_matrix, csc_matrix

ModelInputExample = Union[pd.DataFrame, np.ndarray, dict, list, csr_matrix, csc_matrix]


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
        - csc/csr matric: similar to 2 dims numpy array, csc/csr matric are converted to
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

        def _handle_sparse_matrix(x: Union[csr_matrix, csc_matrix]):
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
                if all([_is_scalar(x) for x in input_ex.values()]):
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
                if all([_is_scalar(x) for x in input_ex]):
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


def plot_lines(data_series, xlabel, ylabel, legend_loc=None, line_kwargs=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    if line_kwargs is None:
        line_kwargs = {}

    for label, data_x, data_y in data_series:
        ax.plot(data_x, data_y, label=label, **line_kwargs)

    if legend_loc:
        ax.legend(loc=legend_loc)

    ax.set(xlabel=xlabel, ylabel=ylabel)

    return fig, ax
