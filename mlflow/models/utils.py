import json
import os
from typing import Union

import numpy as np
import pandas as pd

from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.types.utils import TensorsNotSupportedException
from mlflow.utils.proto_json_utils import NumpyEncoder, _dataframe_from_json

ModelInputExample = Union[pd.DataFrame, np.ndarray, dict, list]


class _Example(object):
    """
    Represents an input example for MLflow model.

    Contains jsonable data that can be saved with the model and meta data about the exported format
    that can be saved with :py:class:`Model <mlflow.models.Model>`.

    The _Example is created from example data provided by user. The example(s) can be provided as
    pandas.DataFrame, numpy.ndarray, python dictionary or python list. The assumption is that the
    example is a DataFrame-like dataset with jsonable elements (see storage format section below).

    NOTE: Multidimensional (>2d) arrays (aka tensors) are not supported at this time.

    NOTE: If the example is 1 dimensional (e.g. dictionary of str -> scalar, or a list of scalars),
    the assumption is that it is a single row of data (rather than a single column).

    Metadata:

    The _Example metadata contains the following information:
        - artifact_path: Relative path to the serialized example within the model directory.
        - type: Type of example data provided by the user. E.g. dataframe.
        - pandas_orient: For dataframes, this attribute specifies how is the dataframe encoded in
                         json. For example, "split" value signals that the data is stored as object
                         with columns and data attributes.

    Storage Format:

    The examples are stored as json for portability and readability. Therefore, the contents of the
    example(s) must be jsonable. Mlflow will make the following conversions automatically on behalf
    of the user:

        - binary values: :py:class:`bytes` or :py:class:`bytearray` are converted to base64
          encoded strings.
        - numpy types: Numpy types are converted to the corresponding python types or their closest
          equivalent.
    """

    def __init__(self, input_example: ModelInputExample):
        def _is_scalar(x):
            return np.isscalar(x) or x is None

        if isinstance(input_example, dict):
            for x, y in input_example.items():
                if isinstance(y, np.ndarray) and len(y.shape) > 1:
                    raise TensorsNotSupportedException(
                        "Column '{0}' has shape {1}".format(x, y.shape))

            if all([_is_scalar(x) for x in input_example.values()]):
                input_example = pd.DataFrame([input_example])
            else:
                input_example = pd.DataFrame.from_dict(input_example)
        elif isinstance(input_example, list):
            for i, x in enumerate(input_example):
                if isinstance(x, np.ndarray) and len(x.shape) > 1:
                    raise TensorsNotSupportedException("Row '{0}' has shape {1}".format(i, x.shape))
            if all([_is_scalar(x) for x in input_example]):
                input_example = pd.DataFrame([input_example], columns=range(len(input_example)))
            else:
                input_example = pd.DataFrame(input_example)
        elif isinstance(input_example, np.ndarray):
            if len(input_example.shape) > 2:
                raise TensorsNotSupportedException("Input array has shape {}".format(
                    input_example.shape))
            input_example = pd.DataFrame(input_example)
        elif not isinstance(input_example, pd.DataFrame):
            try:
                import pyspark.sql.dataframe
                if isinstance(input_example, pyspark.sql.dataframe.DataFrame):
                    raise MlflowException("Examples can not be provided as Spark Dataframe. "
                                          "Please make sure your example is of a small size and "
                                          "turn it into a pandas DataFrame by calling toPandas "
                                          "method.")
            except ImportError:
                pass
            raise TypeError("Unexpected type of input_example. Expected one of "
                            "(pandas.DataFrame, numpy.ndarray, dict, list), "
                            "got {}".format(type(input_example)))
        example_filename = "input_example.json"
        self.data = input_example.to_dict(orient="split")
        # Do not include row index
        del self.data["index"]
        if all(input_example.columns == range(len(input_example.columns))):
            # No need to write default column index out
            del self.data["columns"]
        self.info = {"artifact_path": example_filename,
                     "type": "dataframe",
                     "pandas_orient": "split"}

    def save(self, parent_dir_path: str):
        """Save the example as json at ``parent_dir_path``/`self.info['artifact_path']`.  """
        with open(os.path.join(parent_dir_path, self.info["artifact_path"]), "w") as f:
            json.dump(self.data, f, cls=NumpyEncoder)


def _save_example(mlflow_model: Model, input_example: ModelInputExample, path: str):
    """
    Save example to a file on the given path and updates passed Model with example metadata.

    The metadata is a dictionary with the following fields:
      - 'artifact_path': example path relative to the model directory.
      - 'type': Type of example. Currently the only supported value is 'dataframe'
      - 'pandas_orient': Determines the json encoding for dataframe examples in terms of pandas
                         orient convention. Defaults to 'split'.
    :param mlflow_model: Model metadata that will get updated with the example metadata.
    :param path: Where to store the example file. Should be model the model directory.
    """
    example = _Example(input_example)
    example.save(path)
    mlflow_model.saved_input_example_info = example.info


def _read_example(mlflow_model: Model, path: str):
    """
    Read example from a model directory. Returns None if there is no example metadata (i.e. the
    model was saved without example). Raises IO Exception if there is model metadata but the example
    file is missing.

    :param mlflow_model: Model metadata.
    :param path: Path to the model directory.
    :return: Input example or None if the model has no example.
    """
    if mlflow_model.saved_input_example_info is None:
        return None
    example_type = mlflow_model.saved_input_example_info["type"]
    if example_type != "dataframe":
        raise MlflowException("This version of mlflow can not load example of type {}".format(
            example_type))
    input_schema = mlflow_model.signature.inputs if mlflow_model.signature is not None else None
    path = os.path.join(path, mlflow_model.saved_input_example_info["artifact_path"])
    return _dataframe_from_json(path, schema=input_schema, precise_float=True)
