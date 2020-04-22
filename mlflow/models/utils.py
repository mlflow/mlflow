import base64
import json
import os
from typing import TypeVar

import numpy as np
import pandas as pd

from mlflow.exceptions import MlflowException
from mlflow.utils.proto_json_utils import NumpyEncoder

ModelInputExample = TypeVar('ModelInputExample', pd.DataFrame, np.ndarray, dict, list)


class TensorsNotSupportedException(MlflowException):
    def __init__(self, msg):
        super().__init__("Multidimensional arrays (aka tensors) are not supported. "
                         "{}".format(msg))


def save_example(path: str, input_example: ModelInputExample) -> str:
    """
    Save MLflow example into a file on a given path and return the resulting filename.

    The example(s) can be provided as :py:class:`pandas.DataFrame`, :py:class:`numpy.ndarray',
    python dictionary or python list. The assumption is that the example is a DataFrame-like
    dataset with jsonable elements (see storage format section below).

    NOTE: Multidimensional (>2d) arrays (aka tensors) are not supported at this time.

    NOTE: If the example is 1 dimensional (e.g. dictionary of str -> scalar, or a list of scalars),
    the assumption is that it is a single row of data (rather than 1 column).

    Storage Format
    ==============
    The examples are stored as json for portability and readability. Therefore, the contents of the
    example(s) must be jsonable. Mlflow will make the following conversions automatically on behalf
    of the user:

    - binary values: :py:class`bytes` or :py:class`bytearray` are converted to base64
      encoded strings.
    - numpy types: Numpy types are converted to the corresponding python types or their closest
      equivalent.

    The json output is formatted according to pandas orient='split' convention (we omit index).
    The output is a json object with the following attributes:
     - columns: list of column names. Columns are not included if there are no column names or if
                the column names are ordered sequence 0..N where N is the number of columns in the
                dataset.
    - data: Json array with the data organized row-wise.

    :param path: Path where to store the example.
    :param input_example: Data with the input example(s). Expected to be a DataFrame-like
                          (2 dimensional) dataset with jsonable elements.
    :return: Filename of the stored example.
    """

    def _is_scalar(x):
        return np.isscalar(x) or x is None

    if isinstance(input_example, dict):
        for x, y in input_example.items():
            if isinstance(y, np.ndarray) and len(y.shape) > 1:
                raise TensorsNotSupportedException("Column '{0}' has shape {1}".format(x, y.shape))

        if all([_is_scalar(x) for x in input_example.values()]):
            input_example = pd.DataFrame([input_example])
        else:
            input_example = pd.DataFrame.from_dict(input_example)
    elif isinstance(input_example, list):
        for i, x in enumerate(input_example):
            if isinstance(x, np.ndarray) and len(x.shape) > 1:
                raise TensorsNotSupportedException("Row '{0}' has shape {1}".format(i, x.shape))
        if all([_is_scalar(x) for x in input_example]):
            input_example = pd.DataFrame([input_example])
        else:
            input_example = pd.DataFrame(input_example)
    elif isinstance(input_example, np.ndarray):
        if len(input_example.shape) > 2:
            raise TensorsNotSupportedException("Input array has shape {}".format(
                input_example.shape))
        input_example = pd.DataFrame(input_example)
    elif not isinstance(input_example, pd.DataFrame):
        raise TypeError("Unexpected type of input_example. Expected one of "
                        "(pandas.DataFrame, numpy.ndarray, dict, list), got {}".format(
            type(input_example)))

    example_filename = "input_dataframe_example.json"
    res = input_example.to_dict(orient="split")
    # Do not include row index
    del res["index"]
    if all(input_example.columns == range(len(input_example.columns))):
        # No need to write default column index out
        del res["columns"]

    with open(os.path.join(path, example_filename), "w") as f:
        json.dump(res, f, cls=NumpyEncoder)
    return example_filename
